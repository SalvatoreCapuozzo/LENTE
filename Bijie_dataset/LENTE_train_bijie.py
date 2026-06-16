import os
import time
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils_bijie import (
    BijieDataset, build_train_val_test_subsets, TransformSubset,
    SAM2TinyAdapterModel, criterion, get_best_mask, calculate_metrics, scale_boxes,
    save_checkpoint, load_checkpoint
)

def load_config():
    # Simulating a config file load for architectural parity with L4S
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_root": r"/datadrive/LENTE/Bijie_dataset/Bijie-landslide-dataset",
        "sam_ckpt_path": "facebook/sam2-hiera-tiny",
        "checkpoint_name": "BEST_sam2tiny_adapter_bijie.pth",
        "epochs": 100,
        "patience": 15,
        "batch_size": 8,
        "target_size": 1024,
        "load_size": 256,
        "val_fraction": 0.20,
        "test_fraction": 0.20,
        "random_seed": 42,
        "aug_copies": 2,
        "adapter_dim": 64,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4
    }

def train():
    cfg = load_config()
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    # --- Generate timestamped folder name (YYMMDD_hhmm) ---
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    save_dir = timestamp 
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt_filename = os.path.basename(cfg["checkpoint_name"])
    cfg["checkpoint_name"] = os.path.join(save_dir, ckpt_filename)

    full_ds = BijieDataset(cfg["dataset_root"], load_size=cfg["load_size"])
    train_subset, val_subset, _, pos_weight = build_train_val_test_subsets(
        full_ds, cfg["val_fraction"], cfg["test_fraction"], cfg["random_seed"]
    )

    offline_aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    train_trans = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), ToTensorV2()])
    val_trans   = A.Compose([ToTensorV2()])

    train_pos_indices = [i for i in train_subset.indices if i in set(full_ds.positive_indices)]
    full_ds.add_augmented_samples(indices=train_pos_indices, transform=offline_aug, n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    
    aug_start   = len(full_ds.data) - len(train_pos_indices) * cfg["aug_copies"]
    aug_indices = list(range(aug_start, len(full_ds.data)))
    train_subset = Subset(full_ds, list(train_subset.indices) + aug_indices)

    train_ds = TransformSubset(train_subset, train_trans)
    val_ds   = TransformSubset(val_subset,   val_trans)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, pin_memory=True, num_workers=0)

    model = SAM2TinyAdapterModel(sam_ckpt_path=cfg["sam_ckpt_path"], adapter_dim=cfg["adapter_dim"]).to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(device=DEVICE)

    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, cfg["checkpoint_name"], DEVICE)
    
    patience_counter = 0
    total_start_time = time.time() 

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start_time = time.time()
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['epochs']} [TRAIN]")

        for batch in pbar:
            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                loss = criterion(preds, masks, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iou_fg, _, miou, _, _, _ = calculate_metrics(preds.detach(), masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}", miou=f"{miou:.4f}")

        model.eval()
        v_loss = v_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[VAL]"):
                imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
                orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
                masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
                boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

                with torch.amp.autocast(device_type=DEVICE):
                    all_preds, scores = model(imgs, boxes)
                    preds = get_best_mask(all_preds, scores)
                    v_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                
                iou_fg, _, _, _, _, _ = calculate_metrics(preds, masks)
                v_iou += iou_fg

        avg_val_loss, avg_val_iou = v_loss / len(val_loader), v_iou / len(val_loader)
        scheduler.step(avg_val_loss)
        
        epoch_mins, epoch_secs = divmod(time.time() - epoch_start_time, 60)
        print(f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, cfg["checkpoint_name"])
            patience_counter = 0
            print(f"--> Saved new best checkpoint to: {cfg['checkpoint_name']}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]: break

    print(f"\nTraining completed in {int((time.time() - total_start_time) // 60)}m.")

if __name__ == "__main__":
    train()
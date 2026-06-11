import json
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from our definition file
from LENTE_utils import (
    LandslideH5Dataset, build_train_val_test_subsets, TransformSubset,
    LENTE, criterion, get_best_mask, calculate_metrics, scale_boxes,
    save_checkpoint, load_checkpoint
)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    full_ds = LandslideH5Dataset(cfg["dataset_root"])
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

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    model = LENTE(sam_ckpt_path=cfg["sam_ckpt_path"], adapter_dim=cfg["adapter_dim"]).to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(device=DEVICE)

    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, cfg["checkpoint_name"], DEVICE)
    history = []
    patience_counter = 0

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t_loss = t_iou = t_miou = 0.0
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
            t_loss += loss.item(); t_iou += iou_fg; t_miou += miou
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}", miou=f"{miou:.4f}")

        # VAL Loop
        model.eval()
        v_loss = v_iou = v_miou = 0.0
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
                
                iou_fg, _, miou, _, _, _ = calculate_metrics(preds, masks)
                v_iou += iou_fg; v_miou += miou

        avg_val_loss, avg_val_iou = v_loss / len(val_loader), v_iou / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, cfg["checkpoint_name"])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]: break

if __name__ == "__main__":
    train()
import os
import re
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_train_bijie import load_config
from LENTE_utils_bijie import (
    BijieDataset, build_train_val_test_subsets, TransformSubset,
    SAM2TinyAdapterModel, criterion, get_best_mask, calculate_metrics, scale_boxes
)

def get_latest_checkpoint_path(default_ckpt_path, base_dir="."):
    pattern = re.compile(r"^\d{6}_\d{4}$")
    valid_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and pattern.match(d)]
    
    if not valid_dirs: return default_ckpt_path
    return os.path.join(sorted(valid_dirs, reverse=True)[0], os.path.basename(default_ckpt_path))

def adoptable_inference_function(model, x_tensor: torch.Tensor, boxes: torch.Tensor, target_size: int = 1024):
    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)
    orig_h, orig_w = x_tensor.shape[-2], x_tensor.shape[-1]
    
    x_tensor = F.interpolate(x_tensor, (target_size, target_size), mode="bilinear", align_corners=False)
    boxes = scale_boxes(boxes.to(device), orig_h, orig_w, target_size) if boxes is not None else torch.tensor([[0., 0., float(target_size), float(target_size)]]).to(device)

    model.eval()
    infer_start = time.time()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        all_preds, scores = model(x_tensor, boxes)
        preds = F.interpolate(get_best_mask(all_preds, scores).float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
        binary_mask = (torch.sigmoid(preds) > 0.5).long()
            
    print(f"Standalone Inference Time: {time.time() - infer_start:.4f} seconds")
    return binary_mask

def test():
    total_start_time = time.time()
    cfg = load_config()
    DEVICE, TARGET_SIZE = cfg["device"], cfg["target_size"]

    full_ds = BijieDataset(cfg["dataset_root"], load_size=cfg["load_size"])
    _, _, test_subset, pos_weight = build_train_val_test_subsets(full_ds, cfg["val_fraction"], cfg["test_fraction"], cfg["random_seed"])

    test_loader = DataLoader(TransformSubset(test_subset, A.Compose([ToTensorV2()])), batch_size=cfg["batch_size"], shuffle=False, pin_memory=True, num_workers=0)

    latest_ckpt = get_latest_checkpoint_path(cfg["checkpoint_name"])
    print(f"--> Loading model from: {latest_ckpt}")
    
    model = SAM2TinyAdapterModel(sam_ckpt_path=cfg["sam_ckpt_path"], adapter_dim=cfg["adapter_dim"]).to(DEVICE)
    if os.path.exists(latest_ckpt): model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE)["model_state_dict"], strict=False)
    model.eval()

    te_loss = te_iou = te_miou = te_f1 = total_infer_time = total_images = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TESTING"):
            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            total_images += imgs.size(0)
            
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            batch_infer_start = time.time()
            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                te_loss += criterion(preds, masks, pos_weight=pos_weight).item()
            total_infer_time += (time.time() - batch_infer_start)

            iou_fg, _, miou, f1, _, _ = calculate_metrics(preds, masks)
            te_iou += iou_fg; te_miou += miou; te_f1 += f1

    nte = len(test_loader)
    print("\n" + "=" * 40 + "\nFINAL TEST RESULTS (Bijie)\n" + "=" * 40)
    print(f"Test Loss: {te_loss / nte:.4f} | Test IoU: {te_iou / nte:.4f} | mIoU: {te_miou / nte:.4f} | F1: {te_f1 / nte:.4f}")
    print(f"Per-Image Inference Time: {(total_infer_time / total_images) * 1000:.2f} ms")
    print(f"Overall Testing Time: {int((time.time() - total_start_time) // 60)}m\n" + "=" * 40)

if __name__ == "__main__":
    test()
import os
import re
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    LandslideH5Dataset, build_train_val_test_subsets, TransformSubset,
    LENTE, criterion, get_best_mask, calculate_metrics, scale_boxes
)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- NEW: Helper function to find the latest timestamped folder ---
def get_latest_checkpoint_path(default_ckpt_path, base_dir="."):
    """Scans base_dir for YYMMDD_hhmm folders and returns the path to the latest checkpoint."""
    # Regex pattern to match YYMMDD_hhmm exactly (6 digits _ 4 digits)
    pattern = re.compile(r"^\d{6}_\d{4}$")
    
    valid_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and pattern.match(d)]
    
    if not valid_dirs:
        print(f"No timestamped folders found. Falling back to config path: {default_ckpt_path}")
        return default_ckpt_path
    
    # Because of the YYMMDD_hhmm format, alphabetical sorting is identical to chronological sorting
    latest_dir = sorted(valid_dirs, reverse=True)[0]
    ckpt_filename = os.path.basename(default_ckpt_path)
    latest_path = os.path.join(latest_dir, ckpt_filename)
    
    return latest_path

# ============================================================
# FUNCTION FOR YOU TO ADOPT (STANDALONE INFERENCE)
# ============================================================
def adoptable_inference_function(model: LENTE, x_tensor: torch.Tensor, boxes: torch.Tensor, target_size: int = 1024):
    """
    Function to run a single image tensor through the LENTE model.
    x_tensor: Shape (1, 14, H, W) containing all L4S channels
    boxes: Shape (1, 4) or None
    """
    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)
    
    orig_h, orig_w = x_tensor.shape[-2], x_tensor.shape[-1]
    x_tensor = F.interpolate(x_tensor, (target_size, target_size), mode="bilinear", align_corners=False)
    
    if boxes is not None:
        boxes = boxes.to(device)
        boxes = scale_boxes(boxes, orig_h, orig_w, target_size)
    else:
        # Default box fallback if none provided
        boxes = torch.tensor([[0., 0., float(target_size), float(target_size)]]).to(device)

    model.eval()
    
    # --- NEW: Track pure inference time ---
    infer_start = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            all_preds, scores = model(x_tensor, boxes)
            preds = get_best_mask(all_preds, scores)
            
            # Interpolate back to original resolution
            preds = F.interpolate(preds.float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
            binary_mask = (torch.sigmoid(preds) > 0.5).long()
            
    infer_time = time.time() - infer_start
    print(f"Standalone Inference Time: {infer_time:.4f} seconds")
            
    return binary_mask


# ============================================================
# STANDARD BATCH TEST SCRIPT
# ============================================================
def test():
    # --- NEW: Start overall script timer ---
    total_start_time = time.time()
    
    cfg = load_config()
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    print("Loading test dataset setup...")
    full_ds = LandslideH5Dataset(cfg["dataset_root"])
    _, _, test_subset, pos_weight = build_train_val_test_subsets(
        full_ds, cfg["val_fraction"], cfg["test_fraction"], cfg["random_seed"]
    )

    test_trans = A.Compose([ToTensorV2()])
    test_ds = TransformSubset(test_subset, test_trans)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    # --- NEW: Automatically resolve the latest weights path ---
    latest_ckpt_path = get_latest_checkpoint_path(cfg["checkpoint_name"])
    print(f"--> Loading LENTE model from: {latest_ckpt_path}")
    
    model = LENTE(sam_ckpt_path=cfg["sam_ckpt_path"], adapter_dim=cfg["adapter_dim"]).to(DEVICE)
    ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    te_loss = te_iou = te_miou = te_f1 = 0.0
    
    # --- NEW: Variables for timing batch testing ---
    total_infer_time = 0.0
    total_images_processed = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TESTING"):
            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            
            batch_size = imgs.size(0)
            total_images_processed += batch_size
            
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            # --- NEW: Timer for model forward pass only ---
            batch_infer_start = time.time()
            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                te_loss += criterion(preds, masks, pos_weight=pos_weight).item()
            
            # Add batch time to total inference time
            total_infer_time += (time.time() - batch_infer_start)

            iou_fg, _, miou, f1, _, _ = calculate_metrics(preds, masks)
            te_iou += iou_fg; te_miou += miou; te_f1 += f1

    # --- NEW: Calculate final times ---
    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)
    
    # Avoid division by zero just in case the dataset is empty
    avg_per_image_time = (total_infer_time / total_images_processed) if total_images_processed > 0 else 0

    nte = len(test_loader)
    print("\n" + "=" * 40)
    print("FINAL TEST RESULTS")
    print("=" * 40)
    print(f"Test Loss: {te_loss / nte:.4f}")
    print(f"Test IoU:  {te_iou / nte:.4f}")
    print(f"Test mIoU: {te_miou / nte:.4f}")
    print(f"Test F1:   {te_f1 / nte:.4f}")
    print("-" * 40)
    print(f"Per-Image Inference Time: {avg_per_image_time * 1000:.2f} ms")
    print(f"Overall Testing Time:     {int(total_mins)}m {int(total_secs)}s")
    print("=" * 40)

if __name__ == "__main__":
    test()
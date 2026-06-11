import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# Importing from your updated utility file name
from LENTE_utils import (
    LandslideH5Dataset, build_train_val_test_subsets, TransformSubset,
    LENTE, get_best_mask, scale_boxes
)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def visualize_test_set():
    cfg = load_config()
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]
    
    # Read the parameter, default to 5 if someone forgets to add it to the yaml
    num_samples = cfg.get("viz_num_samples", 5) 

    print("Loading dataset partitions...")
    full_ds = LandslideH5Dataset(cfg["dataset_root"])
    _, _, test_subset, _ = build_train_val_test_subsets(
        full_ds, cfg["val_fraction"], cfg["test_fraction"], cfg["random_seed"]
    )

    # --- NEW LOGIC: Handle 0 or specific number ---
    if num_samples == 0:
        print("viz_num_samples is 0: Visualizing ALL images in the test set.")
        viz_indices = list(test_subset.indices)
    else:
        print(f"viz_num_samples is {num_samples}: Selecting samples (prioritizing positive targets)...")
        pos_test_indices = [i for i in test_subset.indices if i in set(full_ds.positive_indices)]
        
        if len(pos_test_indices) < num_samples:
            print(f"Warning: Only {len(pos_test_indices)} positive samples found. Filling the rest with negative patches.")
            viz_indices = list(test_subset.indices)[:num_samples]
        else:
            viz_indices = pos_test_indices[:num_samples]

    # Reconstruct the evaluation subset based on our selection
    viz_subset = torch.utils.data.Subset(full_ds, viz_indices)
    test_trans = A.Compose([ToTensorV2()])
    viz_ds = TransformSubset(viz_subset, test_trans)
    viz_loader = DataLoader(viz_ds, batch_size=1, shuffle=False)

    print(f"Loading weights into LENTE from {cfg['checkpoint_name']}...")
    model = LENTE(sam_ckpt_path=cfg["sam_ckpt_path"], adapter_dim=cfg["adapter_dim"]).to(DEVICE)
    ckpt = torch.load(cfg["checkpoint_name"], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    os.makedirs("visualizations", exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(viz_loader):
            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]

            imgs_resized = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            boxes_scaled = scale_boxes(boxes.clone(), orig_h, orig_w, TARGET_SIZE)

            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs_resized, boxes_scaled)
                preds_resized = get_best_mask(all_preds, scores)
                
                preds = F.interpolate(preds_resized.float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
                prob_map = torch.sigmoid(preds).squeeze().cpu().numpy()
                pred_mask = (prob_map > 0.5).astype(np.uint8)

            img_np = imgs.squeeze().cpu().numpy()
            gt_np = masks.squeeze().cpu().numpy().astype(np.uint8)

            rgb = img_np[[3, 2, 1], :, :].transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

            error_map = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
            error_map[(pred_mask == 1) & (gt_np == 1)] = [0, 255, 0]   # TP: Green
            error_map[(pred_mask == 1) & (gt_np == 0)] = [255, 0, 0]   # FP: Red
            error_map[(pred_mask == 0) & (gt_np == 1)] = [255, 255, 0] # FN: Yellow

            # Matplotlib Grid Plotting
            # Increased height from 5 to 6 to give titles more room
            fig, axes = plt.subplots(1, 4, figsize=(20, 6))
            
            axes[0].imshow(rgb)
            axes[0].set_title("RGB Composite (B4, B3, B2)", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(gt_np, cmap="gray")
            axes[1].set_title("Ground Truth Mask", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(prob_map, cmap="jet")
            axes[2].set_title("LENTE Probability Map", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(error_map)
            # Shortened the text layout and slightly reduced the font size
            axes[3].set_title("Error Profile\n(Green: TP | Red: FP | Yellow: FN)", fontsize=11)
            axes[3].axis("off")

            # Added padding to prevent the titles from getting cropped
            plt.tight_layout(pad=2.0)
            
            # Grab the actual original index for a cleaner filename reference
            original_idx = viz_indices[i] 
            out_path = f"visualizations/sample_idx{original_idx}.png"
            
            plt.savefig(out_path, dpi=150, bbox_inches='tight') # Added bbox_inches='tight' for safety
            plt.close()
            
            # Use carriage return to overwrite the print line if processing hundreds of images
            print(f"\rSaved {i+1}/{len(viz_indices)}: {out_path}", end="")
            
    print("\nVisualization generation complete!")

if __name__ == "__main__":
    visualize_test_set()
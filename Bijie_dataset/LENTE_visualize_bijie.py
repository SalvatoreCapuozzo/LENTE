import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from LENTE_train_bijie import load_config
from LENTE_test_bijie import get_latest_checkpoint_path
from LENTE_utils_bijie import (
    BijieDataset, build_train_val_test_subsets, TransformSubset,
    SAM2TinyAdapterModel, get_best_mask, scale_boxes
)

def visualize_test_set():
    cfg = load_config()
    DEVICE, TARGET_SIZE = cfg["device"], cfg["target_size"]
    num_samples = cfg.get("viz_num_samples", 5) 

    full_ds = BijieDataset(cfg["dataset_root"], load_size=cfg["load_size"])
    _, _, test_subset, _ = build_train_val_test_subsets(full_ds, cfg["val_fraction"], cfg["test_fraction"], cfg["random_seed"])

    if num_samples == 0:
        viz_indices = list(test_subset.indices)
    else:
        pos_test_indices = [i for i in test_subset.indices if i in set(full_ds.positive_indices)]
        viz_indices = pos_test_indices[:num_samples] if len(pos_test_indices) >= num_samples else list(test_subset.indices)[:num_samples]

    viz_loader = DataLoader(TransformSubset(torch.utils.data.Subset(full_ds, viz_indices), A.Compose([ToTensorV2()])), batch_size=1, shuffle=False)

    latest_ckpt = get_latest_checkpoint_path(cfg["checkpoint_name"])
    model = SAM2TinyAdapterModel(sam_ckpt_path=cfg["sam_ckpt_path"], adapter_dim=cfg["adapter_dim"]).to(DEVICE)
    if os.path.exists(latest_ckpt): model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE)["model_state_dict"], strict=False)
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
                preds = F.interpolate(get_best_mask(all_preds, scores).float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
                prob_map = torch.sigmoid(preds).squeeze().cpu().numpy()
                pred_mask = (prob_map > 0.5).astype(np.uint8)

            img_np, gt_np = imgs.squeeze().cpu().numpy(), masks.squeeze().cpu().numpy().astype(np.uint8)

            # Bijie natively structures RGB on the first 3 channels
            rgb = (img_np[:3, :, :].transpose(1, 2, 0) - img_np[:3, :, :].min()) / (img_np[:3, :, :].max() - img_np[:3, :, :].min() + 1e-8)

            error_map = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
            error_map[(pred_mask == 1) & (gt_np == 1)] = [0, 255, 0]   
            error_map[(pred_mask == 1) & (gt_np == 0)] = [255, 0, 0]   
            error_map[(pred_mask == 0) & (gt_np == 1)] = [255, 255, 0] 

            fig, axes = plt.subplots(1, 4, figsize=(20, 6))
            axes[0].imshow(rgb); axes[0].set_title("RGB Composite"); axes[0].axis("off")
            axes[1].imshow(gt_np, cmap="gray"); axes[1].set_title("Ground Truth Mask"); axes[1].axis("off")
            axes[2].imshow(prob_map, cmap="jet"); axes[2].set_title("SAM2-Tiny Probability"); axes[2].axis("off")
            axes[3].imshow(error_map); axes[3].set_title("Error Profile\n(Green: TP | Red: FP | Yellow: FN)"); axes[3].axis("off")

            plt.tight_layout(pad=2.0)
            out_path = f"visualizations/bijie_sample_idx{viz_indices[i]}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\rSaved {i+1}/{len(viz_indices)}: {out_path}", end="")
            
    print("\nVisualization generation complete!")

if __name__ == "__main__":
    visualize_test_set()
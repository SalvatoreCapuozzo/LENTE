import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    build_model, get_best_mask, scale_boxes
)


def _make_rgb(img_np, cfg):
    """Compone l'RGB dai canali indicati nel config e applica lo stretch."""
    ch = cfg.get("rgb_channels", [0, 1, 2])
    rgb = img_np[ch, :, :].transpose(1, 2, 0)
    if cfg.get("rgb_stretch", "clip") == "minmax":
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    else:
        rgb = np.clip(rgb, 0, 1)
    return rgb


def visualize_test_set(cfg):
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]
    num_samples = cfg.get("viz_num_samples", 5)

    print("Loading dataset partitions...")
    full_ds = build_dataset(cfg)
    _, _, test_subset, _ = build_train_val_test_subsets(full_ds, cfg)

    if num_samples == 0:
        print("viz_num_samples = 0: visualizzo TUTTO il test set.")
        viz_indices = list(test_subset.indices)
    else:
        pos_test_indices = [i for i in test_subset.indices if i in set(full_ds.positive_indices)]
        if len(pos_test_indices) < num_samples:
            print(f"Solo {len(pos_test_indices)} positivi: completo con patch negativi.")
            viz_indices = list(test_subset.indices)[:num_samples]
        else:
            viz_indices = pos_test_indices[:num_samples]

    viz_ds = TransformSubset(Subset(full_ds, viz_indices), A.Compose([ToTensorV2()]))
    viz_loader = DataLoader(viz_ds, batch_size=1, shuffle=False)

    print(f"Loading weights into LENTE from {cfg['checkpoint_name']}...")
    model = build_model(cfg, DEVICE)
    ckpt = torch.load(cfg["checkpoint_name"], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    out_dir = f"visualizations_{cfg['dataset_name']}"
    os.makedirs(out_dir, exist_ok=True)

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
            rgb = _make_rgb(img_np, cfg)

            error_map = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
            error_map[(pred_mask == 1) & (gt_np == 1)] = [0, 255, 0]
            error_map[(pred_mask == 1) & (gt_np == 0)] = [255, 0, 0]
            error_map[(pred_mask == 0) & (gt_np == 1)] = [255, 255, 0]

            fig, axes = plt.subplots(1, 4, figsize=(20, 6))
            axes[0].imshow(rgb);        axes[0].set_title("RGB Composite", fontsize=12); axes[0].axis("off")
            axes[1].imshow(gt_np, cmap="gray"); axes[1].set_title("Ground Truth Mask", fontsize=12); axes[1].axis("off")
            axes[2].imshow(prob_map, cmap="jet"); axes[2].set_title("LENTE Probability Map", fontsize=12); axes[2].axis("off")
            axes[3].imshow(error_map);  axes[3].set_title("Error Profile\n(Green: TP | Red: FP | Yellow: FN)", fontsize=11); axes[3].axis("off")
            plt.tight_layout(pad=2.0)

            out_path = f"{out_dir}/sample_idx{viz_indices[i]}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\rSaved {i+1}/{len(viz_indices)}: {out_path}", end="")

    print("\nVisualization generation complete!")


if __name__ == "__main__":
    visualize_test_set(load_config())

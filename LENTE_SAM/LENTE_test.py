import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    build_model, criterion, get_best_mask, calculate_metrics, scale_boxes
)


def adoptable_inference_function(model, x_tensor, boxes, target_size=1024):
    """
    Single-image inference. x_tensor: (1, C, H, W) con C = n_optical + 2.
    boxes: (1,4) o None.
    """
    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)
    orig_h, orig_w = x_tensor.shape[-2], x_tensor.shape[-1]
    x_tensor = F.interpolate(x_tensor, (target_size, target_size), mode="bilinear", align_corners=False)
    if boxes is not None:
        boxes = scale_boxes(boxes.to(device), orig_h, orig_w, target_size)
    else:
        boxes = torch.tensor([[0., 0., float(target_size), float(target_size)]]).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            all_preds, scores = model(x_tensor, boxes)
            preds = get_best_mask(all_preds, scores)
            preds = F.interpolate(preds.float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
            return (torch.sigmoid(preds) > 0.5).long()


def test(cfg):
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    print("Loading test dataset setup...")
    full_ds = build_dataset(cfg)
    _, _, test_subset, pos_weight = build_train_val_test_subsets(full_ds, cfg)

    test_ds = TransformSubset(test_subset, A.Compose([ToTensorV2()]))
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    print(f"Loading LENTE from {cfg['checkpoint_name']}...")
    model = build_model(cfg, DEVICE)
    ckpt = torch.load(cfg["checkpoint_name"], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    te_loss = te_iou = te_miou = te_f1 = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TESTING"):
            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)
            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                te_loss += criterion(preds, masks, pos_weight=pos_weight).item()
            iou_fg, _, miou, f1, _, _ = calculate_metrics(preds, masks)
            te_iou += iou_fg; te_miou += miou; te_f1 += f1

    nte = len(test_loader)
    print("\n" + "=" * 40)
    print(f"FINAL TEST RESULTS ({cfg['dataset_name']})")
    print("=" * 40)
    print(f"Test Loss: {te_loss / nte:.4f}")
    print(f"Test IoU:  {te_iou / nte:.4f}")
    print(f"Test mIoU: {te_miou / nte:.4f}")
    print(f"Test F1:   {te_f1 / nte:.4f}")


if __name__ == "__main__":
    test(load_config())

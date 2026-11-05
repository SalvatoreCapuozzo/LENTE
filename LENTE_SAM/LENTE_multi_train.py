"""
LENTE_multi_train.py
====================
Training congiunto di UN solo modello LENTEMulti su L4S + Bijie.

Strategia (vedi piano):
  * batch omogenei alternati: ad ogni step si fa forward/backward su un batch di
    un solo dominio; i domini si alternano (l4s, bijie, l4s, bijie, ...);
  * pos_weight per-dataset: ogni batch usa il pos_weight del proprio dominio;
  * bilanciamento 50/50: il dataset piu' piccolo viene ri-iterato (oversampling);
  * normalizzazione z-score per-dataset dell'ottico (statistiche del train);
  * validazione SEPARATA per dominio; checkpoint/early-stopping sulla MEDIA degli
    IoU dei due domini, cosi' il modello non collassa su uno solo.

Uso:
    python LENTE_multi_train.py
    # oppure via launcher: python LENTE_run.py --dataset multi --mode train
"""

import json
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    criterion, get_best_mask, calculate_metrics, scale_boxes,
    save_checkpoint, load_checkpoint,
)
from LENTE_multi_utils import (
    load_multi_config, build_multi_model, prepare_member,
    interleaved_batches, num_interleaved_steps,
)


def _resize_batch(batch, target, device):
    imgs  = batch["image"].to(device)
    masks = batch["mask"].to(device)
    boxes = batch["boxes"].to(device)
    orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
    imgs  = F.interpolate(imgs, (target, target), mode="bilinear", align_corners=False)
    masks = F.interpolate(masks.unsqueeze(1).float(), (target, target),
                          mode="nearest").squeeze(1).long()
    boxes = scale_boxes(boxes, orig_h, orig_w, target)
    return imgs, masks, boxes


def _fg_counts(preds, masks):
    """TP/FP/FN foreground (pixel pooled) di un batch. preds: logits (B,1,h,w)."""
    if preds.shape[-2:] != masks.shape[-2:]:
        preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    pr = (torch.sigmoid(preds) > 0.5).float()
    gt = masks.unsqueeze(1).float()
    tp = (pr * gt).sum().item()
    fp = (pr * (1 - gt)).sum().item()
    fn = ((1 - pr) * gt).sum().item()
    return tp, fp, fn


def _iou(tp, fp, fn, eps=1e-7):
    return (tp + eps) / (tp + fp + fn + eps)


@torch.no_grad()
def _validate(model, member, target, device):
    """
    Valuta un dominio accumulando TP/FP/FN foreground su TUTTO il val set
    (IoU micro, pixel pooled). Ritorna (iou_micro, val_loss, tp, fp, fn) cosi'
    i conteggi si possono sommare fra domini per l'IoU totale aggregato.
    """
    model.eval()
    source = member["source"]
    v_loss = 0.0
    n = 0
    TP = FP = FN = 0.0
    for batch in tqdm(member["val_loader"], desc=f"[VAL {source}]", leave=False):
        imgs, masks, boxes = _resize_batch(batch, target, device)
        with torch.amp.autocast(device_type=device):
            all_preds, scores = model(imgs, boxes, source)
            preds = get_best_mask(all_preds, scores)
            v_loss += criterion(preds, masks, pos_weight=member["pos_weight"]).item()
        tp, fp, fn = _fg_counts(preds, masks)
        TP += tp; FP += fp; FN += fn
        n += 1
    return _iou(TP, FP, FN), v_loss / max(n, 1), TP, FP, FN


def train_multi(cfg):
    device = cfg["device"]
    target = cfg["target_size"]
    ckpt   = cfg["checkpoint_name"]
    hist_f = cfg["history_file"]

    # augmentation: identiche a LENTE_train (dataset-agnostiche)
    offline_aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    train_trans = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                             A.RandomRotate90(p=0.5), ToTensorV2()])
    val_trans   = A.Compose([ToTensorV2()])

    # prepara entrambi i domini
    members = {}
    for name, mcfg in cfg["members_cfg"].items():
        members[name] = prepare_member(mcfg, cfg, offline_aug, train_trans, val_trans)

    # modello + normalizzazione ottica per-dominio
    model = build_multi_model(cfg, device)
    if cfg.get("optical_norm", "zscore") == "zscore":
        for name, m in members.items():
            model.set_optical_norm(name, m["opt_mean"], m["opt_std"])
        print("Normalizzazione ottica z-score per-dataset attiva.")
    else:
        print("Normalizzazione ottica disattivata (optical_norm != 'zscore').")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(device=device)

    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, ckpt, device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining congiunto LENTEMulti | domini={list(members)} | "
          f"params trainabili={n_trainable/1e6:.2f}M | {device}\n")

    train_loaders = {name: m["train_loader"] for name, m in members.items()}
    pos_weights   = {name: m["pos_weight"]   for name, m in members.items()}

    history = []
    if os.path.exists(hist_f):
        with open(hist_f) as f:
            history = json.load(f)
    patience_counter = 0

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        per_src_loss = {s: 0.0 for s in members}
        per_src_iou  = {s: 0.0 for s in members}
        per_src_cnt  = {s: 0   for s in members}

        total = num_interleaved_steps(train_loaders)
        pbar = tqdm(interleaved_batches(train_loaders), total=total,
                    desc=f"Ep {epoch+1}/{cfg['epochs']} [TRAIN]")
        for source, batch in pbar:
            imgs, masks, boxes = _resize_batch(batch, target, device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                all_preds, scores = model(imgs, boxes, source)
                preds = get_best_mask(all_preds, scores)
                loss = criterion(preds, masks, pos_weight=pos_weights[source])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iou_fg, _, _, _, _, _ = calculate_metrics(preds.detach(), masks)
            per_src_loss[source] += loss.item()
            per_src_iou[source]  += iou_fg
            per_src_cnt[source]  += 1
            pbar.set_postfix(src=source, loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}")

        # --- validazione per dominio + IoU totale aggregato ---
        val_ious = {}      # IoU micro per dominio (L4S, Bijie)
        val_losses = {}
        TP_all = FP_all = FN_all = 0.0
        for name, m in members.items():
            v_iou, v_loss, tp, fp, fn = _validate(model, m, target, device)
            val_ious[name] = v_iou
            val_losses[name] = v_loss
            TP_all += tp; FP_all += fp; FN_all += fn

        # IoU TOTALE: pixel pooled su entrambi i domini, senza distinguere
        total_val_iou = _iou(TP_all, FP_all, FN_all)
        # IoU media (bilanciata): media dei due IoU per-dominio
        mean_val_iou = sum(val_ious.values()) / len(val_ious)
        # selezione checkpoint sulla media bilanciata (non penalizza il dominio piccolo)
        scheduler.step(mean_val_iou)

        epoch_rec = {
            "epoch": epoch + 1,
            "train_loss": {s: per_src_loss[s] / max(per_src_cnt[s], 1) for s in members},
            "train_iou":  {s: per_src_iou[s]  / max(per_src_cnt[s], 1) for s in members},
            "val_iou":    val_ious,
            "val_loss":   val_losses,
            "mean_val_iou":  mean_val_iou,
            "total_val_iou": total_val_iou,
        }
        history.append(epoch_rec)
        with open(hist_f, "w") as f:
            json.dump(history, f, indent=2)

        msg = " | ".join(f"{s}: IoU {val_ious[s]:.4f}" for s in members)
        print(f"EPOCA {epoch+1} | {msg} | MEDIA IoU: {mean_val_iou:.4f} | TOTALE IoU: {total_val_iou:.4f}")

        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, ckpt)
            print(f"  Record media IoU: {best_iou:.4f} — checkpoint salvato in {ckpt}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\nEARLY STOPPING — Miglior media IoU: {best_iou:.4f}")
                break


if __name__ == "__main__":
    train_multi(load_multi_config())

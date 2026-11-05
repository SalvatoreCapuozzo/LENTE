"""
LENTE_cascade_train.py — cascata a tre stadi con RIPIEGO
========================================================

    immagine 14ch
        │
        ▼
    ResNet34 congelata  ──► p(frana)
        │
        ├── p >= soglia ──► YOLO congelato ──► box ──┐
        │                        │                    │
        │                        └─ non trova nulla ──┤
        │                                             ▼
        └── p <  soglia ─────────────────────────► SAM2  ──► maschera
                                                  (con box, o prompt-free)

LA DIFFERENZA DAL GATE. Il gate a valle AZZERA la maschera dove il
classificatore dice "vuota": se sbaglia, quella patch vale 0 e non si recupera
(MISURATO: costa 0.0245 su `con frana`). Qui invece un "vuota" non spegne
niente — toglie soltanto il prompt, e SAM continua a cercare da solo. Il
classificatore puo' solo AGGIUNGERE informazione, mai sottrarne.

Vale anche quando YOLO tace pur essendo stato attivato: quel campione ricade in
prompt-free, non riceve un box finto. Un box a tutta immagine dato "per non
lasciarlo vuoto" sarebbe un bias, perche' finirebbe sia sulle patch davvero
vuote sia sui positivi che il detector ha mancato.

METRICA su TUTTE le patch (vuote comprese), in due forme:

    macro   IoU di ogni patch, poi media. Una patch vuota predetta vuota vale
            1.0. E' la metrica storica del progetto.
    micro   TP, FP, FN accumulati su tutti i pixel del test, una divisione sola.
            Una patch vuota indovinata contribuisce ZERO: non regala niente.
            E' quella confrontabile con la letteratura su Landslide4Sense
            (vincitore 2022: IoU 0.594 / F1 0.7454).

Il checkpoint viene selezionato sulla macro TOTALE, cioe' su tutte le immagini.

Uso:
    python LENTE_cascade_train.py -d l4s --checkpoint best_lente_l4s_cascade.pth
"""

import argparse
import json
import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    build_model, criterion, get_best_mask, scale_boxes, as_refine_map,
    save_checkpoint, load_checkpoint, configure_criterion,
    run_record_path, write_json, _jsonable,
)


# ------------------------------------------------------------------
# metriche, accumulate su TUTTE le patch
# ------------------------------------------------------------------

class Metriche:
    """
    Raccoglie macro e micro nella stessa passata.

    macro: si tiene la IoU di OGNI campione, cosi' alla fine si puo' anche
           separare vuote da con-frana senza rifare il calcolo.
    micro: si accumulano tp/fp/fn globali; la divisione si fa una volta sola
           alla fine, ed e' quello che rende le frane grandi dominanti.
    """

    def __init__(self):
        self.iou = []       # per campione
        self.vuota = []     # bool per campione
        self.tp = self.fp = self.fn = 0.0

    @torch.no_grad()
    def add(self, preds, masks, eps=1e-7):
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = F.interpolate(preds.float(), size=masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
        pr = (torch.sigmoid(preds) > 0.5).float()
        gt = masks.unsqueeze(1).float()
        tp = (pr * gt).sum((2, 3)); fp = (pr * (1 - gt)).sum((2, 3))
        fn = ((1 - pr) * gt).sum((2, 3))
        self.iou.append(((tp + eps) / (tp + fp + fn + eps)).squeeze(1).cpu())
        self.vuota.append((gt.flatten(1).sum(1) == 0).cpu())
        self.tp += float(tp.sum()); self.fp += float(fp.sum()); self.fn += float(fn.sum())

    def risultato(self):
        iou = torch.cat(self.iou); vuota = torch.cat(self.vuota)
        micro_iou = self.tp / max(self.tp + self.fp + self.fn, 1.0)
        micro_f1 = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1.0)
        return {"totale": float(iou.mean()),
                "vuote": float(iou[vuota].mean()) if vuota.any() else float("nan"),
                "con_frana": float(iou[~vuota].mean()) if (~vuota).any() else float("nan"),
                "micro_iou": micro_iou, "micro_f1": micro_f1,
                "n": int(iou.numel()), "n_vuote": int(vuota.sum())}


# ------------------------------------------------------------------

def train(cfg, args):
    dev, T = cfg["device"], cfg["target_size"]
    thr = float(args.thr)
    refine_passes = int(cfg.get("refine_passes", 0))
    refine_lambda = float(cfg.get("refine_lambda", 0.5))

    full_ds = build_dataset(cfg)
    tr_sub, va_sub, _, pos_weight = build_train_val_test_subsets(full_ds, cfg)

    offline = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    tr_trans = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                          A.RandomRotate90(p=0.5), ToTensorV2()])
    pos_idx = [i for i in tr_sub.indices if i in set(full_ds.positive_indices)]
    full_ds.add_augmented_samples(indices=pos_idx, transform=offline,
                                  n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    aug_start = len(full_ds.data) - len(pos_idx) * cfg["aug_copies"]
    tr_sub = Subset(full_ds, list(tr_sub.indices) + list(range(aug_start, len(full_ds.data))))

    tr_ld = DataLoader(TransformSubset(tr_sub, tr_trans), batch_size=cfg["batch_size"],
                       shuffle=True, pin_memory=True)
    va_ld = DataLoader(TransformSubset(va_sub, A.Compose([ToTensorV2()])),
                       batch_size=cfg["batch_size"], pin_memory=True)

    # SAM2 in regime "yolo_box": resolve_prompt passa i box quando ci sono e
    # ritorna None quando `boxes` e' None — che e' esattamente il ripiego.
    cfg = dict(cfg); cfg["prompt_source"] = "yolo_box"
    model = build_model(cfg, dev)
    configure_criterion(cfg)

    from LENTE_classifier import ClfPresenceProvider
    from LENTE_yolo_cascade import YoloBoxProvider
    clf = ClfPresenceProvider(args.clf, dev)
    yolo = YoloBoxProvider(cfg["yolo_weights"], cfg)

    fast = [model.dense_log_scale, model.refine_log_scale, model.clf_log_scale]
    fast_ids = {id(p) for p in fast}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in fast_ids]
    groups = [{"params": rest, "lr": cfg["learning_rate"]}]
    if refine_passes:
        groups.append({"params": fast, "lr": cfg.get("boxhead_lr", 1.0e-3)})
    optimizer = optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(device=dev)

    start_epoch, best = load_checkpoint(model, optimizer, scheduler, args.checkpoint, dev)
    hist_path = run_record_path(args.checkpoint, "history")
    record = {"checkpoint": args.checkpoint, "architettura": "clf -> yolo -> sam (ripiego prompt-free)",
              "clf": args.clf, "yolo": cfg["yolo_weights"], "soglia_clf": thr,
              "inizio": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": _jsonable(cfg), "epoche": []}

    print(f"\nCASCATA  classificatore -> YOLO -> SAM2   |   {dev}")
    print(f"  classificatore : {args.clf}"
          + (f" (bilanciata {clf.balanced_acc:.4f})" if clf.balanced_acc else ""))
    print(f"  soglia         : {thr}   (sotto -> prompt-free, SAM cerca lo stesso)")
    print(f"  detector       : {cfg['yolo_weights']} | conf={yolo.conf}")
    print(f"  raffinamento   : {refine_passes} passaggi")
    print(f"  metrica        : macro E micro su TUTTE le patch")
    print(f"  selezione su   : {cfg.get('select_on', 'totale')}")
    print(f"  record         : {hist_path}\n")

    def prompt_del_batch(imgs_cpu):
        """
        (boxes nativi, found). `found` e' True solo se il classificatore ha
        attivato il detector E il detector ha trovato qualcosa: negli altri casi
        il campione va in prompt-free, mai con un box finto.
        """
        p = clf.p_for(imgs_cpu)
        attivi = (p >= thr).cpu()
        boxes = torch.zeros(imgs_cpu.shape[0], 4)
        found = torch.zeros(imgs_cpu.shape[0], dtype=torch.bool)
        if attivi.any():
            b, f = yolo.boxes_for(imgs_cpu[attivi])
            idx = attivi.nonzero(as_tuple=True)[0]
            boxes[idx] = b
            found[idx] = f
        return boxes, found, int(attivi.sum())

    def forward_split(imgs, masks, boxes, found, calcola_loss=True):
        """
        Una forward con i box e una senza. `input_boxes` e' uniforme sul batch,
        quindi non si puo' dare il prompt a meta' campioni nella stessa
        chiamata. Le immagini processate restano le stesse, quindi il costo in
        piu' e' solo la minore saturazione della GPU.
        Ritorna (loss, preds ricomposti nell'ordine originale).
        """
        n = imgs.shape[0]
        loss = imgs.new_zeros(())
        out = None
        for sel, bx in ((found, boxes), (~found, None)):
            k = int(sel.sum())
            if k == 0:
                continue
            b = bx[sel] if bx is not None else None
            preds = get_best_mask(*model(imgs[sel], b))
            for _ in range(refine_passes):
                r = as_refine_map(preds, model.mask_input_size[0])
                preds = get_best_mask(*model(imgs[sel], b, refine=r))
            if calcola_loss:
                loss = loss + (k / n) * criterion(preds, masks[sel], pos_weight=pos_weight)
            # float32 esplicito: sotto autocast `preds` e' Half, e ricomporre in
            # Half farebbe fallire l'assegnazione (o perderebbe precisione nelle
            # metriche, che si calcolano dopo).
            if out is None:
                out = torch.zeros((n, *preds.shape[1:]), dtype=torch.float32,
                                  device=preds.device)
            out[sel] = preds.float()
        return loss, out

    patience = 0
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t_loss = 0.0; n_att = n_box = n_tot = 0
        pbar = tqdm(tr_ld, desc=f"Ep {epoch+1}/{cfg['epochs']} [TRAIN]")
        for batch in pbar:
            boxes_nat, found, att = prompt_del_batch(batch["image"])
            imgs, masks = batch["image"].to(dev), batch["mask"].to(dev)
            h0, w0 = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, (T, T), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (T, T),
                                  mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes_nat.to(dev), h0, w0, T)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=dev):
                loss, _ = forward_split(imgs, masks, boxes, found.to(dev))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

            t_loss += loss.item()
            n_att += att; n_box += int(found.sum()); n_tot += imgs.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             clf=f"{att}/{imgs.shape[0]}", box=f"{int(found.sum())}")

        model.eval()
        m = Metriche(); v_loss = 0.0; v_att = v_box = 0
        with torch.no_grad():
            for batch in tqdm(va_ld, desc="[VAL]", leave=False):
                boxes_nat, found, att = prompt_del_batch(batch["image"])
                imgs, masks = batch["image"].to(dev), batch["mask"].to(dev)
                h0, w0 = imgs.shape[-2], imgs.shape[-1]
                imgs_r = F.interpolate(imgs, (T, T), mode="bilinear", align_corners=False)
                masks_r = F.interpolate(masks.unsqueeze(1).float(), (T, T),
                                        mode="nearest").squeeze(1).long()
                boxes = scale_boxes(boxes_nat.to(dev), h0, w0, T)
                with torch.amp.autocast(device_type=dev):
                    l, preds = forward_split(imgs_r, masks_r, boxes, found.to(dev))
                v_loss += float(l); v_att += att; v_box += int(found.sum())
                # metriche alla risoluzione NATIVA, non a T: e' la griglia del dato
                m.add(preds, masks)

        r = m.risultato()
        vl = v_loss / len(va_ld)
        scheduler.step(vl)
        print(f"EPOCA {epoch+1} | loss {vl:.4f} | TOTALE {r['totale']:.4f} "
              f"[frane {r['con_frana']:.4f} | vuote {r['vuote']:.4f}]  "
              f"micro IoU {r['micro_iou']:.4f} F1 {r['micro_f1']:.4f}  "
              f"| clf {v_att}/{r['n']} box {v_box}")

        riga = {"epoca": epoch + 1, "train_loss": t_loss / len(tr_ld), "val_loss": vl,
                "clf_attivati": v_att, "box_trovati": v_box, **r}
        if refine_passes:
            riga["kr"] = float(torch.exp(model.refine_log_scale))

        # Su quale numero salvare. "micro" e' l'unico confrontabile con la
        # letteratura; "totale" e "frane" sono le due macro.
        scelta = cfg.get("select_on", "totale")
        criterio = {"micro": r["micro_iou"], "frane": r["con_frana"]}.get(scelta, r["totale"])
        riga["criterio"] = {"micro": "micro_iou", "frane": "con_frana"}.get(scelta, "totale")
        if criterio > best:
            best = criterio
            save_checkpoint(model, optimizer, scheduler, epoch, best, args.checkpoint)
            print(f"  record {best:.4f} — salvato in {args.checkpoint}")
            patience = 0; riga["record"] = True
        else:
            patience += 1

        record["epoche"].append(riga); record["best_totale"] = best
        record["ultima_epoca"] = epoch + 1
        write_json(hist_path, record)

        if patience >= cfg["patience"]:
            print(f"\nEARLY STOPPING — miglior TOTALE: {best:.4f}")
            record["stop"] = "early_stopping"; write_json(hist_path, record)
            break


@torch.no_grad()
def test(cfg, args):
    dev, T = cfg["device"], cfg["target_size"]
    thr = float(args.thr)
    refine_passes = int(cfg.get("refine_passes", 0))

    full_ds = build_dataset(cfg)
    _, _, te_sub, pos_weight = build_train_val_test_subsets(full_ds, cfg)
    ld = DataLoader(TransformSubset(te_sub, A.Compose([ToTensorV2()])),
                    batch_size=cfg["batch_size"])

    cfg = dict(cfg); cfg["prompt_source"] = "yolo_box"
    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev)["model_state_dict"],
                          strict=False)
    model.eval()
    configure_criterion(cfg)

    from LENTE_classifier import ClfPresenceProvider
    from LENTE_yolo_cascade import YoloBoxProvider
    clf = ClfPresenceProvider(args.clf, dev)
    yolo = YoloBoxProvider(cfg["yolo_weights"], cfg)

    m = Metriche(); n_att = n_box = 0
    for batch in tqdm(ld, desc="TEST"):
        p = clf.p_for(batch["image"])
        attivi = (p >= thr).cpu()
        boxes_nat = torch.zeros(batch["image"].shape[0], 4)
        found = torch.zeros(batch["image"].shape[0], dtype=torch.bool)
        if attivi.any():
            b, f = yolo.boxes_for(batch["image"][attivi])
            idx = attivi.nonzero(as_tuple=True)[0]
            boxes_nat[idx] = b; found[idx] = f
        n_att += int(attivi.sum()); n_box += int(found.sum())

        imgs, masks = batch["image"].to(dev), batch["mask"].to(dev)
        h0, w0 = imgs.shape[-2], imgs.shape[-1]
        imgs_r = F.interpolate(imgs, (T, T), mode="bilinear", align_corners=False)
        boxes = scale_boxes(boxes_nat.to(dev), h0, w0, T)
        found = found.to(dev)

        out = None
        with torch.amp.autocast(device_type=dev):
            for sel, bx in ((found, boxes), (~found, None)):
                if int(sel.sum()) == 0:
                    continue
                b = bx[sel] if bx is not None else None
                preds = get_best_mask(*model(imgs_r[sel], b))
                for _ in range(refine_passes):
                    r_ = as_refine_map(preds, model.mask_input_size[0])
                    preds = get_best_mask(*model(imgs_r[sel], b, refine=r_))
                if out is None:
                    out = torch.zeros((imgs_r.shape[0], *preds.shape[1:]),
                                      dtype=torch.float32, device=preds.device)
                out[sel] = preds.float()
        m.add(out, masks)

    r = m.risultato()
    print("\n" + "=" * 62)
    print(f"CASCATA clf -> yolo -> SAM   ({args.checkpoint})")
    print("=" * 62)
    print(f"  {r['n']} patch: {r['n_vuote']} vuote, {r['n'] - r['n_vuote']} con frana")
    print(f"  classificatore ha attivato YOLO su {n_att} ({n_att/r['n']:.1%})")
    print(f"  YOLO ha trovato un box su {n_box}; gli altri in prompt-free")
    print(f"\n  MACRO  totale {r['totale']:.4f}  [frane {r['con_frana']:.4f} "
          f"| vuote {r['vuote']:.4f}]")
    print(f"  MICRO  IoU {r['micro_iou']:.4f} | F1 {r['micro_f1']:.4f}")
    print(f"         (vincitore L4S 2022: IoU 0.594 / F1 0.7454 — su dati diversi)")

    out = run_record_path(args.checkpoint, "results")
    write_json(out, {"checkpoint": args.checkpoint,
                     "architettura": "clf -> yolo -> sam (ripiego prompt-free)",
                     "clf": args.clf, "soglia_clf": thr, "yolo": cfg["yolo_weights"],
                     "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "clf_attivati": n_att, "box_trovati": n_box,
                     "config": _jsonable(cfg), **r})
    print(f"\nScritto {out}")


def main():
    ap = argparse.ArgumentParser(description="Cascata classificatore -> YOLO -> SAM2")
    ap.add_argument("-m", "--mode", choices=["train", "test"], default="train")
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="best_lente_l4s_cascade.pth")
    ap.add_argument("--clf", default="best_clf_l4s.pth")
    ap.add_argument("--thr", type=float, default=0.30,
                    help="sotto questa p(frana) il campione va in prompt-free")
    a = ap.parse_args()
    cfg = load_config(a.config, dataset=a.dataset)
    (train if a.mode == "train" else test)(cfg, a)


if __name__ == "__main__":
    main()

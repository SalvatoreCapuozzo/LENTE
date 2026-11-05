import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    build_model, criterion, get_best_mask, calculate_metrics, scale_boxes,
    as_refine_map, configure_criterion, run_record_path, write_json, _jsonable,
    clf_boxes
)
import time


def adoptable_inference_function(model, x_tensor, boxes, target_size=1024):
    """
    Single-image inference. x_tensor: (1, C, H, W) con C = n_optical + 2.
    boxes: (1,4) o None.

    Il prompt lo decide `model.prompt_source` (vedi LENTE_utils.resolve_prompt),
    non questa funzione: se il modello e' prompt-free, `boxes` viene ignorato e
    non va fabbricato alcun box fittizio.

    In regime "yolo_box" il box va passato dal CHIAMANTE, gia' prodotto dal
    detector sull'immagine nativa (vedi YoloBoxProvider.boxes_for): qui non si
    puo' derivarlo, e derivarlo dalla maschera sarebbe il leak.
    """
    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)
    orig_h, orig_w = x_tensor.shape[-2], x_tensor.shape[-1]
    x_tensor = F.interpolate(x_tensor, (target_size, target_size), mode="bilinear", align_corners=False)
    if boxes is not None and getattr(model, "prompt_source", "none") in ("gt_box", "yolo_box"):
        boxes = scale_boxes(boxes.to(device), orig_h, orig_w, target_size)

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

    configure_criterion(cfg)
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

    # ── Cascata YOLO -> SAM2 ────────────────────────────────────────────────
    # OBBLIGATORIO con prompt_source "yolo_box". Senza, `batch["boxes"]` resta
    # quello prodotto dal dataset, cioe' _bbox_from_mask: il box della GROUND
    # TRUTH. resolve_prompt lo passerebbe a SAM2 tale e quale, e il numero
    # stampato qui sotto sarebbe quello dell'ORACOLO (~0.75), non della cascata.
    # Stesso identico regime del training (LENTE_train.py): il detector gira
    # sull'immagine nativa su CPU, prima dell'upsample a TARGET_SIZE.
    yolo = None
    dense_mode = (cfg.get("yolo_prompt", "box") == "dense"
                  and cfg["prompt_source"] == "yolo_box")
    refine_passes = int(cfg.get("refine_passes", 0))
    usa_mask = dense_mode or cfg.get("clf_prompt", False) or refine_passes > 0

    clf = None
    clf_mode = cfg["prompt_source"] == "clf_box"
    clf_thr = float(cfg.get("clf_box_threshold", 0.30))
    if clf_mode or cfg.get("clf_prompt", False):
        from LENTE_classifier import ClfPresenceProvider
        clf = ClfPresenceProvider(cfg.get("clf_weights", "best_clf_l4s.pth"), DEVICE)
        print(f"{'PROMPT' if clf_mode else 'Segnale di PRESENZA'} dal classificatore: "
              f"{cfg.get('clf_weights','best_clf_l4s.pth')}"
              + (f" | soglia {clf_thr}" if clf_mode else ""))
    if cfg["prompt_source"] == "yolo_box":
        from LENTE_yolo_cascade import YoloBoxProvider
        yolo = YoloBoxProvider(cfg["yolo_weights"], cfg)
        if dense_mode:
            print(f"Prompt DENSO dal detector congelato: {cfg['yolo_weights']} | "
                  f"conf={cfg.get('yolo_dense_conf', 0.05)} | "
                  f"max_det={cfg.get('yolo_dense_max_det', 10)}")
        else:
            print(f"Box dal detector congelato: {cfg['yolo_weights']} | conf={yolo.conf}")

    def _forward_split(imgs, boxes, found, pres=None):
        """
        Gemella di LENTE_train._forward_split: una forward con i box e una
        senza, poi le predizioni RICUCITE nell'ordine del batch.

        `input_boxes` e' un tensore uniforme sul batch, quindi non si puo' dare
        un box a meta' campioni e niente agli altri nella stessa chiamata. Dove
        il detector tace il campione va in regime prompt-free: nessun box finto
        a tutta immagine, che sarebbe un bias proprio sui positivi mancati.

        Due difetti corretti. Non faceva il raffinamento, e nel ciclo il ramo
        denso veniva prima di questo: con refine_passes > 0 il box non entrava.
        E non restituiva le predizioni, quindi la micro qui sotto, in questo
        ramo, leggeva quelle di un altro batch.
        """
        n = imgs.shape[0]
        out = None
        for sel, bx in ((found, boxes), (~found, None)):
            if not sel.any():
                continue
            b = bx[sel] if bx is not None else None
            p = pres[sel] if pres is not None else None
            preds = get_best_mask(*model(imgs[sel], b, presence=p))
            for _ in range(refine_passes):
                r = as_refine_map(preds, model.mask_input_size[0])
                preds = get_best_mask(*model(imgs[sel], b, refine=r, presence=p))
            if out is None:
                out = preds.new_zeros((n,) + tuple(preds.shape[1:]))
            out[sel] = preds
        return out

    te_loss = te_iou = te_miou = te_f1 = 0.0
    n_found = n_tot = 0
    # Accumulatori MICRO: tp/fp/fn su tutti i pixel del test, divisione una volta
    # sola alla fine. E' la metrica della letteratura, ed e' un'altra cosa
    # rispetto alle macro qui sopra: una patch vuota predetta vuota contribuisce
    # ZERO a tutti e tre i termini, quindi non regala niente. Le macro invece le
    # danno 1.0, ed e' il 40% del test set.
    mi_tp = mi_fp = mi_fn = 0.0
    # Decomposizione: sulle patch VUOTE il punteggio macro e' 1.0 o 0.0 secco,
    # su quelle con frana e' graduale. La media sola non fa vedere quale si muove.
    iou_vuote, iou_frane = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TESTING"):
            found = dense = pres = None
            if clf is not None:
                pres = clf.p_for(batch["image"])
            if clf_mode:
                h0, w0 = batch["image"].shape[-2], batch["image"].shape[-1]
                batch = dict(batch)
                batch["boxes"] = clf_boxes(pres, clf_thr, h0, w0).cpu()
                n_found += int((pres >= clf_thr).sum()); n_tot += int(pres.numel())
                pres = None
            if yolo is not None and dense_mode:
                dense, nb = yolo.dense_for(batch["image"], model.mask_input_size[0])
                dense = dense.to(DEVICE)
                n_found += nb; n_tot += int(batch["image"].shape[0])
            elif yolo is not None:
                boxes_native, found = yolo.boxes_for(batch["image"])
                batch = dict(batch); batch["boxes"] = boxes_native
                n_found += int(found.sum()); n_tot += int(found.numel())
            if cfg["prompt_source"] == "gt_box":
                # Come in LENTE_train. Oracolo: box vero solo sulle patch con
                # frana. Altrimenti box su tutte, sentinella compresa. In
                # entrambi i casi passa da _forward_split, dove il
                # raffinamento non scavalca il box.
                if cfg.get("gt_senza_sentinella", False):
                    found = batch["mask"].flatten(1).sum(1) > 0
                else:
                    found = torch.ones(batch["mask"].shape[0], dtype=torch.bool)

            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            # La MASCHERA resta sulla griglia NATIVA (128), non viene
            # ingrandita a TARGET_SIZE. Tutte le funzioni a valle
            # (calculate_metrics, iou_per_sample, l'accumulo micro) riportano
            # le PREDIZIONI sulla griglia della maschera, quindi cosi' la
            # metrica e' sempre calcolata sui pixel veri.
            #
            # PERCHE' CONTA. Ingrandendo la maschera, la griglia di valutazione
            # dipende da TARGET_SIZE: a 512 il decoder emette a 128, cioe' al
            # nativo, ma a 1024 emette a 256 — il doppio della risoluzione che
            # la ground truth possiede. Il modello potrebbe collocare i contorni
            # con precisione di mezzo pixel nativo, misurato contro una maschera
            # a blocchi: quel raffinamento non puo' guadagnare e puo' perdere.
            # Il confronto fra risoluzioni misurerebbe due cose insieme.
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)
            with torch.amp.autocast(device_type=DEVICE):
                if clf_mode:
                    all_preds, scores = model(imgs, boxes)
                    preds = get_best_mask(all_preds, scores)
                    for _ in range(refine_passes):
                        r = as_refine_map(preds, model.mask_input_size[0])
                        preds = get_best_mask(*model(imgs, boxes, refine=r))
                    te_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                    iou_fg, _, miou, f1, _, _ = calculate_metrics(preds, masks)
                elif found is not None:
                    # prima del ramo denso, come in LENTE_train: altrimenti con
                    # il raffinamento attivo il box non entra
                    preds = _forward_split(imgs, boxes, found.to(DEVICE), pres)
                    te_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                    iou_fg, _, miou, f1, _, _ = calculate_metrics(preds, masks)
                elif usa_mask:
                    # STESSO numero di passaggi del training: se il modello e'
                    # stato addestrato a rileggersi e qui gira una volta sola,
                    # lo si valuta a meta' del lavoro che sa fare.
                    all_preds, scores = model(imgs, None, dense=dense, presence=pres)
                    preds = get_best_mask(all_preds, scores)
                    for _ in range(refine_passes):
                        r = as_refine_map(preds, model.mask_input_size[0])
                        all_preds, scores = model(imgs, None, dense=dense,
                                                  refine=r, presence=pres)
                        preds = get_best_mask(all_preds, scores)
                    te_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                    iou_fg, _, miou, f1, _, _ = calculate_metrics(preds, masks)
                else:
                    all_preds, scores = model(imgs, boxes)
                    preds = get_best_mask(all_preds, scores)
                    te_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                    iou_fg, _, miou, f1, _, _ = calculate_metrics(preds, masks)
            te_iou += iou_fg; te_miou += miou; te_f1 += f1

            # micro + decomposizione, sulla griglia NATIVA della maschera
            pr = F.interpolate(preds.float(), masks.shape[-2:],
                               mode="bilinear", align_corners=False)
            pr = (torch.sigmoid(pr) > 0.5).float(); gt = masks.unsqueeze(1).float()
            tp = (pr * gt).sum((2, 3)); fp = (pr * (1 - gt)).sum((2, 3))
            fn = ((1 - pr) * gt).sum((2, 3))
            mi_tp += float(tp.sum()); mi_fp += float(fp.sum()); mi_fn += float(fn.sum())
            ps = ((tp + 1e-7) / (tp + fp + fn + 1e-7)).squeeze(1).cpu()
            vuota = (gt.flatten(1).sum(1) == 0).cpu()
            iou_vuote += ps[vuota].tolist(); iou_frane += ps[~vuota].tolist()

    nte = len(test_loader)
    micro_iou = mi_tp / max(mi_tp + mi_fp + mi_fn, 1.0)
    micro_f1 = 2 * mi_tp / max(2 * mi_tp + mi_fp + mi_fn, 1.0)
    v_m = sum(iou_vuote) / max(len(iou_vuote), 1)
    f_m = sum(iou_frane) / max(len(iou_frane), 1)
    res = {"micro_iou": micro_iou, "micro_f1": micro_f1,
           "macro_vuote": v_m, "macro_con_frana": f_m,
           "n_vuote": len(iou_vuote), "n_frane": len(iou_frane),
           "checkpoint": cfg["checkpoint_name"],
           "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
           "config": _jsonable(cfg),
           "n_batch": nte,
           "test_loss": te_loss / nte, "test_iou": te_iou / nte,
           "test_miou": te_miou / nte, "test_f1": te_f1 / nte}
    if yolo is not None:
        res["yolo_proposte" if dense_mode else "yolo_campioni_con_box"] = n_found
        res["yolo_totale"] = n_tot
        res["dense_scale"] = float(torch.exp(model.dense_log_scale))
        if refine_passes:
            res["refine_scale"] = float(torch.exp(model.refine_log_scale))
    out = run_record_path(cfg["checkpoint_name"], "results")
    write_json(out, res)

    print("\n" + "=" * 40)
    print(f"FINAL TEST RESULTS ({cfg['dataset_name']})")
    print("=" * 40)
    print(f"Prompt:    {cfg['prompt_source']}")
    if clf_mode:
        print(f"Classificatore:  {n_found}/{n_tot} campioni ({n_found / max(n_tot,1):.1%}) "
              f"con box a tutta immagine; agli altri la sentinella")
    elif yolo is not None and dense_mode:
        print(f"Box YOLO:  {n_found} proposte dipinte su {n_tot} campioni "
              f"({n_found / max(n_tot, 1):.2f} per campione)")
        print(f"Scala appresa della mappa densa: {float(torch.exp(model.dense_log_scale)):.3f}")
        if refine_passes:
            print(f"Passaggi di raffinamento: {refine_passes} | "
                  f"scala appresa del raffinamento: {float(torch.exp(model.refine_log_scale)):.3f}")
    elif yolo is not None:
        print(f"Box YOLO:  {n_found}/{n_tot} campioni ({n_found / max(n_tot, 1):.1%}); "
              f"il resto e' andato in regime prompt-free")
    print(f"Test Loss: {te_loss / nte:.4f}")
    print(f"Test IoU:  {te_iou / nte:.4f}")
    print(f"Test mIoU: {te_miou / nte:.4f}")
    print(f"Test F1:   {te_f1 / nte:.4f}")
    print(f"\n  macro   vuote {v_m:.4f} (n={len(iou_vuote)}) | "
          f"con frana {f_m:.4f} (n={len(iou_frane)})")
    print(f"  MICRO   IoU {micro_iou:.4f} | F1 {micro_f1:.4f}")
    print(f"          <- questa e' la metrica confrontabile con la letteratura")
    print(f"             vincitore L4S 2022: IoU 0.594 / F1 0.7454")
    print(f"             migliore pubblicato: IoU 0.637 / F1 0.7783   (dati diversi)")
    print(f"\nScritto {out}")


if __name__ == "__main__":
    test(load_config())

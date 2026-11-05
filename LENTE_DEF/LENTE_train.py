import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    build_model, criterion, get_best_mask, calculate_metrics, scale_boxes,
    save_checkpoint, load_checkpoint, jitter_boxes, as_refine_map,
    configure_criterion, run_record_path, write_json, _jsonable, iou_per_sample,
    clf_boxes, SENTINEL_BOX
)


def train(cfg):
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    full_ds = build_dataset(cfg)
    train_subset, val_subset, _, pos_weight = build_train_val_test_subsets(full_ds, cfg)

    offline_aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    train_trans = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), ToTensorV2()])
    val_trans   = A.Compose([ToTensorV2()])

    train_pos_indices = [i for i in train_subset.indices if i in set(full_ds.positive_indices)]
    full_ds.add_augmented_samples(indices=train_pos_indices, transform=offline_aug,
                                  n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    aug_start   = len(full_ds.data) - len(train_pos_indices) * cfg["aug_copies"]
    aug_indices = list(range(aug_start, len(full_ds.data)))

    # ── SAM addestrato SOLO sui positivi ────────────────────────────────
    # Oggi il 41% dei batch sono patch senza nessun bersaglio: li' il gradiente
    # dice solo "non accendere niente da nessuna parte". E' capacita' spesa per
    # imparare a TACERE — un lavoro che un classificatore binario fa meglio, in
    # 15 minuti, e che a valle viene comunque rifatto dal gate.
    # Togliendo i negativi, SAM diventa un modello puro "data una frana,
    # disegnala", con tutta la capacita' sulla colonna `con frana`.
    #
    # IL RISCHIO: non impara piu' a sopprimere i falsi positivi, quindi sulle
    # patch vuote che il gate lascia passare (misurato: il 5.6%, ~17 su 306)
    # produrra' spazzatura. Il guadagno sui positivi deve superare quel costo.
    solo_pos = bool(cfg.get("train_positives_only", False))
    n_orig = len(train_subset.indices)
    base_idx = train_pos_indices if solo_pos else list(train_subset.indices)
    train_subset = Subset(full_ds, base_idx + aug_indices)

    if solo_pos:
        # pos_weight va RICALCOLATO: quello del config viene dal train completo,
        # dove il 41% delle patch e' tutto sfondo. Tenendolo si sovrappeserebbe
        # la classe positiva rispetto al nuovo equilibrio.
        px_fg = px_tot = 0
        for i in train_pos_indices:
            m = np.asarray(full_ds.get_mask(i))
            px_fg += int(m.sum()); px_tot += m.size
        nuovo = ((px_tot - px_fg) / max(px_fg, 1)) ** 0.5
        print(f"  SOLO POSITIVI: {len(train_pos_indices)} patch (+{len(aug_indices)} aumentate) "
              f"invece di {n_orig} | pos_weight {pos_weight:.2f} -> {nuovo:.2f}")
        pos_weight = nuovo

    train_ds = TransformSubset(train_subset, train_trans)
    val_ds   = TransformSubset(val_subset,   val_trans)
    # Quali campioni di validazione contengono una frana: serve a leggere le due
    # colonne separate, e a selezionare il checkpoint sulla colonna giusta.
    val_pos = torch.tensor([i in set(full_ds.positive_indices) for i in val_subset.indices])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    model = build_model(cfg, DEVICE)

    # Canale con cui i box di YOLO raggiungono il decoder (solo con "yolo_box"):
    #   "box"   -> input_boxes, il rettangolo piu' confidente. Uniforme sul
    #              batch, quindi serve _forward_split per chi non ha proposte.
    #   "dense" -> input_masks, TUTTE le proposte dipinte in una mappa, ciascuna
    #              pesata dalla propria confidenza. Niente soglia secca, niente
    #              split, e piu' box possono coesistere.
    dense_mode = (cfg.get("yolo_prompt", "box") == "dense"
                  and cfg["prompt_source"] == "yolo_box")

    # `input_masks` puo' ricevere TRE contributi indipendenti, che si sommano
    # ciascuno con la propria scala appresa:
    #   dense     i box di YOLO rasterizzati        (localizzazione)
    #   presence  p(frana) del classificatore       (presenza, mappa uniforme)
    #   refine    la maschera del passaggio prima   (raffinamento)
    # Basta UNO di questi perche' si usi il percorso denso. Cosi' YOLO diventa
    # opzionale: si puo' girare con solo classificatore + raffinamento.
    usa_mask = dense_mode or cfg.get("clf_prompt", False) or int(cfg.get("refine_passes", 0)) > 0
    clf_thr = float(cfg.get("clf_box_threshold", 0.30))

    # `dense_log_scale` parte da zero come una testa nuova: con learning_rate
    # 1e-5 (tarato su pesi PRETRAINED) non si muoverebbe. E' la stessa lezione
    # gia' pagata col boxhead, dove la testa restava ferma non per il peso della
    # loss ma per il passo di aggiornamento.
    # Passaggi di RAFFINAMENTO: 0 = una forward sola (comportamento storico),
    # 1 = il modello rilegge la propria maschera e la corregge.
    refine_passes = int(cfg.get("refine_passes", 0))
    refine_lambda = float(cfg.get("refine_lambda", 0.5))

    fast = [model.dense_log_scale, model.refine_log_scale, model.clf_log_scale]
    fast_ids = {id(p) for p in fast}

    # Il prompt encoder, se scongelato, va in un gruppo suo. NON alla velocita'
    # delle teste nuove (1e-3): sono pesi PRETRAINED e a quel passo si
    # distruggono. Ma nemmeno a 1e-5, che e' tarato per 27M di parametri e su
    # 4.684 in venti epoche non li muove. `prompt_encoder_lr` sta in mezzo.
    pe_params = []
    if cfg.get("unfreeze_prompt_encoder", False):
        pe_params = [p for p in model.model.prompt_encoder.parameters() if p.requires_grad]
        fast_ids |= {id(p) for p in pe_params}

    # L'encoder scongelato va in un gruppo suo, a learning rate RIDOTTO.
    # Sono pesi pretrained e molti: al passo delle teste nuove si distruggono in
    # poche epoche. `encoder_lr_mult` moltiplica il learning rate di base —
    # 0.1 e' il valore convenzionale per il fine-tuning di un backbone.
    # L'encoder U-Net e' NUOVO, inizializzato a caso: a `learning_rate` (1e-5,
    # tarato per pesi preaddestrati) non si muove. Va al passo delle teste
    # nuove, come il patch embed e gli adapter — che pero' sono gia' nel gruppo
    # `rest` per ragioni storiche. Qui serve un gruppo esplicito.
    unet_params = []
    if getattr(model, "unet_encoder", None) is not None:
        unet_params = [q for q in model.unet_encoder.parameters() if q.requires_grad]
        fast_ids |= {id(q) for q in unet_params}

    # I gate GSRM sono moduli NUOVI, inizializzati da zero: il learning rate
    # base 1e-5 e' tarato per pesi preaddestrati e li muoverebbe appena. E' lo
    # stesso errore che ha rovinato il primo run dell'encoder U-Net, e qui
    # sarebbe piu' insidioso perche' un gate quasi fermo somiglia molto a un
    # gate che ha imparato a restare aperto.
    gsrm_params = [q for q in getattr(model, "gsrm_params", []) if q.requires_grad]
    fast_ids |= {id(q) for q in gsrm_params}

    fus_params = [q for q in getattr(model, "fusione_params", []) if q.requires_grad]
    fast_ids |= {id(q) for q in fus_params}
    hsam_params = [q for q in getattr(model, "hsam_params", []) if q.requires_grad]
    fast_ids |= {id(q) for q in hsam_params}
    fpn_params = [q for q in getattr(model, "fpn_params", []) if q.requires_grad]
    fast_ids |= {id(q) for q in fpn_params}

    enc_params = [q for q in getattr(model, "encoder_params", []) if q.requires_grad]
    enc_ids = {id(q) for q in enc_params}
    fast_ids |= enc_ids

    rest = [p for p in model.parameters()
            if p.requires_grad and id(p) not in fast_ids]
    groups = [{"params": rest, "lr": cfg["learning_rate"]}]
    if usa_mask:
        groups.append({"params": fast, "lr": cfg.get("boxhead_lr", 1.0e-3)})
    if unet_params:
        lr_u = float(cfg.get("unet_lr", 3.0e-4))
        groups.append({"params": unet_params, "lr": lr_u})
        print(f"  encoder sostitutivo: {sum(q.numel() for q in unet_params):,} "
              f"parametri a lr={lr_u:.0e}  (il base {cfg['learning_rate']:.0e} "
              f"e' tarato per pesi preaddestrati e qui non muoverebbe nulla)")
    if gsrm_params:
        lr_g = float(cfg.get("gsrm_lr", 3.0e-4))
        groups.append({"params": gsrm_params, "lr": lr_g})
        print(f"  gate GSRM: {sum(q.numel() for q in gsrm_params):,} "
              f"parametri a lr={lr_g:.0e}")
    if fus_params:
        lr_f = float(cfg.get("fusione_lr", 3.0e-4))
        groups.append({"params": fus_params, "lr": lr_f})
        print(f"  fusione a due stream: {sum(q.numel() for q in fus_params):,} "
              f"parametri a lr={lr_f:.0e}")
    if enc_params:
        lr_enc = cfg["learning_rate"] * float(cfg.get("encoder_lr_mult", 0.1))
        groups.append({"params": enc_params, "lr": lr_enc})
        print(f"  encoder a lr ridotto: {sum(q.numel() for q in enc_params):,} "
              f"parametri a lr={lr_enc:.1e} "
              f"({cfg.get('encoder_lr_mult', 0.1)}x del base)")
    if pe_params:
        lr_pe = float(cfg.get("prompt_encoder_lr", 1.0e-4))
        groups.append({"params": pe_params, "lr": lr_pe})
        print(f"  prompt encoder SCONGELATO: "
              f"{sum(p.numel() for p in pe_params):,} parametri a lr={lr_pe:.0e}")
    if fpn_params:
        lr_p = float(cfg.get("fpn_lr", 3.0e-4))
        groups.append({"params": fpn_params, "lr": lr_p})
        print(f"  FPN completo: {sum(q.numel() for q in fpn_params):,} parametri a lr={lr_p:.0e}")
    if hsam_params:
        # Modulo nuovo, addestrato da zero: al passo dei pesi preaddestrati
        # (1e-5) non si muoverebbe, come gli adapter su Sen12.
        lr_h = float(cfg.get("hsam_lr", 3.0e-4))
        groups.append({"params": hsam_params, "lr": lr_h})
        print(f"  H-SAM: {sum(q.numel() for q in hsam_params):,} parametri a lr={lr_h:.0e}")
    optimizer = optim.AdamW(groups, weight_decay=cfg["weight_decay"])

    # Con H-SAM la forward restituisce il SECONDO stadio. Il primo resta in
    # model.ultimo_prior e va supervisionato anche lui, come nel paper:
    # senza, la guida che riceve il secondo stadio non avrebbe nessun motivo
    # di restare una maschera, e diventerebbe una feature qualunque.
    hsam_lambda = float(cfg.get("hsam_lambda", 0.5))

    def _priore(masks_):
        pr = getattr(model, "ultimo_prior", None)
        if getattr(model, "hsam", None) is None or pr is None:
            return 0.0
        return hsam_lambda * criterion(pr, masks_, pos_weight=pos_weight)

    def _dense_forward(imgs, masks, dense, pres=None):
        """
        Una forward, piu' `refine_passes` riletture della propria maschera.

        La loss e' su TUTTI i passaggi, con peso `refine_lambda` su quelli
        intermedi: senza, il primo passaggio non avrebbe piu' nessun incentivo a
        essere buono da solo e diventerebbe un abbozzo qualunque, mentre a
        inferenza vogliamo che ogni giro sia una predizione valida.
        """
        all_preds, scores = model(imgs, None, dense=dense, presence=pres)
        preds = get_best_mask(all_preds, scores)
        loss = criterion(preds, masks, pos_weight=pos_weight) + _priore(masks)
        for _ in range(refine_passes):
            r = as_refine_map(preds, model.mask_input_size[0])
            all_preds, scores = model(imgs, None, dense=dense, refine=r, presence=pres)
            preds = get_best_mask(all_preds, scores)
            loss = (refine_lambda * loss + criterion(preds, masks, pos_weight=pos_weight)
                    + _priore(masks))
        return loss, preds
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(device=DEVICE)

    # La ripresa e' OPT-IN, tramite --resume. Prima load_checkpoint veniva
    # chiamata sempre: bastava che esistesse un file con lo stesso nome perche'
    # un run "nuovo" ripartisse in silenzio dai pesi di quello vecchio, e il
    # flag --resume non serviva a niente. E' il modo piu' facile di misurare un
    # modello credendo di misurarne un altro.
    if cfg.get("resume", False):
        start_epoch, best_iou = load_checkpoint(
            model, optimizer, scheduler, cfg["checkpoint_name"], DEVICE)
    else:
        start_epoch, best_iou = 0, 0.0
        if os.path.exists(cfg["checkpoint_name"]):
            print(f"  NB: {cfg['checkpoint_name']} esiste gia' e verra' "
                  f"SOVRASCRITTO. Passa --resume per riprenderlo invece.")
    patience_counter = 0

    # ── record del run su JSON ──────────────────────────────────────────
    # Finora questo loop non scriveva NIENTE: l'unica traccia di un run era lo
    # scrollback del terminale, che scorre via. Il file prende il nome dal
    # checkpoint, cosi' segue sempre il modello a cui appartiene.
    hist_path = run_record_path(cfg["checkpoint_name"], "history")
    record = {"checkpoint": cfg["checkpoint_name"],
              "inizio": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": _jsonable(cfg), "epoche": []}
    if cfg.get("resume") and os.path.exists(hist_path):
        try:
            record = json.load(open(hist_path))
        except Exception:
            pass
    print(f"  record del run: {hist_path}")

    # Degrado dei box GT usati come prompt in TRAINING. Ha senso solo con
    # prompt_source="gt_box", ed e' il pezzo che allinea LENTE alla qualita' di
    # box che ricevera' davvero dal detector nella cascata YOLO -> SAM2.
    # Calibrare con `python LENTE_yolo_cascade.py calibrate`, non a occhio.
    jitter = cfg.get("prompt_jitter", 0.0) if cfg["prompt_source"] == "gt_box" else 0.0

    # PROMPT DROPOUT: probabilita' di togliere del tutto il box in training.
    # Serve alla cascata YOLO -> SAM2, dove il detector propone un box solo sul
    # ~50% delle patch positive: senza dropout il modello sa lavorare solo col
    # prompt, e dove YOLO tace la cascata restituirebbe maschera vuota, cioe'
    # IoU 0 su una patch positiva. Con il dropout UN SOLO modello copre
    # entrambi i regimi e la cascata non puo' fare peggio del prompt-free.
    # Per batch e non per campione: input_boxes e' uniforme sul batch.
    drop = cfg.get('prompt_dropout', 0.0) if cfg['prompt_source'] == 'gt_box' else 0.0
    # Oracolo: box perfetti, quindi niente jitter e niente dropout, e nessun
    # box sulle patch vuote. Sono le tre condizioni sotto cui il numero misura
    # davvero "quanto segmenta SAM se gli si dice dove guardare".
    oracolo = (cfg['prompt_source'] == 'gt_box'
               and bool(cfg.get('gt_senza_sentinella', False)))
    if oracolo:
        jitter = drop = 0.0
        print("  ORACOLO: box vero sulle patch con frana, nessun box sulle "
              "vuote, niente jitter, niente dropout")

    # YOLO congelato che propone i box dentro il loop (prompt_source "yolo_box").
    # Gira sull'immagine GIA' AUMENTATA, quindi i box corrispondono alla maschera
    # per costruzione — cosa che precalcolandoli non sarebbe vera.
    yolo = None
    if cfg["prompt_source"] == "yolo_box":
        from LENTE_yolo_cascade import YoloBoxProvider
        yolo = YoloBoxProvider(cfg["yolo_weights"], cfg)

    # Classificatore congelato. In regime "clf_box" e' LUI a produrre il prompt,
    # nella stessa forma del vecchio gt_box: sentinella [0,0,10,10] dove dice
    # "vuota", box a tutta immagine dove dice "c'e' una frana". Stesso
    # meccanismo che dava vuote = 1.0000, ma senza leggere la ground truth.
    clf = None
    clf_mode = cfg["prompt_source"] == "clf_box"
    if clf_mode or cfg.get("clf_prompt", False):
        from LENTE_classifier import ClfPresenceProvider
        clf = ClfPresenceProvider(cfg.get("clf_weights", "best_clf_l4s.pth"), DEVICE)

    configure_criterion(cfg)
    _sel = "frane" if solo_pos else cfg.get("select_on", "totale")
    print("  checkpoint selezionato su: " + {
        "micro": "MICRO IoU  (la metrica confrontabile con la letteratura)",
        "frane": "IoU delle sole PATCH CON FRANA  (le vuote le copre il gate)",
    }.get(_sel, "IoU macro totale, tutte le patch"))

    print(f"\nTraining LENTE su '{cfg['dataset_name']}' | n_optical={cfg['n_optical']} | {DEVICE}")
    if jitter > 0:
        print(f"  box GT degradati in training: rel_std={jitter} (validazione e test NON degradati)")
    if drop > 0:
        print(f"  prompt dropout: {drop:.0%} dei batch senza box (il modello impara entrambi i regimi)")
    if clf_mode:
        print(f"  PROMPT DAL CLASSIFICATORE: {cfg.get('clf_weights','best_clf_l4s.pth')}"
              + (f" (bilanciata {clf.balanced_acc:.4f})" if clf.balanced_acc else "")
              + f" | soglia {clf_thr}")
        print(f"    p < {clf_thr}  ->  sentinella {SENTINEL_BOX}   (come il vecchio gt_box)")
        print(f"    p >= {clf_thr} ->  box a tutta immagine")
    elif clf is not None:
        print(f"  segnale di PRESENZA dal classificatore congelato: "
              f"{cfg.get('clf_weights','best_clf_l4s.pth')}"
              + (f" (bilanciata {clf.balanced_acc:.4f})" if clf.balanced_acc else ""))
    if yolo is not None:
        if dense_mode:
            print(f"  prompt DENSO dai box di YOLO: {cfg['yolo_weights']} | "
                  f"conf={cfg.get('yolo_dense_conf', 0.05)} | "
                  f"max_det={cfg.get('yolo_dense_max_det', 10)} | "
                  f"mappa {model.mask_input_size[0]}x{model.mask_input_size[1]}")
        else:
            print(f"  prompt dai box di YOLO (congelato): {cfg['yolo_weights']} | conf={yolo.conf}")
    print()

    def _forward_split(imgs, masks, boxes, found, pres=None):
        """
        Una forward con i box e una senza, poi loss e metriche mediate PESANDO
        per la numerosita' dei due gruppi.

        Serve perche' `input_boxes` e' un tensore uniforme sul batch: non si puo'
        dare un box a met&agrave; campioni e niente agli altri nella stessa chiamata. Le
        immagini processate restano pero' le stesse (5+3 = 8), quindi il costo
        aggiuntivo e' solo la minore saturazione della GPU, non il doppio.

        L'alternativa — dare un box a tutta immagine a chi non ne ha — sarebbe un
        BIAS: quel prompt finirebbe sia sulle patch vuote sia sui positivi che il
        detector ha mancato, insegnando al modello ad arrendersi proprio dove
        dovrebbe provarci.

        RAFFINAMENTO. Ogni gruppo fa anche le sue `refine_passes` riletture,
        con lo STESSO prompt, come _dense_forward. Prima questa funzione non le
        faceva, e nel ciclo il ramo denso veniva controllato PRIMA di questo:
        con refine_passes > 0 il box non entrava mai, ne' in addestramento ne'
        in validazione. L'oracolo addestrato cosi' non aveva mai visto un box.

        Restituisce le predizioni RICUCITE nell'ordine del batch: servono alla
        micro e alla IoU per campione della validazione, che prima in questo
        ramo leggevano le predizioni di un altro batch.
        """
        n = imgs.shape[0]
        loss = imgs.new_zeros(())
        out = None
        for sel, bx in ((found, boxes), (~found, None)):
            k = int(sel.sum())
            if k == 0:
                continue
            b = bx[sel] if bx is not None else None
            p = pres[sel] if pres is not None else None
            all_preds, scores = model(imgs[sel], b, presence=p)
            preds = get_best_mask(all_preds, scores)
            l = criterion(preds, masks[sel], pos_weight=pos_weight)
            for _ in range(refine_passes):
                r = as_refine_map(preds, model.mask_input_size[0])
                all_preds, scores = model(imgs[sel], b, refine=r, presence=p)
                preds = get_best_mask(all_preds, scores)
                l = refine_lambda * l + criterion(preds, masks[sel], pos_weight=pos_weight)
            loss = loss + (k / n) * l
            if out is None:
                out = preds.new_zeros((n,) + tuple(preds.shape[1:]))
            out[sel] = preds.detach()
        return loss, out

    # Accumulo del gradiente: con l'encoder scongelato il batch che sta in
    # memoria e' piu' piccolo di quello che serve per un confronto onesto coi
    # run precedenti. accum_steps batch da batch_size fanno un passo di
    # ottimizzazione equivalente a batch_size * accum_steps.
    accum = max(1, int(cfg.get("accum_steps", 1)))
    if accum > 1:
        print(f"  accumulo del gradiente: {accum} passi -> batch efficace "
              f"{cfg['batch_size'] * accum}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t_loss = t_iou = t_miou = 0.0
        n_batch = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['epochs']} [TRAIN]")
        for i_batch, batch in enumerate(pbar):
            # YOLO gira sull'immagine NATIVA e gia' aumentata, prima dell'upsample
            # a TARGET_SIZE e prima del trasferimento su GPU (ultralytics vuole
            # numpy su CPU).
            found = dense = pres = None; n_box = 0
            if clf is not None:
                pres = clf.p_for(batch["image"])
            if clf_mode:
                # il prompt VIENE dal classificatore: sentinella o tutta immagine
                h0, w0 = batch["image"].shape[-2], batch["image"].shape[-1]
                batch = dict(batch)
                batch["boxes"] = clf_boxes(pres, clf_thr, h0, w0).cpu()
                n_box = int((pres >= clf_thr).sum())
                pres = None       # non anche come mappa densa: sarebbe doppio
            if yolo is not None and dense_mode:
                dense, n_box = yolo.dense_for(batch["image"], model.mask_input_size[0])
                dense = dense.to(DEVICE)
            elif yolo is not None:
                boxes_native, found = yolo.boxes_for(batch["image"])
                batch = dict(batch); batch["boxes"] = boxes_native
            if oracolo:
                # Box vero SOLO dove c'e' una frana. Sulle patch vuote nessun
                # box: e' cio' che farebbe un detector perfetto, che su un
                # negativo non emette niente. Con la sentinella [0,0,10,10] il
                # box stesso diceva "qui non c'e' niente", ed era la fuga che
                # rendeva lo 0.7501 privo di significato.
                found = batch["mask"].flatten(1).sum(1) > 0
            elif cfg['prompt_source'] == 'gt_box':
                # Box su OGNI campione, sentinella compresa: un gruppo solo,
                # ma passando da _forward_split il raffinamento non scavalca
                # piu' il box.
                found = torch.ones(batch["mask"].shape[0], dtype=torch.bool)

            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)
            if jitter > 0:
                boxes = jitter_boxes(boxes, TARGET_SIZE, rel_std=jitter)
            if drop > 0 and torch.rand(1).item() < drop:
                boxes = None          # questo batch senza prompt

            with torch.amp.autocast(device_type=DEVICE):
                if clf_mode:
                    # Il box del classificatore c'e' per OGNI campione (o la
                    # sentinella o quello a tutta immagine), quindi input_boxes
                    # e' uniforme sul batch e non serve _forward_split.
                    all_preds, scores = model(imgs, boxes)
                    preds = get_best_mask(all_preds, scores)
                    loss = criterion(preds, masks, pos_weight=pos_weight)
                    for _ in range(refine_passes):
                        r = as_refine_map(preds, model.mask_input_size[0])
                        all_preds, scores = model(imgs, boxes, refine=r)
                        preds = get_best_mask(all_preds, scores)
                        loss = refine_lambda * loss + criterion(preds, masks, pos_weight=pos_weight)
                    iou_fg, _, miou, _, _, _ = calculate_metrics(preds.detach(), masks)
                elif found is not None:
                    # PRIMA del ramo denso: `usa_mask` e' vero anche solo per il
                    # raffinamento, e controllato per primo scavalcava questo
                    # ramo, cioe' il box. Qui dentro il raffinamento c'e'.
                    loss, preds = _forward_split(imgs, masks, boxes, found.to(DEVICE), pres)
                    iou_fg, _, miou, _, _, _ = calculate_metrics(preds, masks)
                elif usa_mask:
                    # niente canale sparse: il prompt e' TUTTO nella mappa, e
                    # chi non ha proposte riceve una mappa costante (neutra),
                    # quindi non serve dividere il batch.
                    loss, preds = _dense_forward(imgs, masks, dense, pres)
                    iou_fg, _, miou, _, _, _ = calculate_metrics(preds.detach(), masks)
                else:
                    all_preds, scores = model(imgs, boxes)
                    preds = get_best_mask(all_preds, scores)
                    loss = criterion(preds, masks, pos_weight=pos_weight)
                    iou_fg, _, miou, _, _, _ = calculate_metrics(preds.detach(), masks)
            scaler.scale(loss / accum).backward()
            ultimo = (i_batch + 1) == n_batch
            if (i_batch + 1) % accum == 0 or ultimo:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            t_loss += loss.item(); t_iou += iou_fg; t_miou += miou
            post = dict(loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}", miou=f"{miou:.4f}")
            if clf_mode:
                post["pos"] = f"{n_box}/{imgs.shape[0]}"
            if usa_mask:
                if dense_mode:
                    post["box"] = str(n_box)
                    post["k"] = f"{float(torch.exp(model.dense_log_scale)):.2f}"
                if clf is not None:
                    post["kc"] = f"{float(torch.exp(model.clf_log_scale)):.2f}"
                if refine_passes:
                    post["kr"] = f"{float(torch.exp(model.refine_log_scale)):.2f}"
            # Non piu' in alternativa a `kr`: con box e raffinamento insieme
            # devono vedersi entrambi. `box` nel log e' la prova che il box
            # entra davvero — e' mancando lui che si e' scoperto il difetto.
            if found is not None:
                post["box"] = f"{int(found.sum())}/{found.numel()}"
            pbar.set_postfix(**post)

        model.eval()
        v_loss = v_iou = 0.0
        v_each = []
        mi_tp = mi_fp = mi_fn = 0.0     # accumulatori per la micro
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[VAL]"):
                # Stesso regime del training: i box vengono da YOLO, mai dalla GT.
                # NIENTE jitter e NIENTE dropout qui — la validazione deve
                # riflettere cio' che il sistema fara' davvero a inferenza.
                found = dense = pres = None
                if clf is not None:
                    pres = clf.p_for(batch["image"])
                if clf_mode:
                    h0, w0 = batch["image"].shape[-2], batch["image"].shape[-1]
                    batch = dict(batch)
                    batch["boxes"] = clf_boxes(pres, clf_thr, h0, w0).cpu()
                    pres = None
                if yolo is not None and dense_mode:
                    dense, _ = yolo.dense_for(batch["image"], model.mask_input_size[0])
                    dense = dense.to(DEVICE)
                elif yolo is not None:
                    boxes_native, found = yolo.boxes_for(batch["image"])
                    batch = dict(batch); batch["boxes"] = boxes_native
                if oracolo:
                    found = batch["mask"].flatten(1).sum(1) > 0
                elif cfg['prompt_source'] == 'gt_box':
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
                        v_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                        iou_fg, _, _, _, _, _ = calculate_metrics(preds, masks)
                        v_each.append(iou_per_sample(preds, masks).cpu())
                    elif found is not None:
                        # stesso ordine del training: il box prima del denso
                        l, preds = _forward_split(imgs, masks, boxes, found.to(DEVICE), pres)
                        v_loss += l.item()
                        iou_fg, _, _, _, _, _ = calculate_metrics(preds, masks)
                        v_each.append(iou_per_sample(preds, masks).cpu())
                    elif usa_mask:
                        l, preds = _dense_forward(imgs, masks, dense, pres)
                        v_loss += l.item()
                        iou_fg, _, _, _, _, _ = calculate_metrics(preds, masks)
                        v_each.append(iou_per_sample(preds, masks).cpu())
                    else:
                        all_preds, scores = model(imgs, boxes)
                        preds = get_best_mask(all_preds, scores)
                        v_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                        iou_fg, _, _, _, _, _ = calculate_metrics(preds, masks)
                        v_each.append(iou_per_sample(preds, masks).cpu())
                v_iou += iou_fg
                # accumulo per la MICRO: tp/fp/fn su tutti i pixel, divisione
                # una volta sola alla fine. E' l'unica metrica confrontabile con
                # la letteratura, e non regala nulla alle patch vuote indovinate.
                p_ = F.interpolate(preds.float(), masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
                p_ = (torch.sigmoid(p_) > 0.5).float(); g_ = masks.unsqueeze(1).float()
                mi_tp += float((p_ * g_).sum())
                mi_fp += float((p_ * (1 - g_)).sum())
                mi_fn += float(((1 - p_) * g_).sum())

        micro_iou = mi_tp / max(mi_tp + mi_fp + mi_fn, 1.0)
        micro_f1 = 2 * mi_tp / max(2 * mi_tp + mi_fp + mi_fn, 1.0)
        avg_val_loss, avg_val_iou = v_loss / len(val_loader), v_iou / len(val_loader)
        nt = len(train_loader)
        scheduler.step(avg_val_loss)

        # Le due colonne separate. La media complessiva mescola due compiti
        # diversi: sulle patch VUOTE il punteggio e' 1.0 o 0.0 secco (c'e' o non
        # c'e'), su quelle con frana e' graduale (quanto bene ne ricalchi i
        # contorni). Guardando solo la media non si vede quale si sta muovendo.
        iou_pos = iou_neg = None
        if v_each and sum(len(x) for x in v_each) == len(val_pos):
            each = torch.cat(v_each)
            iou_pos = float(each[val_pos].mean())
            iou_neg = float(each[~val_pos].mean()) if (~val_pos).any() else float("nan")

        msg = f"EPOCA {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}"
        if iou_pos is not None:
            msg += f"   [frane {iou_pos:.4f} | vuote {iou_neg:.4f}]"
        msg += f"   micro {micro_iou:.4f}"
        print(msg)

        riga = {"epoca": epoch + 1,
                "train_loss": t_loss / nt, "train_iou": t_iou / nt, "train_miou": t_miou / nt,
                "val_loss": avg_val_loss, "val_iou": avg_val_iou,
                "val_micro_iou": micro_iou, "val_micro_f1": micro_f1,
                "lr": optimizer.param_groups[0]["lr"]}
        if iou_pos is not None:
            riga["val_iou_frane"] = iou_pos
            riga["val_iou_vuote"] = iou_neg
        if dense_mode:
            riga["k"] = float(torch.exp(model.dense_log_scale))
        if clf is not None:
            riga["kc"] = float(torch.exp(model.clf_log_scale))
        if refine_passes:
            riga["kr"] = float(torch.exp(model.refine_log_scale))

        # SU QUALE NUMERO SELEZIONARE IL CHECKPOINT.
        #
        # Il totale mescola due compiti: "c'e' o non c'e'" sulle patch vuote (il
        # 40% del test, punteggio 1.0 o 0.0 secco) e "quanto bene ricalchi i
        # contorni" su quelle con frana. Ma le vuote a inferenza le sistema il
        # GATE, che e' esterno al modello e vale +0.03 su qualunque checkpoint.
        #
        # Selezionare sul totale significa quindi premiare un modello TIMIDO —
        # penalizzandolo per l'unica cosa che poi correggiamo comunque. E' un
        # rischio concreto con `loss_per_image: true`, che alza `frane` e
        # abbassa `vuote` di proposito: il totale si appiattisce e si finisce
        # per salvare l'epoca sbagliata, senza modo di recuperare l'altra.
        #
        # `select_on: "frane"` seleziona sulla sola colonna che il modello
        # controlla davvero. Con train_positives_only e' obbligato: quel modello
        # non ha mai visto una patch vuota, quindi il suo totale non significa
        # nulla.
        # "micro" e' il criterio da usare quando il numero che si vuole
        # riportare e' quello confrontabile con la letteratura: selezionare
        # sulla macro sarebbe disallineato, perche' quella premia il riconoscere
        # le patch vuote e la micro non gliene da' credito.
        scelta = "frane" if solo_pos else cfg.get("select_on", "totale")
        if scelta == "micro":
            criterio, riga["criterio"] = micro_iou, "val_micro_iou"
        elif scelta == "frane" and iou_pos is not None:
            criterio, riga["criterio"] = iou_pos, "val_iou_frane"
        else:
            criterio, riga["criterio"] = avg_val_iou, "val_iou"

        if criterio > best_iou:
            best_iou = criterio
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, cfg["checkpoint_name"])
            print(f"  Record IoU: {best_iou:.4f} — checkpoint salvato in {cfg['checkpoint_name']}")
            patience_counter = 0
            riga["record"] = True
        else:
            patience_counter += 1

        # Scritto a OGNI epoca, non alla fine: se il run muore o viene ucciso,
        # quello che ha gia' fatto resta.
        record["epoche"].append(riga)
        record["best_val_iou"] = best_iou
        record["ultima_epoca"] = epoch + 1
        write_json(hist_path, record)

        if patience_counter >= cfg["patience"]:
            print(f"\nEARLY STOPPING — Miglior Val IoU: {best_iou:.4f}")
            record["stop"] = "early_stopping"
            write_json(hist_path, record)
            break


if __name__ == "__main__":
    train(load_config())

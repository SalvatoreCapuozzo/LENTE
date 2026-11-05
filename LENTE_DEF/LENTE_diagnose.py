"""
LENTE_diagnose.py — dove va a finire la IoU, e quanto vale ripulire le patch vuote.
====================================================================================

Nasce da una misura precisa: la IoU macro per campione e' la media su 759 patch di
cui **306 sono VUOTE**, e su una patch vuota il punteggio e' tutto-o-niente —
1.0 se il modello non disegna niente, ~0.0 se accende anche un solo pixel.

Il 40% del punteggio si decide quindi su una domanda binaria (c'e' o non c'e'),
non sulla qualita' della segmentazione. Guardando solo il totale i due contributi
sono indistinguibili, e infatti due modelli con lo stesso totale possono essere
opposti dentro. MISURATO:

    modello        vuote     con frana   totale
    denso          0.9346     0.3532     0.5876
    prompt-free    0.7843     0.4468     0.5829

Il canale denso non segmentava meglio: era piu' prudente. Vinceva sulle vuote e
perdeva sulle frane.

Questo script separa i due contributi e valuta tre modi di ripulire le vuote:

  * SOGLIA        alzare il taglio a 0.5 della sigmoide (globale)
  * GATE su AREA  azzerare la maschera se accende meno di N pixel (nessun
                  modulo nuovo: e' il modello che si auto-diagnostica)
  * GATE su YOLO  azzerare se il detector non propone nulla nemmeno a conf 0.02

METODO: ogni parametro (soglia, N, conf) viene scelto sul VALIDATION e poi
applicato al TEST **una volta sola**. Sceglierlo guardando il test sarebbe la
stessa colpa del box della ground truth, spostata di un modulo: si otterrebbe un
numero che non si riproduce.

Uso:
    python LENTE_diagnose.py
Modificare RUNS per scegliere quali checkpoint analizzare. `refine_passes` va
passato QUI, non letto dal config, cosi' un config disallineato non puo'
falsificare la misura in silenzio.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    build_model, get_best_mask, as_refine_map, write_json,
)

OUT_JSON = "diagnose_l4s.json"

# (checkpoint, prompt_source, target_size, canale denso, passaggi di raffinamento)
RUNS = [
    ("best_lente_l4s_refine.pth",  "yolo_box", 512, True,  1),
    ("best_lente_l4s_dense.pth",   "yolo_box", 512, True,  0),
    ("best_lente_l4s_free512.pth", "none",     512, False, 0),
]

SOGLIE = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
AREE   = (0, 5, 10, 15, 25, 40, 50, 75, 100, 150, 200, 300)
CONF   = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)
CLF    = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
# Punteggi del classificatore binario (LENTE_classifier.py scores). Se il file
# non c'e', quel gate viene semplicemente saltato.
CLF_SCORES = "clf_scores_l4s.npz"


# ------------------------------------------------------------------
# raccolta
# ------------------------------------------------------------------

def collect(model, prov, subset, cfg, target, dense_mode, refine_passes):
    """
    Una passata sul sottoinsieme. Ritorna, alla risoluzione NATIVA:

      P (N,H,W) float16  probabilita' predette
      G (N,H,W) uint8    maschere vere
      C (N,)    float32  confidenza MASSIMA di YOLO sulla patch, a soglia 0.02

    Si salvano le probabilita' invece delle maschere binarie perche' cosi' tutte
    le soglie si valutano dopo, a costo zero, senza rifare la forward.

    `C` viene raccolto anche in regime prompt-free: li' YOLO non entra nel
    modello, ma resta un segnale esterno utilizzabile come gate.
    """
    dev = cfg["device"]
    loader = DataLoader(TransformSubset(subset, A.Compose([ToTensorV2()])), batch_size=8)
    P, G, C = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="  raccolta", leave=False):
            imgs, masks = b["image"].to(dev), b["mask"].to(dev)
            d = None
            if dense_mode:
                d, _ = prov.dense_for(b["image"], model.mask_input_size[0])
                d = d.to(dev)
            im = F.interpolate(imgs, (target, target), mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type=dev):
                out, sc = model(im, None, dense=d)
                logits = get_best_mask(out, sc)
                # STESSO numero di passaggi del training: valutare a un passaggio
                # un modello addestrato a due significa misurarlo a meta' lavoro.
                for _ in range(refine_passes):
                    r = as_refine_map(logits, model.mask_input_size[0])
                    out, sc = model(im, None, dense=d, refine=r)
                    logits = get_best_mask(out, sc)
            pr = F.interpolate(logits.float(), masks.shape[-2:],
                               mode="bilinear", align_corners=False)
            P.append(torch.sigmoid(pr).squeeze(1).cpu().numpy().astype(np.float16))
            G.append(masks.cpu().numpy().astype(np.uint8))

            from LENTE_yolo_cascade import _for_yolo
            res = prov.model.predict([_for_yolo(x.numpy(), cfg) for x in b["image"]],
                                     conf=0.02, imgsz=prov.imgsz, max_det=10, verbose=False)
            C += [float(r.boxes.conf.max()) if (r.boxes is not None and len(r.boxes)) else 0.0
                  for r in res]
    return np.concatenate(P), np.concatenate(G), np.array(C, dtype=np.float32)


# ------------------------------------------------------------------
# metrica
# ------------------------------------------------------------------

def per_sample_iou(P, G, thr=0.5, gate=None):
    """
    IoU per campione, stessa formula di calculate_metrics (stesso epsilon), cosi'
    i numeri sono confrontabili con quelli stampati da LENTE_test.

    `gate`: maschera booleana (N,) — dove e' True la predizione viene azzerata.
    Su una patch VUOTA azzerare vale 1.0; su una patch con frana vale 0.0.
    """
    pr = (P > thr).reshape(len(P), -1)
    if gate is not None:
        pr = pr & ~gate[:, None]
    gt = G.reshape(len(G), -1) > 0
    tp = (pr & gt).sum(1); fp = (pr & ~gt).sum(1); fn = (~pr & gt).sum(1)
    return (tp + 1e-7) / (tp + fp + fn + 1e-7)


def _split(v, empty):
    return v[empty].mean(), v[~empty].mean(), v.mean()


def micro(P, G, thr=0.5, gate=None):
    """
    IoU e F1 MICRO (pooled) sulla classe frana: TP, FP e FN accumulati su TUTTI
    i pixel del test set e poi divisi una volta sola.

    E' LA METRICA DELLA LETTERATURA, ed e' un'altra cosa rispetto alla macro che
    usiamo per l'analisi:

      macro  ->  IoU di ogni patch, poi media. Una patch VUOTA predetta vuota
                 vale 1.0 e alza il totale, pur non avendo segmentato nulla.
      micro  ->  una patch vuota predetta vuota contribuisce ZERO a tutti e tre
                 i conteggi: non regala niente. Il punteggio lo fanno solo i
                 pixel di frana, quindi le frane GRANDI pesano di piu'.

    Riferimenti su Landslide4Sense, F1 sulla classe frana:
        Multispectral U-Net .......... 77.83%  ->  IoU 0.637
        vincitore competizione 2022 .. 74.54%  ->  IoU 0.594
    con  IoU = F1 / (2 - F1),  esatta per il caso binario.

    ATTENZIONE nel confronto: loro valutano sul test ufficiale della
    competizione, noi su uno split 60/20/20 di TrainData. Anche a parita' di
    metrica non sono le stesse immagini.

    NON si usa la mIoU: i pixel di frana sono il 3.95% del totale, quindi la IoU
    dello sfondo vale ~0.98 comunque e mediarla con quella della frana comprime
    proprio le differenze che interessano.
    """
    pr = (P > thr).reshape(len(P), -1)
    if gate is not None:
        pr = pr & ~gate[:, None]
    gt = G.reshape(len(G), -1) > 0
    tp = float((pr & gt).sum()); fp = float((pr & ~gt).sum()); fn = float((~pr & gt).sum())
    iou = tp / max(tp + fp + fn, 1.0)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
    return iou, f1


# ------------------------------------------------------------------

def analizza(nome, vP, vG, vC, tP, tG, tC, vS=None, tS=None, out=None):
    v_empty = ~(vG.reshape(len(vG), -1) > 0).any(1)
    t_empty = ~(tG.reshape(len(tG), -1) > 0).any(1)
    v_area = lambda thr: (vP > thr).reshape(len(vP), -1).sum(1)
    t_area = lambda thr: (tP > thr).reshape(len(tP), -1).sum(1)

    print(f"\n{'='*70}\n{nome}")
    print(f"val: {v_empty.sum()} vuote / {(~v_empty).sum()} con frana   |   "
          f"test: {t_empty.sum()} vuote / {(~t_empty).sum()} con frana\n{'='*70}")

    e, p, tot = _split(per_sample_iou(tP, tG), t_empty)
    mi, mf = micro(tP, tG)
    print(f"\nBASELINE (soglia 0.5, nessun gate)")
    print(f"  TEST  macro   vuote {e:.4f} | con frana {p:.4f} | TOTALE {tot:.4f}")
    print(f"  TEST  MICRO   IoU {mi:.4f} | F1 {mf:.4f}"
          f"   <- confrontabile con la letteratura (migliore pubblicato: F1 0.7783 / IoU 0.637)")
    base = tot
    base_mi = mi
    if out is not None:
        out.update(nome=nome, n_val=int(len(vG)), n_test=int(len(tG)),
                   val_vuote=int(v_empty.sum()), test_vuote=int(t_empty.sum()),
                   baseline={"vuote": float(e), "con_frana": float(p), "totale": float(tot),
                             "micro_iou": float(mi), "micro_f1": float(mf)},
                   gate={})

    def riporta(titolo, scelto, dettaglio, v_iou, t_iou, t_gate=None, thr=0.5):
        ve, vp, vt = _split(v_iou, v_empty)
        te_, tp_, tt = _split(t_iou, t_empty)
        mi, mf = micro(tP, tG, thr, t_gate)
        print(f"\n{titolo}")
        print(f"  scelto sul VAL: {scelto}   ({dettaglio})")
        print(f"  val   vuote {ve:.4f} | con frana {vp:.4f} | TOTALE {vt:.4f}")
        print(f"  TEST  vuote {te_:.4f} | con frana {tp_:.4f} | TOTALE {tt:.4f}"
              f"   ({tt-base:+.4f})")
        print(f"  MICRO IoU {mi:.4f} | F1 {mf:.4f}   ({mi-base_mi:+.4f})")
        if out is not None:
            out["gate"][titolo] = {
                "scelto_sul_val": scelto, "dettaglio": dettaglio,
                "val":  {"vuote": float(ve), "con_frana": float(vp), "totale": float(vt)},
                "test": {"vuote": float(te_), "con_frana": float(tp_), "totale": float(tt),
                         "guadagno": float(tt - base),
                         "micro_iou": float(mi), "micro_f1": float(mf),
                         "micro_guadagno": float(mi - base_mi)}}

    # --- 1. soglia globale -------------------------------------------------
    best = max(SOGLIE, key=lambda t: per_sample_iou(vP, vG, t).mean())
    riporta("SOGLIA GLOBALE", f"{best:.2f}", "taglio della sigmoide",
            per_sample_iou(vP, vG, best), per_sample_iou(tP, tG, best), thr=best)

    # --- 2. gate sull'area predetta ---------------------------------------
    best_n = max(AREE, key=lambda n: per_sample_iou(vP, vG, 0.5, v_area(0.5) < n).mean())
    g_v, g_t = v_area(0.5) < best_n, t_area(0.5) < best_n
    riporta("GATE sull'AREA PREDETTA", f"{best_n} pixel",
            f"azzera {g_t[t_empty].mean():.1%} delle vuote, "
            f"{g_t[~t_empty].mean():.1%} delle frane",
            per_sample_iou(vP, vG, 0.5, g_v), per_sample_iou(tP, tG, 0.5, g_t), g_t)

    # --- 3. gate sulla confidenza di YOLO ---------------------------------
    best_c = max(CONF, key=lambda c: per_sample_iou(vP, vG, 0.5, vC < c).mean())
    g_v, g_t = vC < best_c, tC < best_c
    riporta("GATE su YOLO", f"conf_max < {best_c:.2f}",
            f"azzera {g_t[t_empty].mean():.1%} delle vuote, "
            f"{g_t[~t_empty].mean():.1%} delle frane",
            per_sample_iou(vP, vG, 0.5, g_v), per_sample_iou(tP, tG, 0.5, g_t), g_t)

    # --- 3b. gate sul CLASSIFICATORE binario ------------------------------
    if vS is not None:
        best_s = max(CLF, key=lambda s: per_sample_iou(vP, vG, 0.5, vS < s).mean())
        g_v, g_t = vS < best_s, tS < best_s
        riporta("GATE sul CLASSIFICATORE", f"p(frana) < {best_s:.2f}",
                f"azzera {g_t[t_empty].mean():.1%} delle vuote, "
                f"{g_t[~t_empty].mean():.1%} delle frane",
                per_sample_iou(vP, vG, 0.5, g_v), per_sample_iou(tP, tG, 0.5, g_t), g_t)

    # --- 4. gate COMBINATI -------------------------------------------------
    # Due modi opposti di mettere insieme due segnali:
    #   E  -> azzera solo se TUTTI tacciono. Piu' prudente: uccide meno frane
    #         vere, ma ripulisce meno vuote.
    #   O  -> azzera se ANCHE UNO SOLO tace. Piu' aggressivo: il contrario.
    # Quale convenga non e' deducibile a tavolino — dipende da quanto gli errori
    # dei due segnali si sovrappongono, e lo decide il val.
    va, ta = v_area(0.5), t_area(0.5)

    def cerca(titolo, coppie):
        """coppie: lista di (etichetta, maschera_val, maschera_test)."""
        et, gv, gt = max(coppie, key=lambda c: per_sample_iou(vP, vG, 0.5, c[1]).mean())
        riporta(titolo, et,
                f"azzera {gt[t_empty].mean():.1%} delle vuote, "
                f"{gt[~t_empty].mean():.1%} delle frane",
                per_sample_iou(vP, vG, 0.5, gv), per_sample_iou(tP, tG, 0.5, gt), gt)

    cerca("GATE COMBINATO (area E yolo)",
          [(f"area < {n} px  e  conf < {c:.2f}", (va < n) & (vC < c), (ta < n) & (tC < c))
           for n in AREE for c in CONF])

    if vS is not None:
        cerca("GATE COMBINATO (classificatore E area)",
              [(f"p(frana) < {s:.2f}  e  area < {n} px",
                (vS < s) & (va < n), (tS < s) & (ta < n))
               for s in CLF for n in AREE])
        cerca("GATE COMBINATO (classificatore O area)",
              [(f"p(frana) < {s:.2f}  oppure  area < {n} px",
                (vS < s) | (va < n), (tS < s) | (ta < n))
               for s in CLF for n in AREE])
        cerca("GATE COMBINATO (classificatore E yolo)",
              [(f"p(frana) < {s:.2f}  e  conf < {c:.2f}",
                (vS < s) & (vC < c), (tS < s) & (tC < c))
               for s in CLF for c in CONF])

    # --- 5. la soglia ottimizzata PER LA MICRO -----------------------------
    # La soglia che massimizza la macro NON e' quella che massimizza la micro, e
    # non e' un dettaglio: la macro premia il non sporcare le patch vuote (1.0
    # secco), la micro conta solo pixel e quindi premia il recall sulle frane
    # grandi. Sono due ottimi diversi, e il numero destinato al confronto con la
    # letteratura va ottimizzato per la SUA metrica, non ereditato dall'altra.
    best_m = max(SOGLIE, key=lambda t: micro(vP, vG, t)[0])
    mi, mf = micro(tP, tG, best_m)
    print(f"\nMICRO OTTIMIZZATA   soglia {best_m:.2f} (scelta sul val)")
    print(f"  TEST  IoU {mi:.4f} | F1 {mf:.4f}   ({mi-base_mi:+.4f} sulla micro a 0.5)")
    if out is not None:
        out["micro_ottimizzata"] = {"soglia": float(best_m),
                                    "iou": float(mi), "f1": float(mf)}

    # --- tetto teorico ----------------------------------------------------
    v = per_sample_iou(tP, tG)
    tetto = (t_empty.sum() * 1.0 + v[~t_empty].sum()) / len(v)
    print(f"\nTETTO con un classificatore PERFETTO: {tetto:.4f}  ({tetto-base:+.4f})")
    print("  (tutte le vuote a 1.0, le frane come sono adesso)")
    if out is not None:
        out["tetto_classificatore_perfetto"] = float(tetto)


def main():
    risultati = []
    vS = tS = None
    if os.path.exists(CLF_SCORES):
        z = np.load(CLF_SCORES)
        vS, tS = z["val_score"], z["test_score"]
        print(f"punteggi del classificatore da {CLF_SCORES}: "
              f"{len(vS)} val, {len(tS)} test")

    for ckpt, prompt, target, dense_mode, rp in RUNS:
        try:
            sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
        except Exception as exc:
            print(f"\n{ckpt}: {exc}")
            continue
        # adapter_dim dal checkpoint: con un valore sbagliato load_state_dict
        # scarterebbe gli adapter in SILENZIO e si misurerebbe un modello monco.
        ad = [v.shape[0] for k, v in sd.items()
              if k.endswith("adapters.0.adapter.0.weight")]
        cfg = load_config("config.yaml", dataset="l4s", prompt_source=prompt)
        cfg["target_size"] = target
        if ad:
            cfg["adapter_dim"] = ad[0]

        full = build_dataset(cfg)
        _, val_s, test_s, _ = build_train_val_test_subsets(full, cfg)
        model = build_model(cfg, cfg["device"])
        miss, unexp = model.load_state_dict(sd, strict=False)
        model.eval()
        if len(miss) > 3 or unexp:
            print(f"  ATTENZIONE {ckpt}: mancanti={len(miss)} inattese={len(unexp)} "
                  f"— i numeri potrebbero non valere")

        from LENTE_yolo_cascade import YoloBoxProvider
        prov = YoloBoxProvider(cfg["yolo_weights"], cfg)

        vP, vG, vC = collect(model, prov, val_s,  cfg, target, dense_mode, rp)
        tP, tG, tC = collect(model, prov, test_s, cfg, target, dense_mode, rp)
        out = {"checkpoint": ckpt, "prompt_source": prompt, "target_size": target,
               "adapter_dim": cfg["adapter_dim"], "dense": dense_mode,
               "refine_passes": rp, "chiavi_mancanti": len(miss)}
        analizza(f"{ckpt}   ({prompt}@{target}, adapter={cfg['adapter_dim']}, "
                 f"dense={dense_mode}, refine={rp})",
                 vP, vG, vC, tP, tG, tC, vS, tS, out)
        risultati.append(out)
        write_json(OUT_JSON, {"quando": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "classificatore": CLF_SCORES if vS is not None else None,
                              "run": risultati})

        del model, vP, vG, tP, tG
        torch.cuda.empty_cache()

    print(f"\nScritto {OUT_JSON}")


if __name__ == "__main__":
    main()

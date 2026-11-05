#!/usr/bin/env python
"""
LENTE_prompt_audit.py — quanto vale il prompt, prima di addestrare qualsiasi cosa.
=================================================================================
Risponde a una domanda sola: il box della ground truth quanta informazione
contiene davvero? Non serve alcun modello, sono statistiche sul dataset.

Tre numeri per dataset:

  1) AREA DEL BOX / area immagine.
     Se e' ~1.0 il box coincide con l'immagine e di *localizzazione* non regala
     nulla. Se e' piccolo, il box sta indicando dove guardare.

  2) IoU DEL BOX RIEMPITO ( = |GT| / |box| ).
     E' il punteggio che si ottiene predicendo "tutto frana dentro il box".
     La scorciatoia degenere: un modello promptato col GT box deve battere
     questo numero per aver imparato qualcosa. Su Sen12 valeva 0.078 contro
     0.128 del modello addestrato.

  3) ASIMMETRIA POSITIVI / NEGATIVI.
     I negativi ricevono il box fittizio [0,0,10,10] (LENTE_utils._bbox_from_mask).
     Se l'area del box separa positivi e negativi, il prompt sta rivelando
     l'ETICHETTA, non la posizione — un leak che esiste anche quando il box dei
     positivi copre tutta l'immagine.

  4) POS_WEIGHT REALE vs QUELLO CHE USA IL CONFIG.
     `pos_weight_mode: "heuristic"` assume ls_frac=0.35, cioe' che una patch
     positiva sia coperta di frana al 35%. Le specifiche pubblicate di
     Landslide4Sense dicono 2.3% di pixel-frana sul totale con ~58% di patch
     positive, cioe' ~4% dentro le positive: la costante e' ~9x fuori scala e
     il pos_weight che ne esce e' ~10x troppo basso.
     Con il GT box contava poco (la predizione era gia' incanalata dentro il
     box, dove la densita' di frana e' alta). Prompt-free il modello affronta
     lo sbilanciamento vero, e un pos_weight troppo basso lo fa collassare su
     "tutto sfondo".

Uso:
    python LENTE_prompt_audit.py -d bijie
    python LENTE_prompt_audit.py -d l4s
"""

import argparse

import numpy as np

from LENTE_utils import load_config, build_dataset, _bbox_from_mask
import torch


def _pct(x):
    return f"{100.0 * x:6.2f}%"


def _summary(name, values):
    """Riga di riepilogo di una distribuzione (vuota compresa)."""
    if not values:
        return f"  {name:<28} n=0"
    a = np.asarray(values, dtype=np.float64)
    return (f"  {name:<28} n={len(a):<6} "
            f"media {a.mean():.4f} | mediana {np.median(a):.4f} | "
            f"p10 {np.percentile(a, 10):.4f} | p90 {np.percentile(a, 90):.4f}")


def audit(cfg):
    ds = build_dataset(cfg)
    pos_set = set(ds.positive_indices)

    box_frac_pos, box_frac_neg = [], []   # area box / area immagine
    fill_iou = []                         # |GT| / |box|  sui positivi
    fg_frac = []                          # |GT| / area immagine, sui positivi
    px_fg = px_tot = 0                    # conteggi globali -> pos_weight reale

    for i in range(len(ds)):
        mask = torch.from_numpy(np.asarray(ds.get_mask(i)))
        h, w = mask.shape[-2], mask.shape[-1]
        area_img = float(h * w)

        x0, y0, x1, y1 = _bbox_from_mask(mask).tolist()
        # Box inclusivo sui pixel estremi, come lo interpreta SAM2.
        area_box = max(x1 - x0 + 1.0, 1.0) * max(y1 - y0 + 1.0, 1.0)

        gt = float(mask.sum())
        px_fg  += int(gt)
        px_tot += int(area_img)

        if i in pos_set:
            box_frac_pos.append(area_box / area_img)
            fill_iou.append(gt / area_box)      # GT contenuta nel box: IoU = |GT|/|box|
            fg_frac.append(gt / area_img)
        else:
            box_frac_neg.append(area_box / area_img)

    print("\n" + "=" * 72)
    print(f"AUDIT DEL PROMPT — dataset '{cfg['dataset_name']}'  ({len(ds)} campioni)")
    print("=" * 72)

    print("\n[1] Area del box / area immagine")
    print(_summary("positivi", box_frac_pos))
    print(_summary("negativi (box fittizio)", box_frac_neg))

    print("\n[2] IoU riempiendo il box (scorciatoia degenere, solo positivi)")
    print(_summary("|GT| / |box|", fill_iou))

    print("\n[3] Frazione di pixel-frana / immagine (solo positivi)")
    print(_summary("|GT| / |immagine|", fg_frac))

    # ── Lettura dei numeri ────────────────────────────────────────────
    print("\n" + "-" * 72)
    if box_frac_pos:
        mean_box = float(np.mean(box_frac_pos))
        mean_fill = float(np.mean(fill_iou))
        mean_fg = float(np.mean(fg_frac))
        print(f"Il box copre in media {_pct(mean_box)} dell'immagine.")
        if mean_box > 0.9:
            print("  -> praticamente TUTTA l'immagine: di localizzazione non regala nulla,")
            print("     e 'gt_box' e' quasi equivalente a 'image_box' sui positivi.")
        else:
            print(f"  -> il box concentra la ricerca su {_pct(mean_box)} dell'immagine:")
            print(f"     la densita' di frana passa da {_pct(mean_fg)} a {_pct(mean_fill)}"
                  f" ({mean_fill / max(mean_fg, 1e-9):.1f}x).")
        print(f"\nRiempire il box da IoU {mean_fill:.4f}. Un modello promptato col GT box")
        print("deve battere QUESTO, non lo zero, per aver imparato qualcosa.")

    if box_frac_pos and box_frac_neg:
        sep = float(np.mean(box_frac_pos)) / max(float(np.mean(box_frac_neg)), 1e-9)
        print(f"\nAsimmetria positivi/negativi: il box dei positivi e' {sep:.0f}x quello")
        print("dei negativi. Se questo rapporto e' grande, il prompt rivela l'ETICHETTA")
        print("(frana / non frana) a prescindere da quanto sia stretto sui positivi.")
    print("-" * 72)

    # ── [4] pos_weight: quello vero contro quello che uscirebbe dal config ──
    n_pos, n_neg = len(box_frac_pos), len(box_frac_neg)
    real_ratio = (px_tot - px_fg) / max(px_fg, 1)
    ls_frac = 0.35   # la costante cablata in LENTE_utils.build_train_val_test_subsets
    pixel_ls = n_pos * ls_frac
    pixel_bg = n_neg + n_pos * (1.0 - ls_frac)
    heur = pixel_bg / max(pixel_ls, 1.0)
    # ls_frac implicito nei dati: copertura media di frana dentro una patch positiva
    ls_frac_real = float(np.mean(fg_frac)) if fg_frac else 0.0

    print("\n[4] pos_weight della BCE")
    print(f"  pixel-frana sul totale        {_pct(px_fg / max(px_tot, 1))}")
    print(f"  copertura media nei positivi  {_pct(ls_frac_real)}   "
          f"(la formula 'heuristic' assume {_pct(ls_frac)})")
    print(f"  ---")
    print(f"  \"pixels\"    (rapporto reale)          {real_ratio:8.2f}")
    print(f"  \"sqrt\"      (smorzamento standard)    {real_ratio ** 0.5:8.2f}")
    print(f"  \"heuristic\" (formula attuale, 0.35)   {heur:8.2f}")
    print(f"  in uso adesso: \"{cfg.get('pos_weight_mode', 'pixels')}\"")

    if cfg.get("pos_weight_mode") == "heuristic" and heur > 0:
        print(f"\n  -> la modalita' attiva sotto-pesa la classe positiva di "
              f"{real_ratio / heur:.1f}x rispetto al rapporto reale.")
        print("     Con il GT box passava inosservato; prompt-free spinge il modello")
        print("     a predire tutto sfondo. Valutare \"pixels\" o \"sqrt\".")
    print("-" * 72 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Statistiche sul prompt box (nessun modello richiesto)")
    ap.add_argument("-d", "--dataset", default=None, help="nome dataset in config.yaml")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    audit(load_config(args.config, dataset=args.dataset))


if __name__ == "__main__":
    main()

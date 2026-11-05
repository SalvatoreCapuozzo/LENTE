"""
La focal loss aiuterebbe? Analisi su un checkpoint gia' addestrato, senza addestrare.

LA DOMANDA. `con frana` e' fermo attorno a 0.49 su una dozzina di configurazioni.
La focal toglie peso ai pixel gia' classificati bene e lo sposta su quelli
difficili. Serve pero' sapere in che regime stanno i pixel che oggi sbagliamo:

    p vicino a 0.5   ->  "difficili": il modello e' incerto, la focal li
                         solleva e c'e' segnale da cui imparare.
    p vicino a 0     ->  "invisibili": il modello non ha alcun segnale li'.
                         La focal li solleva lo stesso, ma il gradiente della
                         BCE su un positivo con p->0 e' gia' massimo (|p-1|->1)
                         e la modulazione (1-pt)^gamma tende a 1: la focal NON
                         aggiunge quasi nulla proprio dove serve.

Quindi il vero effetto della focal e' TOGLIERE peso agli esempi facili, non
aggiungerlo a quelli difficili. Se il peso sugli esempi facili e' gia' basso —
e `pos_weight` 6.44 piu' il termine Dice lo riducono gia' due volte — non
resta molto da spostare.

COSA MISURA. Sul VAL, per fascia di dimensione della componente connessa:
  * quanti pixel, che probabilita' media riceve il modello, quanti superano la soglia
  * quale QUOTA della loss BCE viene da ciascuna fascia
  * come quella quota cambia applicando la modulazione focal

Se passando a gamma=2 la quota delle frane piccole sale di poco, la focal non
sposta il problema e il run da cinque ore non serve.

    python analisi_focal.py --ckpt best_lente_l4s_free_ad128both_selmicro_1024px.pth \
           --adapter-where both --adapter-dim 128 --target-size 1024
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (load_config, build_dataset, build_train_val_test_subsets,
                         TransformSubset, build_model, get_best_mask)

try:
    from scipy import ndimage
except ImportError:
    raise SystemExit("Serve scipy per le componenti connesse")

# Fasce di dimensione in pixel NATIVI. La mediana L4S e' 29 px, il 34.5% delle
# frane sta sotto i 16 px — cioe' sotto una cella di decisione a 512.
FASCE = [(1, 16), (17, 64), (65, 256), (257, 10**9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--adapter-where", choices=["attn", "mlp", "both"], default=None)
    ap.add_argument("--adapter-dim", type=int, default=None)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--pos-weight", type=float, default=6.44)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config(a.config, dataset=a.dataset, prompt_source="none")
    if a.adapter_where is not None: cfg["adapter_where"] = a.adapter_where
    if a.adapter_dim is not None:   cfg["adapter_dim"] = a.adapter_dim
    if a.target_size is not None:   cfg["target_size"] = a.target_size
    DEV, T = cfg["device"], cfg["target_size"]

    full = build_dataset(cfg)
    _, val_subset, _, _ = build_train_val_test_subsets(full, cfg)
    loader = DataLoader(TransformSubset(val_subset, A.Compose([ToTensorV2()])),
                        batch_size=a.batch_size, shuffle=False, pin_memory=True)

    model = build_model(cfg, DEV)
    ck = torch.load(a.ckpt, map_location=DEV)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    print(f"\ncheckpoint: {a.ckpt}   target_size {T}   gamma {a.gamma}\n")

    # accumulatori: una riga per fascia, piu' lo sfondo
    n_px = np.zeros(len(FASCE) + 1)
    somma_p = np.zeros(len(FASCE) + 1)
    n_sopra = np.zeros(len(FASCE) + 1)
    bce_tot = np.zeros(len(FASCE) + 1)
    foc_tot = np.zeros(len(FASCE) + 1)
    isto_p = np.zeros((len(FASCE), 10))     # distribuzione di p, solo frane

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(DEV)
            masks = batch["mask"].to(DEV)                  # NATIVA, 128
            x = F.interpolate(imgs, (T, T), mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type=DEV):
                preds = get_best_mask(*model(x, None))
            # sulla griglia NATIVA, come la metrica
            preds = F.interpolate(preds.float(), masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
            p = torch.sigmoid(preds).squeeze(1).float().cpu().numpy()
            g = masks.float().cpu().numpy()

            for pi, gi in zip(p, g):
                # etichetta di fascia per ogni pixel: 0..3 frane, 4 sfondo
                fascia = np.full(gi.shape, len(FASCE), dtype=np.int64)
                if gi.sum() > 0:
                    lab, n = ndimage.label(gi > 0)
                    aree = np.bincount(lab.ravel())
                    for c in range(1, n + 1):
                        area = aree[c]
                        for k, (lo, hi) in enumerate(FASCE):
                            if lo <= area <= hi:
                                fascia[lab == c] = k
                                break

                pc = np.clip(pi, 1e-6, 1 - 1e-6)
                # BCE per pixel con pos_weight, identica a quella del training
                bce = -(a.pos_weight * gi * np.log(pc) + (1 - gi) * np.log(1 - pc))
                pt = pc * gi + (1 - pc) * (1 - gi)
                foc = bce * (1 - pt) ** a.gamma

                for k in range(len(FASCE) + 1):
                    m = fascia == k
                    if not m.any():
                        continue
                    n_px[k] += m.sum()
                    somma_p[k] += pi[m].sum()
                    n_sopra[k] += (pi[m] > 0.5).sum()
                    bce_tot[k] += bce[m].sum()
                    foc_tot[k] += foc[m].sum()
                    if k < len(FASCE):
                        isto_p[k] += np.histogram(pi[m], bins=10, range=(0, 1))[0]

    nomi = [f"{lo}-{hi} px" for lo, hi in FASCE[:-1]] + [f">{FASCE[-1][0]-1} px", "sfondo"]
    qb = bce_tot / bce_tot.sum()
    qf = foc_tot / foc_tot.sum()

    print(f"{'fascia':>12} {'pixel':>10} {'p media':>9} {'p>0.5':>8} "
          f"{'quota BCE':>10} {'quota focal':>12} {'variazione':>11}")
    for k in range(len(FASCE) + 1):
        if n_px[k] == 0:
            continue
        print(f"{nomi[k]:>12} {int(n_px[k]):>10,} {somma_p[k]/n_px[k]:>9.3f} "
              f"{n_sopra[k]/n_px[k]:>8.3f} {qb[k]:>10.3f} {qf[k]:>12.3f} "
              f"{qf[k]-qb[k]:>+11.3f}")

    print("\ndistribuzione della probabilita' sui pixel di FRANA (righe = fasce)")
    print(f"{'fascia':>12} " + " ".join(f"{i/10:>5.1f}" for i in range(10)))
    for k in range(len(FASCE)):
        tot = isto_p[k].sum()
        if tot == 0:
            continue
        print(f"{nomi[k]:>12} " + " ".join(f"{v:>5.2f}" for v in isto_p[k] / tot))

    print("""
COME LEGGERE
  quota focal - quota BCE  sulle fasce piccole:
     > +0.05   la focal sposta davvero peso sulle frane piccole: vale provarla
     ~  0      non cambia la distribuzione del gradiente: il run non serve

  distribuzione di p sulle frane piccole:
     massa attorno a 0.3-0.6   pixel INCERTI, la focal ha su cosa lavorare
     massa nel primo bin       pixel INVISIBILI: il gradiente della BCE e' gia'
                               massimo li', la focal non aggiunge quasi nulla
""")


if __name__ == "__main__":
    main()

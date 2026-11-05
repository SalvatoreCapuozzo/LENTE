#!/usr/bin/env python
"""
statistiche_ingresso.py — che scala hanno i dati che diamo agli encoder.

TerraCodec e Prithvi sono stati addestrati su ingressi STANDARDIZZATI con le
loro statistiche, cioe' valori attorno a 0 con deviazione 1. Noi gli passiamo
i valori di L4S attraverso un'affine inizializzata all'identita'. Se L4S sta
gia' in quella scala, l'identita' va bene; se no, un modello congelato vede
ingressi fuori dalla distribuzione su cui e' nato.

Per un ViT conta poco: la LayerNorm quasi all'ingresso assorbe un errore di
scala. Per un codec conta molto: la GDN e' una normalizzazione divisiva tarata
sulla distribuzione di addestramento, e un fattore sbagliato sull'ingresso ne
cambia il comportamento.

    python statistiche_ingresso.py
"""

import numpy as np

from LENTE_utils import load_config, build_dataset, build_train_val_test_subsets

NOMI = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
        "B9", "B10", "B11", "B12", "slope", "DEM"]


def main():
    cfg = load_config("config.yaml", dataset="l4s", prompt_source="none")
    full = build_dataset(cfg)
    tr, _, _, _ = build_train_val_test_subsets(full, cfg)

    rng = np.random.default_rng(0)
    scelti = rng.choice(tr.indices, size=min(400, len(tr.indices)), replace=False)
    campioni = []
    for i in scelti:
        x = full[int(i)]["image"].numpy()             # (C, H, W)
        campioni.append(x.reshape(x.shape[0], -1)[:, ::7])
    X = np.concatenate(campioni, axis=1)

    print(f"\n  {len(scelti)} patch del train, {X.shape[1]:,} pixel per banda\n")
    print(f"  {'banda':>6} {'media':>9} {'dev.std':>9} {'min':>9} {'max':>9}")
    print("  " + "-" * 46)
    for c in range(X.shape[0]):
        v = X[c]
        print(f"  {NOMI[c] if c < len(NOMI) else c:>6} {v.mean():>9.4f} "
              f"{v.std():>9.4f} {v.min():>9.4f} {v.max():>9.4f}")

    med = np.abs(X[:12].mean(1)).mean()
    dev = X[:12].std(1).mean()
    print(f"\n  sulle dodici bande ottiche: |media| tipica {med:.4f}, "
          f"deviazione tipica {dev:.4f}")
    print("  un modello addestrato su dati standardizzati si aspetta ~0 e ~1")
    if dev < 0.2 or dev > 5 or med > 1:
        print(f"\n  FUORI SCALA: con l'affine all'identita' il codec riceve "
              f"ingressi circa {1 / max(dev, 1e-9):.0f}x piu' "
              f"{'piccoli' if dev < 1 else 'grandi'} di quelli su cui e' "
              f"addestrato.")
    else:
        print("\n  in scala: la normalizzazione non spiega il risultato di "
              "TerraCodec, il problema e' altrove.")
    print()


if __name__ == "__main__":
    main()

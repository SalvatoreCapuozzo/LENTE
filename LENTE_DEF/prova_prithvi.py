#!/usr/bin/env python
"""
prova_prithvi.py — controllo di un minuto prima di impegnare sei ore.

Verifica le due sole cose che non si possono sapere senza il backbone vero:
che gli embedding posizionali si interpolino dal lato di addestramento a
quello che gli diamo, e in che FORMA escono le feature. `_mappa` conosce i
casi noti — lista di livelli, token con o senza CLS, mappa gia' fatta — e si
ferma a voce alta sugli altri: qui si vede subito quale caso e'.

    python prova_prithvi.py
    python prova_prithvi.py --prithvi-size 1024      # se hai memoria
"""

import argparse

import torch

from LENTE_utils import load_config, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("--prithvi-size", type=int, default=512)
    ap.add_argument("--prithvi-model", type=str, default=None)
    ap.add_argument("--target-size", type=int, default=1024)
    a = ap.parse_args()

    cfg = load_config(a.config, dataset=a.dataset, prompt_source="none")
    cfg["encoder"] = "prithvi"
    cfg["prithvi_size"] = a.prithvi_size
    cfg["target_size"] = a.target_size
    if a.prithvi_model:
        cfg["prithvi_model"] = a.prithvi_model
    dev = cfg["device"]

    print("\n--- 1. costruzione del modello -------------------------------")
    model = build_model(cfg, dev)
    enc = model.unet_encoder

    print("\n--- 2. uscita GREZZA del backbone ----------------------------")
    # unsqueeze(2) e non (1): il patch embed e' un Conv3d e vuole
    # (N, C, D, H, W). Deve restare uguale a quello che fa PrithviEncoder.
    x = torch.randn(1, 6, enc.lato, enc.lato, device=dev)
    with torch.no_grad():
        grezza = enc.backbone(x.unsqueeze(2))
    if isinstance(grezza, (list, tuple)):
        print(f"  lista di {len(grezza)} elementi; forme: "
              f"{[tuple(g.shape) for g in grezza[:4]]}")
    else:
        print(f"  tensore {tuple(grezza.shape)}")
    atteso = (enc.lato // 16) ** 2
    print(f"  token attesi a {enc.lato}px con patch 16: {atteso}")

    print("\n--- 3. le tre mappe per il decoder di SAM2 -------------------")
    s = cfg["target_size"]
    n_ch = cfg["n_optical"] + len(cfg.get("terrain_bands", []))
    with torch.no_grad():
        mappe = enc(torch.randn(1, cfg["n_optical"], s, s, device=dev))
    att = [(1, 32, s // 4, s // 4), (1, 64, s // 8, s // 8),
           (1, 256, s // 16, s // 16)]
    tutto = True
    for m, t in zip(mappe, att):
        ok = tuple(m.shape) == t
        tutto &= ok
        print(f"  {str(tuple(m.shape)):>22}  atteso {str(t):>22}  "
              f"{'ok' if ok else 'NON COMBACIA'}")

    print("\n--- 4. una forward completa ----------------------------------")
    with torch.no_grad():
        with torch.amp.autocast(device_type=dev):
            pred, punteggi = model(torch.randn(1, n_ch, s, s, device=dev), None)
    print(f"  maschere {tuple(pred.shape)} | punteggi "
          f"{tuple(punteggi.shape) if punteggi is not None else None}")

    n_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"\n  addestrabili in tutto: {n_tr:,}")
    print("\n  " + ("TUTTO A POSTO: puoi lanciare l'addestramento."
                    if tutto else
                    "LE FORME NON COMBACIANO: non lanciare, guarda il punto 2."))
    print()


if __name__ == "__main__":
    main()

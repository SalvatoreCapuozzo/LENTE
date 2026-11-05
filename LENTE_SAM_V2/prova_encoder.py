#!/usr/bin/env python
"""
prova_encoder.py — controllo di un minuto prima di impegnare sei ore.

Sostituisce prova_prithvi.py e vale per qualunque encoder sostitutivo.
Verifica cio' che non si puo' sapere senza i pesi veri: che il backbone si
costruisca, in che FORMA escono le sue feature, e che le tre mappe rispettino
il contratto del mask decoder di SAM2.

    python prova_encoder.py --encoder prithvi    --prithvi-size 512
    python prova_encoder.py --encoder terracodec --tec-variante fp
    python prova_encoder.py --encoder unet
"""

import argparse

import torch

from LENTE_utils import load_config, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["unet", "prithvi", "terracodec"],
                    required=True)
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("--target-size", type=int, default=1024)
    ap.add_argument("--adapter-dim", type=int, default=0)
    ap.add_argument("--prithvi-size", type=int, default=512)
    ap.add_argument("--prithvi-model", type=str, default=None)
    ap.add_argument("--prithvi-adapter-dim", type=int, default=0)
    ap.add_argument("--prithvi-lora", type=int, default=0)
    ap.add_argument("--tec-variante", type=str, default="fp")
    ap.add_argument("--tec-compression", type=int, default=10)
    ap.add_argument("--tec-size", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config, dataset=a.dataset, prompt_source="none")
    cfg["encoder"] = a.encoder
    cfg["target_size"] = a.target_size
    cfg["adapter_dim"] = a.adapter_dim
    cfg["prithvi_size"] = a.prithvi_size
    cfg["prithvi_adapter_dim"] = a.prithvi_adapter_dim
    cfg["prithvi_lora"] = a.prithvi_lora
    cfg["tec_variante"] = a.tec_variante
    cfg["tec_compression"] = a.tec_compression
    cfg["tec_size"] = a.tec_size
    if a.prithvi_model:
        cfg["prithvi_model"] = a.prithvi_model
    dev = cfg["device"]

    print(f"\n--- 1. costruzione ({a.encoder}) -----------------------------")
    model = build_model(cfg, dev)
    enc = model.unet_encoder
    if enc is None:
        raise SystemExit("nessun encoder sostitutivo costruito")

    print("\n--- 2. le tre mappe per il decoder di SAM2 -------------------")
    s = cfg["target_size"]
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

    print("\n--- 3. i tre livelli portano contenuto DIVERSO? --------------")
    # Se i livelli superficiali sono la mappa profonda ingrandita — il difetto
    # noto dello schema A su Prithvi — le loro versioni ridotte alla stessa
    # griglia sono quasi identiche. Con una piramide VERA no. E' la differenza
    # che separa i due trapianti, e qui si vede come numero.
    import torch.nn.functional as F
    g = [F.adaptive_avg_pool2d(m.float().mean(1, keepdim=True), 16).flatten()
         for m in mappe]
    for i, j in ((0, 1), (1, 2), (0, 2)):
        c = torch.nn.functional.cosine_similarity(g[i], g[j], dim=0).item()
        print(f"  somiglianza fra livello {i} e {j}: {c:+.3f}")
    print("  vicino a +1.000 su tutte le coppie = un livello solo travestito "
          "da tre")

    print("\n--- 4. una forward completa ----------------------------------")
    n_ch = cfg["n_optical"] + len(cfg.get("terrain_bands", []))
    with torch.no_grad():
        with torch.amp.autocast(device_type=dev):
            pred, punteggi = model(torch.randn(1, n_ch, s, s, device=dev), None)
    print(f"  maschere {tuple(pred.shape)}")

    n_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"\n  addestrabili in tutto: {n_tr:,}")
    print("\n  " + ("TUTTO A POSTO: puoi lanciare l'addestramento."
                    if tutto else
                    "LE FORME NON COMBACIANO: non lanciare."))
    print()


if __name__ == "__main__":
    main()

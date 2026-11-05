#!/usr/bin/env python
"""
rete_nativa.py — una U-Net che decide alla risoluzione NATIVA di L4S (128 px).

PERCHE'. La sonda `cella` ha misurato che LENTE perde le frane piccole per
come e' fatto SAM2: decide sulla griglia a stride 16, dove a 512 di ingresso
una cella copre 4x4 pixel nativi e una frana sotto i 16 px e' minoranza nella
sua cella (la cella piu' piena e' frana solo al 28%). Le frane in una sola
cella vengono prese nel 55% dei casi, quelle su cinque o piu' celle nel 95%.

Una U-Net non ha quel collo di bottiglia: il suo decoder riporta la semantica
profonda a OGNI livello fino alla risoluzione piena (e' un FPN completo per
costruzione), e decide pixel per pixel sulla griglia nativa.

COSA NON CI SI ASPETTA. Che batta LENTE da sola. Gia' misurato (HTML,
"il limite e' il backbone"): una U-Net da zero al posto di Hiera pareggia
sulle patch con frana e perde tutto sulle vuote, perche' 2.280 patch non
bastano a insegnare quando NON c'e' niente. Qui la U-Net serve solo come
SECONDO RAMO: LENTE decide se nella patch c'e' una frana, la U-Net dove sono
quelle piccole. La fusione e' in `LENTE_sonde.py fusione`.

PROTOCOLLO, identico a LENTE_train per tutto cio' che non e' il modello:
stesso dataset e split, stesse copie offline dei positivi, stesse
trasformazioni, stessa loss con lo stesso pos_weight, checkpoint scelto sul
val. Si seleziona sul MICRO: la selezione per immagine premia i modelli
prudenti, e a questo ramo serve il contrario.

USO (da LENTE_SAM_V2, con LENTE_DATASET_ROOT impostata)

    python rete_nativa.py --epochs 100
"""

import argparse
import json
import os
import time

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from LENTE_utils import (load_config, build_dataset, build_train_val_test_subsets,
                         TransformSubset, criterion, configure_criterion)


def costruisci(n_canali, encoder="resnet18", pesi=None):
    """La rete: U-Net di smp, uscita a un canale alla stessa risoluzione dell'ingresso."""
    return smp.Unet(encoder_name=encoder, encoder_weights=pesi,
                    in_channels=n_canali, classes=1)


@torch.no_grad()
def _valida(model, loader, dev, pos_weight):
    model.eval()
    tp = fp = fn = 0.0
    perdita, n = 0.0, 0
    for b in loader:
        x, m = b["image"].to(dev), b["mask"].to(dev)
        with torch.autocast(device_type="cuda", enabled=dev.startswith("cuda")):
            lg = model(x)
            perdita += float(criterion(lg, m, pos_weight=pos_weight))
        p = torch.sigmoid(lg.float()) > 0.5
        g = m.unsqueeze(1) > 0
        tp += float((p & g).sum()); fp += float((p & ~g).sum()); fn += float((~p & g).sum())
        n += 1
    return tp / max(tp + fp + fn, 1.0), perdita / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-d", "--dataset", default="l4s", choices=["l4s", "bijie"])
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--pesi", default=None, choices=[None, "imagenet"],
                    help="pesi preaddestrati dell'encoder (default: nessuno)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limite", type=int, default=0, help="solo per prove: N patch di train")
    ap.add_argument("--checkpoint", default=None,
                    help="default: best_rete_nativa_<dataset>.pth")
    a = ap.parse_args()

    a.checkpoint = a.checkpoint or f"best_rete_nativa_{a.dataset}.pth"
    cfg = load_config(a.config, dataset=a.dataset, prompt_source="none")
    dev = cfg["device"]
    configure_criterion(cfg)
    full = build_dataset(cfg)
    tr, va, _, pos_weight = build_train_val_test_subsets(full, cfg)

    # le stesse copie offline dei positivi del train che usa LENTE_train
    aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    pos = set(full.positive_indices)
    tr_pos = [i for i in tr.indices if i in pos]
    full.add_augmented_samples(indices=tr_pos, transform=aug,
                               n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    copie = list(range(len(full.data) - len(tr_pos) * cfg["aug_copies"], len(full.data)))
    idx_tr = list(tr.indices) + copie
    if a.limite:
        idx_tr, va = idx_tr[:a.limite], Subset(full, list(va.indices)[:max(4, a.limite // 4)])

    t_tr = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.RandomRotate90(p=0.5), ToTensorV2()])
    t_va = A.Compose([ToTensorV2()])
    ld_tr = DataLoader(TransformSubset(Subset(full, idx_tr), t_tr), batch_size=a.batch_size,
                       shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    ld_va = DataLoader(TransformSubset(va, t_va), batch_size=a.batch_size,
                       shuffle=False, num_workers=a.workers, pin_memory=True)

    n_canali = int(full[0]["image"].shape[0])
    model = costruisci(n_canali, a.encoder, a.pesi).to(dev)
    print(f"  U-Net {a.encoder} | {n_canali} canali a 128 px nativi | "
          f"{sum(p.numel() for p in model.parameters()):,} parametri | "
          f"{a.dataset} | train {len(idx_tr)} (con le copie) | val {len(va)} | pos_weight {pos_weight:.2f}")
    ottim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(ottim, mode="max", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(enabled=dev.startswith("cuda"))

    storia = {"checkpoint": a.checkpoint, "inizio": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": vars(a), "epoche": []}
    migliore, attesa = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        somma = 0.0
        for b in tqdm(ld_tr, desc=f"Ep {ep}/{a.epochs}", leave=False):
            x, m = b["image"].to(dev), b["mask"].to(dev)
            with torch.autocast(device_type="cuda", enabled=dev.startswith("cuda")):
                loss = criterion(model(x), m, pos_weight=pos_weight)
            ottim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(ottim)
            scaler.update()
            somma += float(loss)
        micro, vloss = _valida(model, ld_va, dev, pos_weight)
        sched.step(micro)
        riga = {"epoca": ep, "train_loss": somma / max(len(ld_tr), 1), "val_loss": vloss,
                "val_micro_iou": micro, "lr": ottim.param_groups[0]["lr"]}
        print(f"EPOCA {ep} | train {riga['train_loss']:.4f} | val loss {vloss:.4f} | micro {micro:.4f}")
        if micro > migliore:
            migliore, attesa = micro, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "micro": micro,
                        "encoder": a.encoder, "n_canali": n_canali}, a.checkpoint)
            riga["record"] = True
            print(f"  Record: micro {micro:.4f} — checkpoint salvato in {a.checkpoint}")
        else:
            attesa += 1
        storia["epoche"].append(riga)
        cart, base = os.path.split(os.path.splitext(a.checkpoint)[0])
        with open(os.path.join(cart, f"history_{base}.json"), "w") as f:
            json.dump(storia, f, indent=1)
        if attesa >= a.patience:
            print(f"\nEARLY STOPPING — miglior micro in validazione {migliore:.4f}")
            break


if __name__ == "__main__":
    main()

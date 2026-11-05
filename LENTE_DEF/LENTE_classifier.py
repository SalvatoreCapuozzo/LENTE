"""
LENTE_classifier.py — c'e' una frana in questa patch, si' o no?
================================================================

PERCHE' ESISTE. La IoU macro per campione e' la media su 759 patch di cui **306
sono VUOTE**, e su una patch vuota il punteggio e' tutto-o-niente: 1.0 se il
modello non disegna niente, ~0.0 se accende anche un solo pixel. Il 40% della
metrica si decide quindi su una domanda BINARIA, non sulla qualita' dei contorni.

MISURATO sul modello prompt-free a 512:

    vuote 0.7843 | con frana 0.4468 | totale 0.5829
    tetto con un classificatore PERFETTO ................. 0.6699  (+0.087)

Nessuna leva sul prompt ha mai reso piu' di +0.030. Questa e' l'unica cosa nel
progetto con un ordine di grandezza diverso.

COSA FA. Addestra un classificatore binario sulle STESSE patch e con lo STESSO
split di LENTE (60/20/20, seed 42 — se lo split divergesse, il gate sarebbe
valutato su patch viste in addestramento), e poi salva un punteggio per ogni
campione di val e test. Quel file lo legge LENTE_diagnose.py e lo usa come gate:
dove il classificatore dice "vuota", la maschera viene azzerata.

PERCHE' NON RESNET101. Su 2280 immagini di addestramento e' sovradimensionata.
Il default e' resnet34; resnet50 e' li' se serve. Nota che come CLASSIFICATORE
la rete e' interamente addestrabile — non ha nessuno dei vincoli che bloccano
LENTE, dove l'encoder e' congelato e 9 bande su 12 partono da rumore.

LA SOGLIA DEL GATE NON SI SCEGLIE QUI. Va scelta sul validation dentro
LENTE_diagnose.py, insieme alle altre, e riportata sul test una volta sola.
Qui si salvano i punteggi grezzi, non le decisioni.

Uso:
    python LENTE_classifier.py train           # addestra e salva il checkpoint
    python LENTE_classifier.py scores          # scrive i punteggi di val e test
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torchvision

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    run_record_path, write_json,
)

CKPT_DEFAULT = "best_clf_l4s.pth"
SCORES_DEFAULT = "clf_scores_l4s.npz"


# ------------------------------------------------------------------
# modello
# ------------------------------------------------------------------

def build_classifier(n_channels, arch="resnet34", pretrained=True):
    """
    ResNet con lo stem allargato a n_channels, come fa LENTE col patch embed di
    SAM2: i primi 3 canali ereditano i pesi ImageNet, gli altri partono da
    rumore piccolo.

    Differenza cruciale rispetto a LENTE: qui **tutta la rete e' addestrabile**,
    quindi i canali nuovi non devono farsi capire da un encoder congelato. E'
    il motivo per cui questo compito e' molto piu' facile della segmentazione.
    """
    net = getattr(torchvision.models, arch)(
        weights="DEFAULT" if pretrained else None)
    old = net.conv1
    new = nn.Conv2d(n_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        if n_channels > 3:
            nn.init.normal_(new.weight[:, 3:], mean=0.0, std=0.001)
    net.conv1 = new
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def _labels(subset, full_ds):
    """1 se la patch contiene almeno un pixel di frana."""
    pos = set(full_ds.positive_indices)
    return torch.tensor([1.0 if i in pos else 0.0 for i in subset.indices])


# ------------------------------------------------------------------

class ClfPresenceProvider:
    """
    Classificatore CONGELATO che fornisce a SAM il segnale di presenza durante
    l'addestramento, esattamente come YoloBoxProvider fornisce i box.

    PERCHE' DENTRO IL LOOP e non precalcolato: le immagini che LENTE vede sono
    gia' aumentate (flip, rotazioni), e il punteggio deve riferirsi a QUELLE.
    Precalcolandolo sulle originali si darebbe al modello un segnale scollegato
    dal campione — lo stesso difetto che affliggeva i box in `TransformSubset`.
    (Per la classificazione binaria l'augmentation non dovrebbe cambiare la
    risposta, ma farlo nel loop costa poco e toglie l'ipotesi di mezzo.)

    PERCHE' CONGELATO: e' gia' addestrato sul suo compito, e coaddestrarlo con
    la segmentazione ha un precedente negativo — il classificatore pesato sulla
    IoU del segmentatore e' andato PEGGIO di quello semplice (0.6555 vs 0.6592).
    """

    def __init__(self, ckpt, device, size=None):
        ck = torch.load(ckpt, map_location=device)
        self.net = build_classifier(ck["n_channels"], ck["arch"], pretrained=False).to(device)
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.size = ck["size"] if size is None else size
        self.device = device
        self.balanced_acc = ck.get("balanced_acc")

    @torch.no_grad()
    def p_for(self, imgs):
        """imgs: (B, C, H, W) alla risoluzione nativa. Ritorna (B,) in [0, 1]."""
        x = imgs.to(self.device)
        x = F.interpolate(x, (self.size, self.size), mode="bilinear", align_corners=False)
        with torch.amp.autocast(device_type=self.device):
            return torch.sigmoid(self.net(x).squeeze(1)).float()


class WeightedSubset(torch.utils.data.Dataset):
    """
    Attacca un peso a ogni campione, dentro il dizionario che il DataLoader
    restituisce.

    NON si puo' tenere i pesi in un array esterno e indicizzarli con la
    posizione nel batch: con `shuffle=True` il DataLoader mescola, e il peso
    finirebbe sul campione sbagliato — un errore che non da' nessun errore.
    """

    def __init__(self, base, weights):
        self.base, self.w = base, weights

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = dict(self.base[i])
        s["w"] = float(self.w[i])
        return s


def _run(net, loader, dev, size, opt=None, pos_weight=None):
    """
    Una passata. opt=None -> valutazione. Ritorna (loss media, logit, label).

    Se il batch porta la chiave "w", quello e' il PESO PER CAMPIONE. Cambia la
    domanda a cui il classificatore risponde:

      senza pesi ->  "c'e' una frana in questa patch?"
      con pesi   ->  "conviene AZZERARE questa patch?"

    e sono due cose diverse. Su una patch con 30 pixel di frana la risposta alla
    prima e' "si' c'e'", ma il segmentatore su quella frana prende ~0.05, quindi
    azzerarla costa quasi niente: la risposta alla seconda e' "azzera pure".
    Il peso e' proprio quanto e' in gioco (vedi comando `weights`), cosi' il
    classificatore non viene punito dove sbagliare e' gratis, ed e' punito
    moltissimo dove una frana grande verrebbe cancellata.
    """
    train = opt is not None
    net.train(train)
    tot, n = 0.0, 0
    L, Y = [], []
    for b in tqdm(loader, desc="train" if train else "eval", leave=False):
        x = b["image"].to(dev)
        y = (b["mask"].flatten(1).sum(1) > 0).float().to(dev)
        x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
        w = b["w"].float().to(dev) if "w" in b else None
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=dev):
                logit = net(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(
                    logit, y, weight=w, pos_weight=pos_weight)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss) * len(y); n += len(y)
        L.append(logit.detach().float().cpu()); Y.append(y.cpu())
    return tot / max(n, 1), torch.cat(L), torch.cat(Y)


def _report(logit, y):
    """
    Accuratezza bilanciata: la media fra "quante vuote riconosce" e "quante
    frane riconosce". Con classi sbilanciate l'accuratezza semplice premia chi
    risponde sempre la classe maggioritaria, e qui sarebbe fuorviante.
    """
    p = (torch.sigmoid(logit) > 0.5).float()
    rec_pos = float((p[y == 1] == 1).float().mean()) if (y == 1).any() else 0.0
    rec_neg = float((p[y == 0] == 0).float().mean()) if (y == 0).any() else 0.0
    return 0.5 * (rec_pos + rec_neg), rec_pos, rec_neg


@torch.no_grad()
def weights(cfg, args):
    """
    Peso per campione = QUANTO E' IN GIOCO nella decisione di azzerare.

        patch VUOTA      w = 1 - IoU   (quanto GUADAGNI azzerandola)
        patch CON FRANA  w = IoU       (quanto PERDI azzerandola)

    Con questi pesi il classificatore smette di ottimizzare "c'e' una frana?" e
    ottimizza "conviene azzerare?". Una patch con una frana minuscola, che il
    segmentatore sbaglia comunque, ha peso ~0.05: il classificatore e' libero di
    azzerarla senza quasi penalita' — ed e' esattamente li' che oggi perde
    l'8.6% dei positivi. Una frana grande e ben segmentata ha peso 0.7 e diventa
    intoccabile.

    NOTA sul bias: sul TRAIN il segmentatore e' ottimista, ci si e' addestrato,
    quindi i pesi dei positivi sono un po' gonfiati. Serve solo l'ordine di
    grandezza relativo, quindi si accetta — ma va saputo.
    """
    from LENTE_utils import build_model, get_best_mask, as_refine_map
    dev = cfg["device"]
    ck = torch.load(args.lente, map_location="cpu")
    sd = ck["model_state_dict"]
    ad = [v.shape[0] for k, v in sd.items() if k.endswith("adapters.0.adapter.0.weight")]
    if ad:
        cfg["adapter_dim"] = ad[0]
    seg = build_model(cfg, dev)
    seg.load_state_dict(sd, strict=False); seg.eval()

    prov = None
    if cfg["prompt_source"] == "yolo_box":
        from LENTE_yolo_cascade import YoloBoxProvider
        prov = YoloBoxProvider(cfg["yolo_weights"], cfg)
    dense_mode = prov is not None and cfg.get("yolo_prompt") == "dense"
    rp = int(cfg.get("refine_passes", 0))
    T = cfg["target_size"]
    print(f"pesi da {args.lente} | {cfg['prompt_source']}@{T} | dense={dense_mode} "
          f"| refine={rp}")

    full = build_dataset(cfg)
    tr, va, _, _ = build_train_val_test_subsets(full, cfg)
    out = {}
    for nome, sub in (("train", tr), ("val", va)):
        ld = DataLoader(TransformSubset(sub, A.Compose([ToTensorV2()])),
                        batch_size=8)
        I, E = [], []
        for b in tqdm(ld, desc=nome, leave=False):
            imgs, masks = b["image"].to(dev), b["mask"].to(dev)
            d = None
            if dense_mode:
                d, _ = prov.dense_for(b["image"], seg.mask_input_size[0]); d = d.to(dev)
            im = F.interpolate(imgs, (T, T), mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type=dev):
                o, s = seg(im, None, dense=d)
                lg = get_best_mask(o, s)
                for _ in range(rp):
                    lg = get_best_mask(*seg(im, None, dense=d,
                                            refine=as_refine_map(lg, seg.mask_input_size[0])))
            pr = F.interpolate(lg.float(), masks.shape[-2:],
                               mode="bilinear", align_corners=False)
            pr = (torch.sigmoid(pr) > 0.5).float(); gt = masks.unsqueeze(1).float()
            tp = (pr*gt).sum((2, 3)); fp = (pr*(1-gt)).sum((2, 3)); fn = ((1-pr)*gt).sum((2, 3))
            I.append(((tp + 1e-7) / (tp + fp + fn + 1e-7)).squeeze(1).cpu())
            E.append((gt.flatten(1).sum(1) == 0).cpu())
        iou = torch.cat(I).numpy(); emp = torch.cat(E).numpy()
        w = np.where(emp, 1.0 - iou, iou).astype(np.float32)
        out[f"{nome}_w"] = w
        print(f"  {nome}: peso medio vuote {w[emp].mean():.3f} | "
              f"con frana {w[~emp].mean():.3f} | "
              f"frane con peso <0.1: {(w[~emp] < 0.1).mean():.1%}")
    np.savez(args.wout, **out)
    print(f"\nScritto {args.wout} — usalo con:  train --weights {args.wout}")


def train(cfg, args):
    dev = cfg["device"]
    size = args.size
    full = build_dataset(cfg)
    tr, va, _, _ = build_train_val_test_subsets(full, cfg)

    aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                     A.RandomRotate90(p=0.5), ToTensorV2()])
    tr_ds = TransformSubset(tr, aug)
    va_ds = TransformSubset(va, A.Compose([ToTensorV2()]))
    if args.weights:
        z = np.load(args.weights)
        tr_ds = WeightedSubset(tr_ds, z["train_w"])
        va_ds = WeightedSubset(va_ds, z["val_w"])
        print(f"pesi per campione da {args.weights}: il bersaglio non e' piu' "
              f"'c'e' una frana?' ma 'conviene azzerare?'")
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch, pin_memory=True)

    y_tr = _labels(tr, full)
    pw = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=dev)
    print(f"train {len(tr_ds)} ({int((y_tr==1).sum())} con frana) | val {len(va_ds)}"
          f" | pos_weight {float(pw):.3f} | {args.arch}@{size}")

    net = build_classifier(cfg["n_optical"] + 2, args.arch).to(dev)
    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = run_record_path(args.ckpt, "history")
    record = {"checkpoint": args.ckpt, "inizio": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": {"arch": args.arch, "size": size, "lr": args.lr,
                         "batch": args.batch, "epochs": args.epochs,
                         "n_channels": cfg["n_optical"] + 2,
                         "random_seed": cfg["random_seed"]},
              "epoche": []}
    print(f"  record del run: {hist}")

    best, bad = -1.0, 0
    for ep in range(args.epochs):
        tl, _, _ = _run(net, tr_ld, dev, size, opt, pw)
        vl, logit, y = _run(net, va_ld, dev, size, None, pw)
        sch.step()
        bal, rp, rn = _report(logit, y)
        print(f"EPOCA {ep+1:3d} | train {tl:.4f} | val {vl:.4f} | "
              f"bilanciata {bal:.4f}  (frane {rp:.3f} / vuote {rn:.3f})")
        riga = {"epoca": ep + 1, "train_loss": tl, "val_loss": vl,
                "bilanciata": bal, "recall_frane": rp, "recall_vuote": rn,
                "lr": opt.param_groups[0]["lr"]}
        if bal > best:
            best, bad = bal, 0
            torch.save({"model_state_dict": net.state_dict(), "arch": args.arch,
                        "size": size, "n_channels": cfg["n_optical"] + 2,
                        "balanced_acc": bal}, args.ckpt)
            print(f"  record {bal:.4f} — salvato in {args.ckpt}")
            riga["record"] = True
        else:
            bad += 1
        record["epoche"].append(riga); record["best_balanced_acc"] = best
        write_json(hist, record)
        if bad >= args.patience:
            print(f"\nEARLY STOPPING — migliore accuratezza bilanciata: {best:.4f}")
            break


@torch.no_grad()
def scores(cfg, args):
    """
    Punteggi grezzi (probabilita') per ogni campione di val e test, nell'ordine
    del rispettivo Subset. Nessuna soglia applicata qui: la sceglie
    LENTE_diagnose.py sul validation.
    """
    dev = cfg["device"]
    ck = torch.load(args.ckpt, map_location=dev)
    net = build_classifier(ck["n_channels"], ck["arch"], pretrained=False).to(dev)
    net.load_state_dict(ck["model_state_dict"]); net.eval()
    size = ck["size"]
    print(f"{args.ckpt} | {ck['arch']}@{size} | accuratezza bilanciata val "
          f"{ck['balanced_acc']:.4f}")

    full = build_dataset(cfg)
    _, va, te, _ = build_train_val_test_subsets(full, cfg)
    out = {}
    for nome, sub in (("val", va), ("test", te)):
        ld = DataLoader(TransformSubset(sub, A.Compose([ToTensorV2()])),
                        batch_size=args.batch)
        P, Y = [], []
        for b in tqdm(ld, desc=nome, leave=False):
            x = F.interpolate(b["image"].to(dev), (size, size),
                              mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type=dev):
                P.append(torch.sigmoid(net(x).squeeze(1)).float().cpu())
            Y.append((b["mask"].flatten(1).sum(1) > 0).float())
        p, y = torch.cat(P).numpy(), torch.cat(Y).numpy()
        out[f"{nome}_score"] = p
        out[f"{nome}_label"] = y
        bal, rp, rn = _report(torch.tensor(p).logit().nan_to_num(), torch.tensor(y))
        print(f"  {nome}: {len(p)} campioni | bilanciata {bal:.4f} "
              f"(frane {rp:.3f} / vuote {rn:.3f})")
    np.savez(args.out, **out)
    print(f"\nScritto {args.out} — usalo come gate in LENTE_diagnose.py")


def main():
    ap = argparse.ArgumentParser(description="Classificatore binario frana / non frana")
    ap.add_argument("cmd", choices=["train", "scores", "weights"])
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--out", default=SCORES_DEFAULT)
    # pesi per campione (comando `weights`, poi `train --weights`)
    ap.add_argument("--lente", default="best_lente_l4s_refine.pth",
                    help="checkpoint del segmentatore da cui misurare la IoU per campione")
    ap.add_argument("--wout", default="clf_weights_l4s.npz")
    ap.add_argument("--weights", default=None,
                    help="file di pesi prodotto dal comando `weights`")
    a = ap.parse_args()
    cfg = load_config(a.config, dataset=a.dataset)
    {"train": train, "scores": scores, "weights": weights}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()

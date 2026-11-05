#!/usr/bin/env python
"""
LENTE_multi_v2.py — un solo LENTE per satellite (L4S, 10 m, 12 bande) e
altissima risoluzione (Bijie, 0,8 m, RGB). Versione 3 dell'ingresso.

I DUE DATI DIFFERISCONO IN TRE COSE, e l'ingresso le tratta una per una.

1. SPETTRO. Stem RGB CONDIVISO + ramo multispettrale aggiuntivo.
   Rosso, verde e blu passano per LO STESSO patch embed preaddestrato in
   entrambi i domini: Bijie e L4S allenano gli stessi pesi per il visibile, e
   la conoscenza si condivide davvero. Per L4S il visibile e' B4, B3, B2
   (`rgb_channels` del config), non i primi tre canali: il LENTE a dominio
   singolo copia i pesi RGB su B1, B2, B3, cioe' il "rosso" del
   preaddestramento finisce sulla banda costiera. Le altre 9 bande di L4S
   entrano da un ramo separato, SOMMATO, che parte da zero: all'inizio L4S
   viene visto come un'immagine RGB, e il resto lo impara.

2. SCALA. FiLM sulla risoluzione a terra (GSD). Una frana e' 16 pixel su L4S e
   migliaia su Bijie, e il decoder e' uno solo. Ogni mappa FPN riceve una
   modulazione per canale, scala e spostamento, calcolata da log10(GSD):
   10 m -> 1, 0,8 m -> -0,1. Lineare e inizializzata a zero, quindi parte
   dall'identita'; essendo continua, un sensore a 3 m entra senza ripensare
   niente.

3. NORMALIZZAZIONE. z-score per dominio sull'ingresso (statistiche del train,
   salvate nel checkpoint) e BatchNorm PER DOMINIO in DSDF, stream del terreno
   e MGEM/DGFM. Con batch alternati e BatchNorm condivise, le statistiche
   correnti diventano la media dei due domini, e a inferenza ciascuno viene
   normalizzato con statistiche sbagliate. Convoluzioni condivise, BatchNorm
   separate. Hiera e il decoder di SAM2 usano LayerNorm e non ne soffrono.

LE DUE LOSS. Stessa forma (BCE con pos_weight + 0.3 Dice per immagine), con il
pos_weight del proprio dominio. Ogni passo dell'ottimizzatore vede un batch per
dominio (accumulo 2), campionamento 50/50. I pesi si IMPARANO con la
ponderazione per incertezza (Kendall et al., 2018):
    L = sum_d  exp(-s_d) * L_d + s_d
due parametri, che bilanciano da soli un dominio difficile (L4S) e uno facile
(Bijie). Il checkpoint si sceglie su `--criterio` (default: media dei micro).

Tutto il resto e' il LENTE di oggi (`LENTEMultiDominio` eredita da `LENTE`):
fusione, rilettura, FPN completo, H-SAM. Split e pos_weight sono quelli dei
run singoli, quindi i test sono gli stessi e il confronto e' diretto.

USO (da LENTE_SAM_V2, con LENTE_DATASET_ROOT e LENTE_BIJIE_ROOT impostate)

    python LENTE_multi_v2.py --unfreeze-blocks 2 --fusione entrambi --fusione-aux terreno

Test, un dominio per volta, con la sonda:

    python LENTE_sonde.py soglia -d bijie --multi --ckpt <ckpt> --target-size 512 ...
"""

import argparse
import copy
import itertools
import json
import math
import os
import time

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from LENTE_utils import (LENTE, load_config, build_dataset, build_train_val_test_subsets,
                         TransformSubset, criterion, configure_criterion, get_best_mask,
                         as_refine_map, _norm_da_cfg)

DOMINI = ("l4s", "bijie")
GSD = {"l4s": 10.0, "bijie": 0.8}            # metri per pixel nativo


# ═══════════════════════════════════════════════════════════════════════════
#  Pezzi dell'ingresso
# ═══════════════════════════════════════════════════════════════════════════

class StemSpettrale(nn.Module):
    """Patch embed RGB condiviso + un ramo per le bande in piu' di ciascun dominio."""

    def __init__(self, conv_rgb, indici, n_extra):
        super().__init__()
        self.rgb = conv_rgb                                     # pesi RGB preaddestrati
        self.extra = nn.ModuleDict()
        for d, n in n_extra.items():
            if n > 0:
                c = nn.Conv2d(n, conv_rgb.out_channels, kernel_size=conv_rgb.kernel_size,
                              stride=conv_rgb.stride, padding=conv_rgb.padding, bias=False)
                nn.init.zeros_(c.weight)                        # parte dall'identita'
                self.extra[d] = c
        self.indici = indici                                    # d -> (rgb, extra)
        self.attivo = DOMINI[0]

    # Sam2PatchEmbeddings legge `projection.weight` (per il dtype) prima di
    # chiamarla: lo stem deve presentarsi come una Conv2d.
    @property
    def weight(self):
        return self.rgb.weight

    @property
    def bias(self):
        return self.rgb.bias

    def forward(self, x):
        rgb, extra = self.indici[self.attivo]
        y = self.rgb(x[:, rgb])
        if extra:
            y = y + self.extra[self.attivo](x[:, extra])
        return y


class BNDominio(nn.Module):
    """BatchNorm separate per dominio dietro un'unica interfaccia."""

    def __init__(self, bn, domini, stato):
        super().__init__()
        self.bn = nn.ModuleDict({d: copy.deepcopy(bn) for d in domini})
        self.stato = stato                                      # dict condiviso, non un modulo

    def forward(self, x):
        return self.bn[self.stato["d"]](x)


def _bn_per_dominio(modulo, domini, stato):
    n = 0
    for nome, figlio in modulo.named_children():
        if isinstance(figlio, nn.BatchNorm2d):
            setattr(modulo, nome, BNDominio(figlio, domini, stato))
            n += 1
        else:
            n += _bn_per_dominio(figlio, domini, stato)
    return n


class FiLMScala(nn.Module):
    """Scala e spostamento per canale delle mappe FPN, funzione di log10(GSD)."""

    def __init__(self, canali=(32, 64, 256)):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(1, 2 * c) for c in canali])
        for l in self.lin:
            nn.init.zeros_(l.weight)
            nn.init.zeros_(l.bias)

    def forward(self, emb, log_gsd):
        out = []
        for f, l in zip(emb, self.lin):
            gb = l(torch.full((1, 1), float(log_gsd), device=f.device, dtype=torch.float32))
            gam, bet = gb.chunk(2, dim=-1)
            out.append(f * (1 + gam.view(1, -1, 1, 1).to(f.dtype))
                       + bet.view(1, -1, 1, 1).to(f.dtype))
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Modello
# ═══════════════════════════════════════════════════════════════════════════

class LENTEMultiDominio(LENTE):

    def __init__(self, canali, rgb, gsd=None, **kw):
        super().__init__(n_optical=canali["l4s"], **kw)
        if (self.fusione == "none" and self.gate_s0 is None and self.hsam is None
                and self.fpn is None):
            raise SystemExit("serve il percorso con image_embeddings: usa --fusione "
                             "(o --fpn-completo / --hsam), altrimenti la FiLM non ha dove agire")
        self.canali, self.gsd = dict(canali), dict(gsd or GSD)

        # 1. stem: il patch embed a 12 canali di LENTE porta i pesi RGB originali
        #    sui primi tre canali; si riprendono quelli per lo stem condiviso
        c12 = self.patch_proj
        conv_rgb = nn.Conv2d(3, c12.out_channels, kernel_size=c12.kernel_size,
                             stride=c12.stride, padding=c12.padding, bias=c12.bias is not None)
        with torch.no_grad():
            conv_rgb.weight.copy_(c12.weight[:, :3])
            if c12.bias is not None:
                conv_rgb.bias.copy_(c12.bias)
        indici = {}
        for d in DOMINI:
            r = list(rgb[d])
            indici[d] = (r, [i for i in range(canali[d]) if i not in r])
        stem = StemSpettrale(conv_rgb, indici, {d: len(indici[d][1]) for d in DOMINI})
        venc = self.model.vision_encoder
        for nome, mod in venc.named_modules():
            if mod is c12:
                parti = nome.split(".")
                padre = venc.get_submodule(".".join(parti[:-1])) if len(parti) > 1 else venc
                setattr(padre, parti[-1], stem)
                break
        else:
            raise RuntimeError("patch embed non trovato nell'encoder")
        self.patch_proj = stem
        for p in stem.parameters():
            p.requires_grad = True

        # 3. BatchNorm per dominio nei moduli nuovi (Hiera e decoder: LayerNorm)
        self._stato = {"d": DOMINI[0]}
        n_bn = 0
        for nome in ("early_guidance", "aux_encoder", "mgem", "dgfm", "gate_s0", "gate_s1"):
            m = getattr(self, nome, None)
            if m is not None:
                n_bn += _bn_per_dominio(m, DOMINI, self._stato)
        # le liste di parametri create da LENTE puntano alle BatchNorm vecchie
        if self.fusione != "none":
            self.fusione_params = [q for m in (self.aux_encoder, self.mgem, self.dgfm)
                                   if m is not None for q in m.parameters()]

        # 2. FiLM sulla scala
        self.film = FiLMScala()
        self.film_params = list(self.film.parameters())

        for d, n in self.canali.items():                     # ottico + 2 di terreno
            self.register_buffer(f"media_{d}", torch.zeros(n + 2))
            self.register_buffer(f"dev_{d}", torch.ones(n + 2))
        print(f"  ingresso multi-dominio: stem RGB condiviso su {indici['l4s'][0]} (L4S) e "
              f"{indici['bijie'][0]} (Bijie) + {len(indici['l4s'][1])} bande extra a zero | "
              f"{n_bn} BatchNorm per dominio | FiLM su GSD {self.gsd}")
        self.imposta_dominio(DOMINI[0])

    def imposta_dominio(self, d):
        self.dominio = d
        self.patch_proj.attivo = d
        self._stato["d"] = d
        self.n_optical = self.canali[d]

    def _filtra_skip(self, emb):
        emb = super()._filtra_skip(emb)
        return self.film(emb, math.log10(self.gsd[self.dominio]))

    def forward(self, x, *args, **kw):
        m = getattr(self, f"media_{self.dominio}").view(1, -1, 1, 1)
        s = getattr(self, f"dev_{self.dominio}").view(1, -1, 1, 1)
        return super().forward((x - m.to(x.dtype)) / s.to(x.dtype), *args, **kw)


def costruisci_multi(cfg, dev, config_path="config.yaml"):
    """Come build_model, ma multi-dominio. `cfg` porta l'architettura condivisa."""
    per_d = {d: load_config(config_path, dataset=d, prompt_source="none") for d in DOMINI}
    canali = {d: int(per_d[d]["n_optical"]) for d in DOMINI}
    rgb = {d: per_d[d].get("rgb_channels", [0, 1, 2]) for d in DOMINI}
    gsd = {d: float(per_d[d].get("gsd", GSD[d])) for d in DOMINI}
    m = LENTEMultiDominio(
        canali=canali, rgb=rgb, gsd=gsd,
        sam_ckpt_path=cfg["sam_ckpt_path"], adapter_dim=cfg["adapter_dim"],
        prompt_source=cfg.get("prompt_source", "none"), image_size=cfg["target_size"],
        unfreeze_prompt_encoder=cfg.get("unfreeze_prompt_encoder", False),
        lora_rank=int(cfg.get("lora_rank", 0)), lora_alpha=float(cfg.get("lora_alpha", 16.0)),
        adapter_where=cfg.get("adapter_where", "attn"),
        unfreeze_blocks=int(cfg.get("unfreeze_blocks", 0)),
        gsrm=bool(cfg.get("gsrm", False)),
        fusione=cfg.get("fusione", "none"), fusione_width=int(cfg.get("fusione_width", 24)),
        fusione_aux=cfg.get("fusione_aux", "terreno"),
        norm_ingresso=_norm_da_cfg(cfg),
        hsam=bool(cfg.get("hsam", False)),
        fpn_completo=bool(cfg.get("fpn_completo", False))).to(dev)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  LENTE multi-dominio {canali} | addestrabili {n_tr:,}")
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  Dati
# ═══════════════════════════════════════════════════════════════════════════

def _statistiche(full, indici, max_campioni=500, seme=0):
    """Media e deviazione per canale sul TRAIN di un dominio."""
    rng = np.random.default_rng(seme)
    scelti = rng.choice(indici, min(max_campioni, len(indici)), replace=False)
    somma = somma2 = None
    n = 0
    for i in scelti:
        x = full[int(i)]["image"].double().flatten(1)          # (C, H*W)
        somma = x.sum(1) if somma is None else somma + x.sum(1)
        somma2 = (x ** 2).sum(1) if somma2 is None else somma2 + (x ** 2).sum(1)
        n += x.shape[1]
    media = somma / n
    dev = (somma2 / n - media ** 2).clamp_min(1e-12).sqrt()
    return media.float(), dev.float()


def _ciclo(loader):
    """
    Il DataLoader all'infinito, rimescolato e ri-aumentato a ogni giro.

    NON itertools.cycle: quello conserva in memoria una copia di ogni batch che
    restituisce, per poterli ripetere. Su un'epoca sono gigabyte, e il run del
    2026-10-16 e' morto all'epoca 21 per RAM esaurita. In piu' ripeteva i batch
    del dominio piu' piccolo identici, stesse augmentation e stesso ordine.
    """
    while True:
        for b in loader:
            yield b


def _dominio(cfg, batch_size, workers):
    full = build_dataset(cfg)
    tr, va, te, pos_weight = build_train_val_test_subsets(full, cfg)
    media, dev = _statistiche(full, list(tr.indices), seme=cfg["random_seed"])
    aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    pos = set(full.positive_indices)
    tr_pos = [i for i in tr.indices if i in pos]
    full.add_augmented_samples(indices=tr_pos, transform=aug,
                               n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    copie = list(range(len(full.data) - len(tr_pos) * cfg["aug_copies"], len(full.data)))
    t_tr = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.RandomRotate90(p=0.5), ToTensorV2()])
    t_va = A.Compose([ToTensorV2()])
    ld_tr = DataLoader(TransformSubset(Subset(full, list(tr.indices) + copie), t_tr),
                       batch_size=batch_size, shuffle=True, drop_last=True,
                       num_workers=workers, pin_memory=True)
    ld_va = DataLoader(TransformSubset(va, t_va), batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True)
    return {"ld_tr": ld_tr, "ld_va": ld_va, "pos_weight": pos_weight,
            "media": media, "dev": dev, "n_tr": len(tr.indices) + len(copie)}


# ═══════════════════════════════════════════════════════════════════════════
#  Addestramento
# ═══════════════════════════════════════════════════════════════════════════

def _gruppi(model, cfg, extra_veloci=()):
    """Gli stessi gruppi di learning rate di LENTE_train, piu' FiLM e pesi delle loss."""
    lr = float(cfg["learning_rate"])
    veloci = [model.dense_log_scale, model.refine_log_scale, model.clf_log_scale, *extra_veloci]
    gruppi, usati = [], {id(p) for p in veloci}
    gruppi.append({"params": veloci, "lr": 1e-3})
    for attr, chiave in (("fusione_params", "fusione_lr"), ("hsam_params", "hsam_lr"),
                         ("fpn_params", "fpn_lr"), ("gsrm_params", "gsrm_lr"),
                         ("film_params", "film_lr")):
        ps = [q for q in getattr(model, attr, []) if q.requires_grad and id(q) not in usati]
        if ps:
            gruppi.append({"params": ps, "lr": float(cfg.get(chiave, 3e-4))})
            usati |= {id(q) for q in ps}
    enc = [q for q in getattr(model, "encoder_params", []) if q.requires_grad and id(q) not in usati]
    if enc:
        gruppi.append({"params": enc, "lr": lr * float(cfg.get("encoder_lr_mult", 0.1))})
        usati |= {id(q) for q in enc}
    resto = [p for p in model.parameters() if p.requires_grad and id(p) not in usati]
    gruppi.insert(0, {"params": resto, "lr": lr})
    return gruppi


def _passo(model, batch, T, pos_weight, n_ref, refine_lambda, hsam_lambda, dev):
    imgs = F.interpolate(batch["image"].to(dev), (T, T), mode="bilinear", align_corners=False)
    masks = F.interpolate(batch["mask"].to(dev).unsqueeze(1).float(), (T, T),
                          mode="nearest").squeeze(1).long()

    def priore():
        pr = getattr(model, "ultimo_prior", None)
        if getattr(model, "hsam", None) is None or pr is None:
            return 0.0
        return hsam_lambda * criterion(pr, masks, pos_weight=pos_weight)

    preds = get_best_mask(*model(imgs, None))
    loss = criterion(preds, masks, pos_weight=pos_weight) + priore()
    for _ in range(n_ref):
        r = as_refine_map(preds, model.mask_input_size[0])
        preds = get_best_mask(*model(imgs, None, refine=r))
        loss = refine_lambda * loss + criterion(preds, masks, pos_weight=pos_weight) + priore()
    return loss


@torch.no_grad()
def _valida(model, loader, T, n_ref, dev, limite=0):
    """Micro IoU alla soglia 0.5, sulla griglia NATIVA come nella sonda."""
    model.eval()
    tp = fp = fn = 0.0
    for j, b in enumerate(loader):
        if limite and j >= limite:
            break
        imgs, masks = b["image"].to(dev), b["mask"].to(dev)
        h, w = imgs.shape[-2:]
        x = F.interpolate(imgs, (T, T), mode="bilinear", align_corners=False)
        with torch.autocast(device_type="cuda", enabled=str(dev).startswith("cuda")):
            lg = get_best_mask(*model(x, None))
            for _ in range(n_ref):
                lg = get_best_mask(*model(x, None, refine=as_refine_map(lg, model.mask_input_size[0])))
        p = torch.sigmoid(F.interpolate(lg.float(), (h, w), mode="bilinear",
                                        align_corners=False)) > 0.5
        g = masks.unsqueeze(1) > 0
        tp += float((p & g).sum()); fp += float((p & ~g).sum()); fn += float((~p & g).sum())
    model.train()
    return tp / max(tp + fp + fn, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--unfreeze-blocks", type=int, default=2)
    ap.add_argument("--fusione", default="entrambi", choices=["none", "mgem", "dgfm", "entrambi"])
    ap.add_argument("--fusione-aux", default="terreno", choices=["terreno"])
    ap.add_argument("--fpn-completo", action="store_true")
    ap.add_argument("--hsam", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=2,
                    help="2 = un batch per dominio a ogni passo dell'ottimizzatore")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pesi-loss", default="incertezza", choices=["incertezza", "fissi"],
                    help="incertezza: pesi delle due loss appresi (Kendall 2018); fissi: 1 e 1")
    ap.add_argument("--criterio", default="media", choices=["media", "l4s", "bijie"],
                    help="su cosa si sceglie il checkpoint: media dei due micro o un dominio")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limite", type=int, default=0, help="solo per prove: passi per epoca")
    ap.add_argument("--resume", action="store_true",
                    help="riprende dall'ultima epoca finita (file last_), oppure, se manca, "
                         "dai pesi del checkpoint migliore")
    ap.add_argument("--fold", type=int, default=0,
                    help="fold dello split a rotazione (0-4), lo stesso nei due domini. "
                         "0 = lo split di sempre")
    a = ap.parse_args()

    cfgs = {}
    for d in DOMINI:
        c = load_config(a.config, dataset=d, prompt_source="none")
        c.update({"target_size": a.target_size, "unfreeze_blocks": a.unfreeze_blocks,
                  "fusione": a.fusione, "fusione_aux": a.fusione_aux,
                  "fpn_completo": a.fpn_completo, "hsam": a.hsam, "fold": a.fold})
        cfgs[d] = c
    cfg = cfgs["l4s"]                      # architettura e iperparametri condivisi
    dev = cfg["device"]
    configure_criterion(cfg)
    T, n_ref = a.target_size, int(cfg.get("refine_passes", 0))
    refine_lambda = float(cfg.get("refine_lambda", 0.5))
    hsam_lambda = float(cfg.get("hsam_lambda", 0.5))

    dati = {d: _dominio(cfgs[d], a.batch_size, a.workers) for d in DOMINI}
    model = costruisci_multi(cfg, dev, a.config)
    with torch.no_grad():
        for d in DOMINI:
            getattr(model, f"media_{d}").copy_(dati[d]["media"])
            getattr(model, f"dev_{d}").copy_(dati[d]["dev"])
    for d in DOMINI:
        print(f"  {d}: train {dati[d]['n_tr']} (con le copie) | pos_weight "
              f"{dati[d]['pos_weight']:.2f} | GSD {model.gsd[d]} m")

    # pesi delle due loss: s_d con L = exp(-s_d) L_d + s_d; a "fissi" restano a zero
    s = nn.Parameter(torch.zeros(len(DOMINI), device=dev),
                     requires_grad=(a.pesi_loss == "incertezza"))
    marche = (f"_unf{a.unfreeze_blocks}" + ("" if a.fusione == "none" else f"_{a.fusione}")
              + ("_fpn" if a.fpn_completo else "") + ("_hsam" if a.hsam else "")
              + ("" if a.pesi_loss == "incertezza" else "_pesifissi")
              + f"_{T}px_sel{a.criterio}" + (f"_fold{a.fold}" if a.fold else ""))
    nome = a.checkpoint or f"best_lente_multi_v3{marche}.pth"
    ottim = torch.optim.AdamW(_gruppi(model, cfg, [s] if s.requires_grad else []),
                              weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(ottim, mode="max", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(enabled=str(dev).startswith("cuda"))
    print(f"\n  checkpoint {nome} | ingresso {T} | batch {a.batch_size} x {a.accum_steps} | "
          f"pesi delle loss: {a.pesi_loss} | selezione su {a.criterio}")

    storia = {"checkpoint": nome, "inizio": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": vars(a), "epoche": []}
    migliore, attesa, inizio = -1.0, 0, 1
    cart, base = os.path.split(os.path.splitext(nome)[0])
    # Se il nome non comincia con best_ (es. --checkpoint prova.pth), il replace
    # non cambierebbe niente e lo stato di fine epoca sovrascriverebbe il migliore.
    ultimo = os.path.join(cart, (base.replace("best_", "last_", 1) if base.startswith("best_")
                                 else "last_" + base) + ".pth")
    if a.resume:
        if os.path.exists(ultimo):
            # ripresa esatta: pesi, ottimizzatore, scheduler, pesi delle loss, pazienza
            st = torch.load(ultimo, map_location=dev, weights_only=False)
            model.load_state_dict(st["model"])
            ottim.load_state_dict(st["ottim"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            with torch.no_grad():
                s.copy_(st["s"])
            migliore, attesa, storia = st["migliore"], st["attesa"], st["storia"]
            inizio = st["epoca"] + 1
            print(f"\n  RIPRESA ESATTA da {ultimo}: epoca {st['epoca']} finita")
        elif os.path.exists(nome):
            # solo i pesi del migliore: l'ottimizzatore riparte da capo, i pesi
            # delle loss ripartono dai valori salvati
            st = torch.load(nome, map_location=dev, weights_only=False)
            model.load_state_dict(st["model_state_dict"])
            with torch.no_grad():
                for j, d in enumerate(DOMINI):
                    s[j] = -math.log(st["pesi_loss"][d])
            migliore = (sum(st["micro"].values()) / 2 if a.criterio == "media"
                        else st["micro"][a.criterio])
            inizio = st["epoch"] + 1
            print(f"\n  RIPRESA dai pesi di {nome} (epoca {st['epoch']}): ottimizzatore e "
                  f"scheduler ripartono da capo | record {migliore:.4f}")
        else:
            raise SystemExit(f"--resume: ne' {ultimo} ne' {nome} esistono")
    model.train()
    for ep in range(inizio, a.epochs + 1):
        passi = max(len(dati[d]["ld_tr"]) for d in DOMINI)
        if a.limite:
            passi = min(passi, a.limite)
        iteratori = {d: _ciclo(dati[d]["ld_tr"]) for d in DOMINI}
        somma = {d: 0.0 for d in DOMINI}
        ottim.zero_grad(set_to_none=True)
        k = 0
        for _ in tqdm(range(passi), desc=f"Ep {ep}/{a.epochs}"):
            for j, d in enumerate(DOMINI):                     # L4S, Bijie, L4S, ...
                model.imposta_dominio(d)
                with torch.autocast(device_type="cuda", enabled=str(dev).startswith("cuda")):
                    l = _passo(model, next(iteratori[d]), T, dati[d]["pos_weight"],
                               n_ref, refine_lambda, hsam_lambda, dev)
                    loss = torch.exp(-s[j]) * l + s[j]
                scaler.scale(loss / a.accum_steps).backward()
                somma[d] += float(l)
                k += 1
                if k % a.accum_steps == 0:
                    scaler.step(ottim)
                    scaler.update()
                    ottim.zero_grad(set_to_none=True)
        micro = {}
        for d in DOMINI:
            model.imposta_dominio(d)
            micro[d] = _valida(model, dati[d]["ld_va"], T, n_ref, dev, a.limite)
        punteggio = (micro["l4s"] + micro["bijie"]) / 2 if a.criterio == "media" else micro[a.criterio]
        sched.step(punteggio)
        pesi = {d: float(torch.exp(-s[j])) for j, d in enumerate(DOMINI)}
        riga = {"epoca": ep, **{f"train_loss_{d}": somma[d] / passi for d in DOMINI},
                **{f"val_micro_{d}": micro[d] for d in DOMINI},
                **{f"peso_loss_{d}": pesi[d] for d in DOMINI},
                "punteggio": punteggio, "lr": ottim.param_groups[0]["lr"]}
        print(f"EPOCA {ep} | micro val  L4S {micro['l4s']:.4f}  Bijie {micro['bijie']:.4f} | "
              f"{a.criterio} {punteggio:.4f} | pesi loss L4S {pesi['l4s']:.2f} "
              f"Bijie {pesi['bijie']:.2f}")
        if punteggio > migliore:
            migliore, attesa = punteggio, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "micro": micro,
                        "pesi_loss": pesi}, nome)
            riga["record"] = True
            print(f"  Record: {a.criterio} {punteggio:.4f} — checkpoint salvato in {nome}")
        else:
            attesa += 1
        storia["epoche"].append(riga)
        torch.save({"model": model.state_dict(), "ottim": ottim.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "s": s.detach().clone(), "epoca": ep, "migliore": migliore,
                    "attesa": attesa, "storia": storia}, ultimo + ".tmp")
        os.replace(ultimo + ".tmp", ultimo)
        cart, base = os.path.split(os.path.splitext(nome)[0])
        with open(os.path.join(cart, f"history_{base}.json"), "w") as f:
            json.dump(storia, f, indent=1)
        if attesa >= a.patience:
            print(f"\nEARLY STOPPING — miglior {a.criterio} {migliore:.4f}")
            break


if __name__ == "__main__":
    main()

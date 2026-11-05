#!/usr/bin/env python
"""
banco.py — modifica DIRETTA dei pesi di LENTE, senza addestramento.

Adattamento a LENTE dell'esperimento 8 di "Sparks of Weight Space Programming"
(github.com/RikiyaT/lm_understand-logs, exp8_math): un modello linguistico
modifica i pesi di una rete leggendoli e guardando un numero LIMITATO di
passate in avanti su dati di pratica, mai il test. Qui l'editor e' Claude, la
rete e' LENTE L4S singolo unf2 del fold 0.

REGOLE, copiate dall'originale e fatte rispettare dal banco dove si puo':
  - niente addestramento: niente backward, autograd, ottimizzatori, ne'
    adattamento iterativo a una loss. Il banco gira tutto in no_grad e con
    requires_grad spento;
  - la rete si esegue SOLO tramite `passata`, con un budget fisso (25). Una
    passata = fino a 512 patch, con i pesi originali e/o modificati, e
    restituisce probabilita', maschere e le attivazioni richieste;
  - le passate usano SOLO patch del TRAIN del fold 0. Val e test non sono
    raggiungibili da qui: la soglia si sceglie sul val dopo, il test si misura
    una volta sola alla fine, con LENTE_sonde.py;
  - i pesi si leggono e si elaborano liberamente (niente budget): sono numeri,
    non esecuzioni della rete;
  - vietato modificare l'ultimo stadio (le MLP che producono la maschera, le
    teste di IoU e di oggetto), i parametri scalari e moltiplicare un intero
    tensore per una costante: sarebbero modi di spostare la soglia, che e'
    gia' scelta sul val;
  - ogni passata e ogni modifica richiedono una spiegazione, che finisce nel
    registro (`registro.md`), come il loro "EXPLAIN EVERY STEP".

Tutto cio' che una passata restituisce viene salvato su disco: rianalizzarlo
non consuma budget.
"""

import json
import os
import time

import numpy as np
import torch

import LENTE_sonde as S
from LENTE_utils import load_config, build_dataset, build_train_val_test_subsets, TransformSubset
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

QUI = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(os.path.dirname(QUI), "best_lente_l4s_free_selmicro_unf2_entrambi24.pth")
BUDGET = 25
MAX_PATCH = 512
VIETATI = ("model.mask_decoder.output_hypernetworks_mlps", "model.mask_decoder.iou_prediction_head",
           "model.mask_decoder.pred_obj_score_head", "model.mask_decoder.iou_token",
           "model.mask_decoder.obj_score_token")
# U-Net nativa (rete_nativa.py): l'ultimo stadio e' la convoluzione 3x3 che
# produce il logit, `segmentation_head`
CKPT_UNET = os.path.join(os.path.dirname(QUI), "best_rete_nativa_l4s.pth")
VIETATI_UNET = ("segmentation_head",)


def _cfg():
    cfg = load_config(os.path.join(os.path.dirname(QUI), "config.yaml"), dataset="l4s",
                      prompt_source="none")
    cfg.update({"target_size": 512, "unfreeze_blocks": 2, "fusione": "entrambi",
                "fusione_aux": "terreno", "fold": 0})
    return cfg


class Banco:
    def __init__(self, run="run1", budget=BUDGET, rete="lente"):
        self.budget = budget
        self.rete = rete
        self.vietati = VIETATI if rete == "lente" else VIETATI_UNET
        self.dir = os.path.join(QUI, run)
        os.makedirs(os.path.join(self.dir, "passate"), exist_ok=True)
        self.cfg = _cfg()
        self.dev = self.cfg["device"]
        full = build_dataset(self.cfg)
        tr, va, _, _ = build_train_val_test_subsets(full, self.cfg)
        self.full, self.train_idx = full, list(tr.indices)   # il test: mai tenuto
        # Il val e' raggiungibile SOLO con insieme="val", per le conferme
        # (dalla run 3: il train non ha piu' patch vuote mai guardate).
        self.val_idx = list(va.indices)
        self.pos = set(full.positive_indices)
        if rete == "lente":
            self.model = S._carica_modello(self.cfg, self.dev, CKPT)
        else:
            from rete_nativa import costruisci
            ck = torch.load(CKPT_UNET, map_location="cpu")
            self.model = costruisci(ck["n_canali"], ck["encoder"]).to(self.dev)
            self.model.load_state_dict(ck["model_state_dict"], strict=True)
            print(f"  U-Net {ck['encoder']} | epoca {ck['epoch']} | val micro {ck['micro']:.4f}")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.base = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.f_stato = os.path.join(self.dir, "stato.json")
        self.f_edit = os.path.join(self.dir, "edit.pt")
        self.stato = (json.load(open(self.f_stato)) if os.path.exists(self.f_stato)
                      else {"passate": 0, "storia": []})
        self.edit = torch.load(self.f_edit) if os.path.exists(self.f_edit) else {}

    # ---------------- pesi: lettura libera
    def pesi(self):
        """Copia dei pesi ORIGINALI (CPU)."""
        return {k: v.clone() for k, v in self.base.items()}

    def pesi_preaddestrati(self):
        """I pesi di partenza dell'addestramento (SAM2 preaddestrato + init dei moduli nuovi), CPU.
        Utili solo per le parti venute da SAM2 (chiavi `model.`): i moduli nuovi
        hanno init casuale o a zero, e interpolare verso quella non ha senso."""
        from LENTE_utils import build_model
        torch.manual_seed(0)
        m = build_model(self.cfg, "cpu")
        return {k: v.detach().clone() for k, v in m.state_dict().items() if k.startswith("model.")}

    # ---------------- modifiche
    def salva_edit(self, nome, tensore, spiegazione):
        if not spiegazione or len(spiegazione) < 20:
            raise ValueError("serve una spiegazione")
        if nome not in self.base:
            raise KeyError(nome)
        if nome.startswith(self.vietati):
            raise PermissionError(f"{nome}: ultimo stadio, vietato")
        b = self.base[nome]
        t = tensore.detach().cpu().to(b.dtype)
        if t.shape != b.shape:
            raise ValueError(f"forma {tuple(t.shape)} != {tuple(b.shape)}")
        if b.numel() == 1:
            raise PermissionError(f"{nome}: parametro scalare, vietato")
        if b.abs().sum() > 0:
            c = (t.flatten() @ b.flatten()) / (b.flatten() @ b.flatten())
            if torch.allclose(t, c * b, rtol=1e-4, atol=1e-7) and not torch.allclose(c, torch.ones(())):
                raise PermissionError(f"{nome}: tensore intero per una costante, vietato")
        self.edit[nome] = t
        torch.save(self.edit, self.f_edit)
        self._registra(f"EDIT {nome} (|delta| {float((t - b).norm()):.4g} su |w| {float(b.norm()):.4g})",
                       spiegazione)

    def togli_edit(self, nome, spiegazione):
        self.edit.pop(nome, None)
        torch.save(self.edit, self.f_edit)
        self._registra(f"TOLTO EDIT {nome}", spiegazione)

    def _carica(self, versione):
        sd = dict(self.base)
        if versione == "modificato":
            sd.update(self.edit)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

    # ---------------- passate: budget
    def passata(self, indici, spiegazione, versioni=("base",), cattura=(), insieme="train"):
        """
        Esegue la rete su `indici` (posizioni in self.train_idx, max 512) con le
        versioni richieste. `cattura`: nomi di moduli di cui salvare l'uscita.
        Consuma UNA passata. Restituisce e salva un dict per versione con
        prob (N,1,128,128) float16, mask (N,128,128) uint8, att {modulo: tensore}.
        """
        if self.stato["passate"] >= self.budget:
            raise RuntimeError("budget di passate esaurito")
        if not spiegazione or len(spiegazione) < 20:
            raise ValueError("serve una spiegazione")
        indici = list(indici)
        if insieme == "val" and "CONFERMA" not in spiegazione:
            raise PermissionError("il val serve solo per le conferme")
        pool = {"train": self.train_idx, "val": self.val_idx}[insieme]
        if len(indici) > MAX_PATCH or not all(0 <= i < len(pool) for i in indici):
            raise ValueError("indici fuori dall'insieme o piu' di 512")
        sub = Subset(self.full, [pool[i] for i in indici])
        ld = DataLoader(TransformSubset(sub, A.Compose([ToTensorV2()])), batch_size=4, shuffle=False)
        moduli = dict(self.model.named_modules())
        out = {}
        with torch.no_grad():
            for v in versioni:
                self._carica(v)
                att = {m: [] for m in cattura}
                # "nome:stat" = solo somma e somma dei quadrati per canale (per
                # mappe troppo grandi da salvare intere, es. le statistiche di BN)
                def _gancio(mod, inp, o, m):
                    t = (o[0] if isinstance(o, (tuple, list)) else o).detach().float()
                    if m.endswith(":stat"):
                        x = t.transpose(0, 1).reshape(t.shape[1], -1)
                        att[m].append(torch.stack([x.sum(1), (x * x).sum(1),
                                                   torch.full((x.shape[0],), float(x.shape[1]), device=x.device)]).cpu()[None])
                    else:
                        att[m].append(t.half().cpu())
                ganci = [moduli[m.split(":")[0]].register_forward_hook(
                    lambda mod, inp, o, m=m: _gancio(mod, inp, o, m)) for m in cattura]
                P, M = [], []
                try:
                    for p, m in self._probabilita(ld, v):
                        P.append(p.half().cpu()); M.append(m.to(torch.uint8).cpu())
                finally:
                    for g in ganci:
                        g.remove()
                out[v] = {"prob": torch.cat(P), "mask": torch.cat(M),
                          "att": {m: torch.cat(x) for m, x in att.items() if x}}
        self.stato["passate"] += 1
        k = self.stato["passate"]
        torch.save({"indici": indici, "versioni": list(versioni), "out": out, "insieme": insieme},
                   os.path.join(self.dir, "passate", f"p{k:02d}.pt"))
        self._registra(f"PASSATA {k}/{self.budget}: {len(indici)} patch di {insieme}, versioni {list(versioni)}, "
                       f"cattura {list(cattura)}", spiegazione)
        json.dump(self.stato, open(self.f_stato, "w"), indent=1)
        return out

    def _probabilita(self, ld, desc):
        """(probabilita' sulla griglia nativa, maschera) per batch."""
        if self.rete == "lente":
            yield from S.probabilita(self.model, ld, self.cfg, self.dev, desc=desc)
            return
        # la U-Net lavora gia' a 128 px nativi: niente ingrandimento, niente rilettura
        for b in ld:
            x = b["image"].to(self.dev)
            yield torch.sigmoid(self.model(x).float()), b["mask"]

    def rileggi(self, k):
        """Rilegge la passata k da disco: non consuma budget."""
        return torch.load(os.path.join(self.dir, "passate", f"p{k:02d}.pt"))

    # ---------------- uscita
    def checkpoint_modificato(self, nome="best_lente_l4s_pesi_diretti.pth"):
        sd = dict(self.base)
        sd.update(self.edit)
        percorso = os.path.join(os.path.dirname(QUI), nome)
        torch.save({"model_state_dict": sd, "edit_di": os.path.basename(CKPT),
                    "edit": sorted(self.edit)}, percorso)
        self._registra(f"CHECKPOINT {nome} con {len(self.edit)} tensori modificati", "fine del lavoro")
        return percorso

    def _registra(self, cosa, perche):
        voce = {"quando": time.strftime("%Y-%m-%d %H:%M:%S"), "cosa": cosa, "perche": perche}
        self.stato["storia"].append(voce)
        json.dump(self.stato, open(self.f_stato, "w"), indent=1)
        with open(os.path.join(self.dir, "registro.md"), "a", encoding="utf-8") as f:
            f.write(f"- **{voce['quando']}** {cosa}\n  {perche}\n")

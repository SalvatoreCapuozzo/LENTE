#!/usr/bin/env python
"""
LENTE_boxhead.py — prompt box APPRESO, con e senza EMA.
=======================================================
Il GT box non e' illegittimo come *supervisione*: lo e' come *input a test time*,
perche' per calcolarlo serve la maschera, cioe' la risposta. Qui il box torna in
gioco nel modo in cui lo usa qualunque object detector: come target di una testa
che impara a predirlo, e che a inferenza se lo produce da sola.

Struttura dei risultati che ne esce (misurati sullo stesso split L4S):

    nessun prompt        IoU 0.5917     pavimento
    box appreso          IoU   ?        il sistema vero
    box dalla GT         IoU 0.7501     soffitto (oracolo: box perfetto)

Verificato che il gradiente attraversa le coordinate del box: l'encoding di
Sam2PositionalEmbedding e' sin/cos(2*pi * (2c-1) @ W), liscio in c. Quindi la
testa riceve gradiente sia dalla regressione sul box sia, attraverso il prompt,
dalla loss di segmentazione.

Due varianti, selezionabili dal menu di LENTE_run.py:
    boxhead       -> training normale
    boxhead_ema   -> in piu' tiene una media esponenziale dei pesi (Polyak) e
                     valuta/salva QUELLA.

NOTA su cosa e' l'EMA qui: e' EMA dei pesi, non ancora il Mean Teacher. Il Mean
Teacher richiede un dominio target NON etichettato e una consistency loss, quindi
lo split leave-one-region-out. Questa modalita' e' il passo che lo precede: dice
se l'EMA aiuta, prima di investire nella pipeline semi-supervisionata. E la testa
di box e' comunque il prerequisito, perche' sul target senza etichette il teacher
un box deve pur produrlo.

Uso:
    python LENTE_run.py -d l4s --mode boxhead
    python LENTE_run.py -d l4s --mode boxhead_ema
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    LENTE, criterion, get_best_mask, calculate_metrics, scale_boxes,
    save_checkpoint, jitter_boxes,
)


# ============================================================
# TESTA DI PREDIZIONE DEL BOX
# ============================================================

class BoxPromptHeadGAP(nn.Module):
    """
    Variante "gap" — la prima versione, tenuta per confronto.

    Da (B, in_ch, H, W) — lo stack COMPLETO, ottico + slope + DEM non filtrati
    (vedi LENTEBoxHead.__init__) — a:
      - box   (B, 4)  in coordinate NORMALIZZATE [0,1], ordine (x0, y0, x1, y1)
      - obj   (B,)    logit di objectness ("c'e' una frana in questa patch?")

    Parametrizzazione (cx, cy, w, h) invece dei quattro angoli: garantisce per
    costruzione x0 < x1 e y0 < y1.

    LIMITE NOTO, misurato: il global average pooling collassa la mappa 8x8 a
    1x1 e media via tutta la geometria. Alla Linear finale arriva "quanta frana
    c'e'" ma non "dove". Su L4S la box L1 si e' fermata a ~0.19 dopo 20 epoche,
    cioe' la testa imparava la dimensione tipica e non la posizione. Resta
    selezionabile come termine di paragone per la variante "heatmap".
    """

    def __init__(self, in_ch, n_terrain=2, width=32, work_size=128,
                 terrain_prior=True):
        super().__init__()
        self.work_size = work_size

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(8, cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_ch, width),            # 128 -> 64
            block(width, width * 2),        #  64 -> 32
            block(width * 2, width * 4),    #  32 -> 16
            block(width * 4, width * 4),    #  16 ->  8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.box  = nn.Linear(width * 4, 4)   # cx, cy, w, h (pre-sigmoid)
        self.obj  = nn.Linear(width * 4, 1)

        # Pesi piccoli ma NON nulli, bias a zero: con weight=0 il gradiente non
        # risale oltre queste Linear e il tronco resta fermo.
        for layer in (self.box, self.obj):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x_stack):
        x = F.interpolate(x_stack, (self.work_size, self.work_size),
                          mode="bilinear", align_corners=False)
        h = self.pool(self.features(x)).flatten(1)

        cx, cy, w, hh = torch.sigmoid(self.box(h)).unbind(dim=-1)
        w  = w.clamp(min=0.02)
        hh = hh.clamp(min=0.02)
        x0 = (cx - w / 2).clamp(0.0, 1.0)
        x1 = (cx + w / 2).clamp(0.0, 1.0)
        y0 = (cy - hh / 2).clamp(0.0, 1.0)
        y1 = (cy + hh / 2).clamp(0.0, 1.0)
        box = torch.stack([x0, y0, x1, y1], dim=-1)
        # terzo valore = mappa densa: questa variante non ne produce (il pooling
        # medio l'ha gia' distrutta), quindi None. Con boxhead_prompt "dense" o
        # "both" va usata la variante "heatmap".
        return box, self.obj(h).squeeze(-1), None


class BoxPromptHeadHeatmap(nn.Module):
    """
    Variante "heatmap" — il box si legge dai momenti di una mappa di calore
    spaziale invece che da una regressione su un vettore mediato.

    Stessi ingressi e stesse uscite di BoxPromptHeadGAP, cosi' le due sono
    intercambiabili e confrontabili a parita' di tutto il resto.
    """

    def __init__(self, in_ch, n_terrain=2, width=32, work_size=128,
                 terrain_prior=True):
        super().__init__()
        self.work_size = work_size
        self.n_terrain = n_terrain

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(8, cout),      # GroupNorm, non BatchNorm: niente
                nn.ReLU(inplace=True),      # running stats da mediare fra domini
            )

        # TRE blocchi, non quattro: la mappa si ferma a 16x16 e NON viene
        # collassata a 1x1. La versione precedente finiva in AdaptiveAvgPool2d(1),
        # che media via tutta la geometria: alla testa arrivava un descrittore
        # "quanta frana c'e'" senza il "dove". Misurato: box L1 ferma a ~0.19
        # dopo 20 epoche, cioe' la testa imparava la dimensione tipica e non la
        # posizione.
        self.features = nn.Sequential(
            block(in_ch, width),            # 128 -> 64
            block(width, width * 2),        #  64 -> 32
            block(width * 2, width * 4),    #  32 -> 16
        )

        # Mappa di calore spaziale: da qui il box si ricava per centroide.
        self.heat = nn.Conv2d(width * 4, 1, kernel_size=1)

        # Branch di terreno (opzionale, `boxhead_terrain_prior`): prior fisico
        # "le frane stanno sui pendii" che si somma ai logit della heatmap.
        # E' SEPARATA da DSDFEarlyGuidance apposta: riusare quella rimetterebbe
        # la testa in dipendenza dalla DSDF e l'ablazione `no_dfds` tornerebbe a
        # cambiare due cose insieme.
        # Nota: lo slope da solo indica TUTTO il terreno ripido, non le frane —
        # e' un prior, non una risposta; l'ottico lo modula tramite self.heat.
        self.terrain_prior = nn.Sequential(
            nn.Conv2d(n_terrain, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        ) if terrain_prior else None

        # L'objectness invece resta su pooling medio: "c'e' una frana?" e' una
        # domanda di classificazione, e li' mediare e' la scelta giusta.
        self.obj = nn.Linear(width * 4, 1)

        # Fattore che lega la dispersione della heatmap al lato del box.
        # Per una mappa uniforme su [0,1] la deviazione standard vale 1/sqrt(12)
        # ~= 0.289, quindi ~3.46 sigma coprono tutto: si parte da 4 e si lascia
        # che il modello aggiusti in scala logaritmica.
        self.log_scale = nn.Parameter(torch.zeros(1))

        # Pesi piccoli ma NON nulli, bias a zero. Con weight=0 il gradiente non
        # risale oltre l'ultimo strato e il tronco a monte resta fermo: e' il
        # bug gia' incontrato (3 tensori su 20 con gradiente).
        _to_init = [self.heat, self.obj]
        if self.terrain_prior is not None:
            _to_init.append(self.terrain_prior[-1])
        for layer in _to_init:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x_stack):
        x = F.interpolate(x_stack, (self.work_size, self.work_size),
                          mode="bilinear", align_corners=False)
        feat = self.features(x)                              # (B, C, G, G)
        g = feat.shape[-1]

        logits = self.heat(feat)                             # (B, 1, G, G)
        if self.terrain_prior is not None:
            terr = F.interpolate(x_stack[:, -self.n_terrain:], (g, g),
                                 mode="bilinear", align_corners=False)
            logits = logits + self.terrain_prior(terr)

        # soft-argmax: la heatmap normalizzata diventa una distribuzione di
        # probabilita' spaziale, e il box si legge dai suoi momenti. Tutto
        # differenziabile, e la geometria non viene mai buttata.
        attn = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)

        # centro di ogni cella in coordinate normalizzate [0,1]
        coord = (torch.arange(g, device=attn.device, dtype=attn.dtype) + 0.5) / g
        xs = coord.view(1, 1, 1, g)
        ys = coord.view(1, 1, g, 1)

        cx = (attn * xs).sum(dim=(1, 2, 3))                  # centroide
        cy = (attn * ys).sum(dim=(1, 2, 3))
        varx = (attn * (xs - cx.view(-1, 1, 1, 1)) ** 2).sum(dim=(1, 2, 3))
        vary = (attn * (ys - cy.view(-1, 1, 1, 1)) ** 2).sum(dim=(1, 2, 3))

        k = 4.0 * torch.exp(self.log_scale)                  # dispersione -> lato
        w  = (k * varx.clamp_min(1e-8).sqrt()).clamp(0.02, 1.0)
        hh = (k * vary.clamp_min(1e-8).sqrt()).clamp(0.02, 1.0)

        x0 = (cx - w / 2).clamp(0.0, 1.0)
        x1 = (cx + w / 2).clamp(0.0, 1.0)
        y0 = (cy - hh / 2).clamp(0.0, 1.0)
        y1 = (cy + hh / 2).clamp(0.0, 1.0)
        box = torch.stack([x0, y0, x1, y1], dim=-1)

        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        # `logits` viene restituita GREZZA (non il softmax): e' la mappa densa da
        # dare a SAM2 come `input_masks`, che si aspetta logit, non probabilita'.
        # Il softmax serve solo qui dentro per il centroide.
        return box, self.obj(pooled).squeeze(-1), logits


BOX_HEADS = {"gap": BoxPromptHeadGAP, "heatmap": BoxPromptHeadHeatmap}


def build_box_head(kind, in_ch, n_terrain=2, width=32, work_size=128,
                   terrain_prior=True):
    """Factory delle due varianti di testa (config: `boxhead_type`)."""
    if kind not in BOX_HEADS:
        raise ValueError(f"boxhead_type sconosciuto: {kind!r} (attesi: {list(BOX_HEADS)})")
    return BOX_HEADS[kind](in_ch, n_terrain=n_terrain, width=width,
                           work_size=work_size, terrain_prior=terrain_prior)


# ============================================================
# MODELLO: LENTE + testa di box
# ============================================================

class LENTEBoxHead(LENTE):
    """
    LENTE con il prompt prodotto da BoxPromptHead invece che dalla ground truth.

    forward(x, gt_boxes=None, use_gt_prompt=False) -> (pred_masks, scores, box, obj)

    `use_gt_prompt` serve SOLO in training, per lo scheduled sampling: nelle prime
    epoche il box predetto e' rumore e il decoder impara su un prompt pessimo.
    In validazione e in test va sempre False, altrimenti si torna al leak.
    """

    PROMPT_MODES = ("box", "dense", "both")

    def __init__(self, sam_ckpt_path, n_optical=12, adapter_dim=64,
                 head_width=32, head_work_size=128,
                 head_type="heatmap", head_terrain_prior=True,
                 prompt_mode="box", image_size=1024, adapter_where="attn"):
        # prompt_source="none" per il padre: il prompt lo costruiamo qui.
        # image_size DEVE arrivare da cfg["target_size"]: e' la dimensione a cui
        # train_boxhead/test_boxhead interpolano le immagini, e SAM2 va costruito
        # sulla stessa. Se divergono, `_prepare_vision_features` rimodella le
        # feature su una griglia sbagliata e il forward esplode.
        super().__init__(sam_ckpt_path, n_optical=n_optical,
                         adapter_dim=adapter_dim, prompt_source="none",
                         image_size=image_size, adapter_where=adapter_where)
        self.prompt_source = "learned_box"          # solo per log/checkpoint
        # Creata DOPO _freeze() del padre, quindi i suoi parametri restano
        # requires_grad=True senza doverli riabilitare a mano.
        #
        # in_ch = n_optical + 2, cioe' lo stack COMPLETO (ottico + slope + DEM),
        # non l'ottico modulato dalla DSDF. Il gate DSDF comprime slope e DEM in
        # un solo scalare moltiplicativo per pixel: per l'encoder e' una scelta
        # difendibile, per una testa che deve LOCALIZZARE no, perche' il terreno
        # e' il segnale piu' informativo che ha (le frane stanno sui pendii).
        # E' la stessa conclusione di VIDEO_ALL_CHANNELS in Sen12Landslides/config.py.
        # Cosi' la testa e' anche indipendente dalla DSDF: si puo' ablare l'una
        # senza cambiare l'input dell'altra.
        self.box_head = build_box_head(head_type, n_optical + 2, n_terrain=2,
                                       width=head_width, work_size=head_work_size,
                                       terrain_prior=head_terrain_prior)
        self.head_type = head_type

        # ── canale del prompt ────────────────────────────────────────────
        # "box"   -> solo sparse (due token d'angolo), come finora
        # "dense" -> solo input_masks: la heatmap intera invece che compressa
        #            in quattro numeri. E' il canale che SAM2 usava per il
        #            raffinamento iterativo e che finora era sempre rimasto al
        #            default `no_mask_embed`.
        # "both"  -> entrambi: sparse e dense sono indipendenti nel prompt
        #            encoder, quindi non si escludono.
        if prompt_mode not in self.PROMPT_MODES:
            raise ValueError(f"boxhead_prompt sconosciuto: {prompt_mode!r} "
                             f"(attesi: {self.PROMPT_MODES})")
        if prompt_mode in ("dense", "both") and head_type != "heatmap":
            raise ValueError(
                f"boxhead_prompt={prompt_mode!r} richiede boxhead_type='heatmap': "
                f"la variante '{head_type}' non produce nessuna mappa densa.")
        self.prompt_mode = prompt_mode

        # Scala appresa dei logit densi. `mask_embed` e' una conv, quindi
        # l'AMPIEZZA dei valori conta: logit tarati per un softmax spaziale non
        # hanno per forza la scala giusta per quel canale. exp(0)=1 all'inizio.
        self.dense_log_scale = nn.Parameter(torch.zeros(1))
        # Risoluzione attesa da input_masks (256x256 con image_size=1024).
        self.mask_input_size = tuple(self.model.prompt_encoder.mask_input_size)

    def forward(self, x, gt_boxes=None, use_gt_prompt=False, gt_prompt_mask=None):
        n = self.n_optical
        optical = x[:, :n]
        terrain = x[:, n:]

        # La testa legge lo stack COMPLETO e non filtrato (vedi __init__):
        # il terrain le serve intero per localizzare, non compresso nel gate.
        size = x.shape[-1]
        box_norm, obj_logit, heat = self.box_head(x)

        # L'ottico che va a SAM resta modulato dalla DSDF, come in LENTE.
        optical = optical * self.early_guidance(terrain)

        # da normalizzato a pixel; il gradiente risale fino a box_head
        pred_px = box_norm * (size - 1)

        # ── canale sparse (box) ──────────────────────────────────────────
        input_boxes = None
        if self.prompt_mode in ("box", "both"):
            if use_gt_prompt:
                if gt_boxes is None:
                    raise ValueError("use_gt_prompt=True ma nessun gt_boxes fornito.")
                gt = gt_boxes.squeeze(1) if gt_boxes.ndim == 3 else gt_boxes
                gt = gt.to(pred_px.dtype)
                if gt_prompt_mask is not None:
                    # Il box GT come prompt SOLO sui campioni che ce l'hanno
                    # (positivi). I negativi usano il box predetto: dargli il
                    # fittizio [0,0,10,10] ricreerebbe, per le epoche di teacher
                    # forcing, la stessa asimmetria "box nell'angolo <=> negativo"
                    # che era il leak di classificazione appena rimosso.
                    input_boxes = torch.where(gt_prompt_mask.unsqueeze(1), gt, pred_px)
                else:
                    input_boxes = gt
            else:
                input_boxes = pred_px
            input_boxes = input_boxes.unsqueeze(1)

        # ── canale denso (heatmap) ───────────────────────────────────────
        # NIENTE teacher forcing qui: l'analogo del box GT sarebbe dare la
        # MASCHERA GT come prompt, cioe' l'input coinciderebbe con il bersaglio
        # e il modello imparerebbe a copiarlo. E' una scorciatoia molto peggiore
        # del gt_box. La mappa e' sempre e solo quella predetta.
        input_masks = None
        if self.prompt_mode in ("dense", "both"):
            dense = F.interpolate(heat.float(), self.mask_input_size,
                                  mode="bilinear", align_corners=False)
            input_masks = (dense * torch.exp(self.dense_log_scale)).to(optical.dtype)

        outputs = self.model(pixel_values=optical, input_boxes=input_boxes,
                             input_masks=input_masks, multimask_output=False)
        scores = getattr(outputs, "iou_scores",
                         getattr(outputs, "iou_predictions",
                                 getattr(outputs, "pred_ious", None)))
        # `heat` esce anche verso il chiamante: serve alla deep supervision
        # (boxhead_lambda_map). E' None per la testa "gap".
        return outputs.pred_masks, scores, box_norm, obj_logit, heat


# ============================================================
# EMA DEI PESI (Polyak)
# ============================================================

class ModelEMA:
    """
    Media esponenziale dei pesi: ema = decay*ema + (1-decay)*model.

    IMPLEMENTAZIONE A SHADOW + SWAP, NON deepcopy del modulo. Il motivo e' un
    bug verificato: LENTE aggancia gli Adapter al vision encoder con forward
    hook, e ogni hook e' una chiusura che cattura l'Adapter originale. Per
    copy.deepcopy le funzioni sono atomiche (non vengono copiate), quindi gli
    hook della COPIA continuano a chiamare gli Adapter del modello VIVO:
    misurato, l'adapter catturato dall'hook di ema.module apparteneva a `model`
    e non a `ema.module`. Risultato: la validazione EMA girava con gli adapter
    live e tutto il resto EMA — selezione del checkpoint corrotta in silenzio.

    Qui invece i pesi EMA vivono in un dict di tensori (`shadow`); per la
    validazione si CARICANO nel modello vivo (swap_in) e poi si ripristinano
    (swap_out). C'e' un solo modulo, quindi gli hook puntano sempre agli
    adapter giusti. load_state_dict copia in place, percio' l'optimizer
    (che tiene i riferimenti ai parametri) non si accorge di nulla.

    Altri due dettagli che, sbagliati, la rendono silenziosamente inutile:
    * i BUFFER si COPIANO, non si mediano: un buffer e' una statistica gia'
      aggregata (running stats della BatchNorm della DSDF), mediarla una
      seconda volta la sfoca senza motivo.
    * warmup del decay: all'inizio lo shadow e' ancora l'inizializzazione;
      con decay fisso a 0.999 servono migliaia di step per dimenticarla.
    """

    def __init__(self, model, decay=0.999, warmup=1000):
        self.decay = decay
        self.warmup = max(int(warmup), 1)
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.buffer_keys = {k for k, _ in model.named_buffers()}

    def _current_decay(self):
        # sale da 0 verso `decay` con costante di tempo `warmup`
        import math
        return self.decay * (1.0 - math.exp(-self.updates / self.warmup))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self._current_decay()
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if k in self.buffer_keys or not s.dtype.is_floating_point:
                s.copy_(v)                            # buffer e interi: copia
            else:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model):
        """Carica i pesi EMA nel modello vivo. Ritorna il backup dei pesi live."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    @torch.no_grad()
    def swap_out(self, model, backup):
        """Ripristina i pesi live salvati da swap_in."""
        model.load_state_dict(backup)


# ============================================================
# LOSS
# ============================================================

def map_supervision(heat, masks, pos_weight=1.0):
    """
    Deep supervision sulla heatmap: BCE cella per cella contro la maschera GT
    ridotta alla griglia della mappa.

    Perche' serve. Senza questo termine la heatmap impara solo per vie
    indirette: la L1 sul box vincola 4 numeri ricavati da lei (centroide e
    dispersione), e il gradiente della segmentazione le arriva smorzato dopo
    essere passato per mask_embed e il decoder. Nessuno dei due le dice mai
    QUALE cella accendere. Qui il bersaglio e' esplicito e per cella: da 4
    numeri di feedback si passa a G*G.

    Il bersaglio e' MORBIDO: avg_pool della maschera, cioe' la frazione di
    pixel-frana in ogni cella. Con max_pool una sola cella accesa in un blocco
    da 64 pixel varrebbe 1.0 come un blocco pieno, e la mappa imparerebbe a
    dilatare. La frazione conserva l'intensita'.

    E' anche il termine che rende la mappa capace di rappresentare frane
    MULTIPLE: due macchie separate diventano due zone calde, senza passare da
    nessun rettangolo.
    """
    g = heat.shape[-1]
    m = masks.float().unsqueeze(1)                       # (B,1,H,W)
    k = max(m.shape[-1] // g, 1)
    target = F.avg_pool2d(m, kernel_size=k, stride=k)
    if target.shape[-2:] != heat.shape[-2:]:             # H non multiplo di g
        target = F.interpolate(target, heat.shape[-2:], mode="area")
    pw = torch.tensor([pos_weight], device=heat.device, dtype=heat.dtype)
    return F.binary_cross_entropy_with_logits(heat, target.to(heat.dtype), pos_weight=pw)


def boxhead_criterion(preds, masks, box_pred, box_gt, obj_logit, has_obj,
                      pos_weight=1.0, lambda_box=5.0, lambda_obj=0.5,
                      heat=None, lambda_map=0.0):
    """
    Segmentazione + regressione del box + objectness.

    La L1 sul box e' calcolata SOLO sui positivi: su una patch senza frane il
    box target non esiste (LENTE_utils._bbox_from_mask restituisce il fittizio
    [0,0,10,10]) e regredire verso quello insegnerebbe al modello a piazzare un
    riquadro in un angolo ogni volta che non vede niente. Il "non c'e' niente"
    lo dice la testa di objectness, che e' il posto giusto.

    lambda_box=5.0: la L1 su coordinate normalizzate in [0,1] produce valori
    ~0.1-0.3, un ordine di grandezza sotto la loss di segmentazione; senza un
    peso esplicito la testa imparerebbe troppo lentamente per essere utile.
    """
    seg = criterion(preds, masks, pos_weight=pos_weight)

    obj = F.binary_cross_entropy_with_logits(obj_logit, has_obj.float())

    if has_obj.any():
        box = F.l1_loss(box_pred[has_obj], box_gt[has_obj])
    else:
        box = preds.sum() * 0.0          # tensore a zero che resta nel grafo

    if lambda_map > 0.0 and heat is not None:
        mp = map_supervision(heat, masks, pos_weight=pos_weight)
    else:
        mp = preds.sum() * 0.0           # zero che resta nel grafo

    total = seg + lambda_box * box + lambda_obj * obj + lambda_map * mp
    return total, seg.detach(), box.detach(), obj.detach(), mp.detach()


# ============================================================
# TRAINING
# ============================================================

# jitter_boxes vive in LENTE_utils: la usano sia lo scheduled sampling qui sia
# il training gt_box in LENTE_train (per allinearsi alla qualita' dei box YOLO).


def _prep_batch(batch, target, device):
    imgs  = batch["image"].to(device)
    masks = batch["mask"].to(device)
    boxes = batch["boxes"].to(device)
    oh, ow = imgs.shape[-2], imgs.shape[-1]
    imgs  = F.interpolate(imgs, (target, target), mode="bilinear", align_corners=False)
    masks = F.interpolate(masks.unsqueeze(1).float(), (target, target),
                          mode="nearest").squeeze(1).long()
    boxes = scale_boxes(boxes, oh, ow, target)
    # target normalizzato per la L1, e flag di presenza frana per l'objectness
    box_gt  = boxes / float(target - 1)
    has_obj = masks.flatten(1).sum(dim=1) > 0
    return imgs, masks, boxes, box_gt, has_obj


@torch.no_grad()
def _validate(model, loader, target, device, pos_weight, cfg):
    """Valutazione SEMPRE con il box predetto: mai use_gt_prompt qui."""
    model.eval()
    v_loss = v_iou = v_f1 = v_boxl1 = 0.0
    v_area_pred = v_area_gt = 0.0
    obj_ok = obj_tot = n_box_batches = 0
    for batch in tqdm(loader, desc="[VAL]", leave=False):
        imgs, masks, boxes, box_gt, has_obj = _prep_batch(batch, target, device)
        with torch.amp.autocast(device_type=device):
            all_preds, scores, box_pred, obj_logit, heat = model(imgs, use_gt_prompt=False)
            preds = get_best_mask(all_preds, scores)
            loss, _, _, _, _ = boxhead_criterion(
                preds, masks, box_pred, box_gt, obj_logit, has_obj,
                pos_weight=pos_weight,
                lambda_box=cfg["boxhead_lambda_box"],
                lambda_obj=cfg["boxhead_lambda_obj"],
                heat=heat, lambda_map=cfg["boxhead_lambda_map"])
        v_loss += loss.item()
        iou_fg, _, _, f1, _, _ = calculate_metrics(preds.float(), masks)
        v_iou += iou_fg; v_f1 += f1
        if has_obj.any():
            bp, bg = box_pred[has_obj].float(), box_gt[has_obj]
            v_boxl1 += F.l1_loss(bp, bg).item()
            # AREA del box, non solo la distanza: la L1 da sola non distingue
            # "box a tutta immagine" da "box giusto ma spostato". Con l'area
            # accanto, area_pred ~1.0 dice che la testa non sta dimensionando
            # (clamp o heatmap troppo piatta); area_pred ~ area_gt con L1 alta
            # dice che dimensiona ma non localizza.
            v_area_pred += ((bp[:, 2] - bp[:, 0]) * (bp[:, 3] - bp[:, 1])).clamp_min(0).mean().item()
            v_area_gt   += ((bg[:, 2] - bg[:, 0]) * (bg[:, 3] - bg[:, 1])).clamp_min(0).mean().item()
            n_box_batches += 1        # la media della L1 e' solo sui batch che
                                      # hanno positivi, non su tutti (i batch di
                                      # soli negativi la diluirebbero verso 0)
        obj_ok  += ((obj_logit > 0) == has_obj).sum().item()
        obj_tot += has_obj.numel()
    n  = max(len(loader), 1)
    nb = max(n_box_batches, 1)
    return (v_loss / n, v_iou / n, v_f1 / n, v_boxl1 / nb,
            obj_ok / max(obj_tot, 1), v_area_pred / nb, v_area_gt / nb)


def _rotate(path):
    """Se `path` esiste lo rinomina in <base>_bak<N><ext> (primo N libero).
    Niente viene mai sovrascritto o cancellato: ogni run fresco archivia i
    prodotti del precedente invece di clobberarli."""
    if not os.path.exists(path):
        return None
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{base}_bak{n}{ext}"):
        n += 1
    dst = f"{base}_bak{n}{ext}"
    os.replace(path, dst)
    return dst


def _last_path(ckpt_name):
    """best_lente_l4s_boxhead.pth -> last_lente_l4s_boxhead.pth (stato di resume)."""
    d, b = os.path.split(ckpt_name)
    b = "last_" + (b[len("best_"):] if b.startswith("best_") else b)
    return os.path.join(d, b)


def _save_resume_state(path, model, optimizer, scheduler, ema,
                       epoch, best_iou, patience_counter):
    """
    Stato completo per riprendere il training, scritto a FINE di ogni epoca.

    E' un file distinto dal checkpoint "best" per una ragione precisa: con
    l'EMA attiva il best contiene i pesi MEDIATI (e' il deliverable per il
    test), mentre per riprendere il training servono i pesi VIVI piu' lo
    shadow EMA con il suo contatore. Riprendere dal best ripartirebbe dalla
    media, perdendo la traiettoria dell'ottimizzazione.
    """
    state = {
        "epoch": epoch,
        "best_iou": best_iou,
        "patience_counter": patience_counter,
        "model_state_dict": model.state_dict(),        # pesi VIVI (post swap_out)
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if ema is not None:
        state["ema_shadow"]  = ema.shadow
        state["ema_updates"] = ema.updates
    torch.save(state, path)


@torch.no_grad()
def test_boxhead(cfg, use_ema=False):
    """
    Test dei modi boxhead sul TEST set, sempre col box PREDETTO.

    Perche' serve una funzione dedicata invece di riusare LENTE_test.test():
    quella costruisce un `LENTE` normale (senza testa di box) e carica con
    `strict=False`. Dandogli un checkpoint boxhead, tutte le chiavi `box_head.*`
    verrebbero scartate IN SILENZIO: nessun errore, un numero plausibile, e un
    modello monco valutato in un regime diverso da quello con cui e' stato
    addestrato. Qui si carica con strict=True, cosi' un checkpoint sbagliato
    fallisce subito invece di produrre un risultato falso.

    Le metriche sono calcolate come in LENTE_test (macro per campione, soglia
    0.5), piu' `box L1` e `obj acc` che valgono solo per questa architettura.
    """
    device = cfg["device"]
    target = cfg["target_size"]
    ckpt_name = cfg["checkpoint_name"]

    if not os.path.exists(ckpt_name):
        raise SystemExit(
            f"Checkpoint non trovato: {ckpt_name}\n"
            f"I modi boxhead derivano il nome dal dataset e dalla variante "
            f"(best_lente_<dataset>_boxhead[_ema].pth) e lo cercano nella "
            f"cartella da cui lanci python. Se l'hai archiviato altrove, "
            f"riportalo qui o passa --checkpoint con il percorso giusto.")

    full_ds = build_dataset(cfg)
    _, _, test_subset, pos_weight = build_train_val_test_subsets(full_ds, cfg)
    test_loader = DataLoader(TransformSubset(test_subset, A.Compose([ToTensorV2()])),
                             batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    model = LENTEBoxHead(cfg["sam_ckpt_path"], n_optical=cfg["n_optical"],
                         adapter_dim=cfg["adapter_dim"],
                         head_width=cfg["boxhead_width"],
                         head_work_size=cfg["boxhead_work_size"],
                         head_type=cfg["boxhead_type"],
                         head_terrain_prior=cfg["boxhead_terrain_prior"],
                         prompt_mode=cfg["boxhead_prompt"],
                         image_size=cfg["target_size"],
                         adapter_where=cfg.get("adapter_where", "attn")).to(device)

    print(f"Caricamento da {ckpt_name} ...")
    ckpt = torch.load(ckpt_name, map_location=device)
    # strict=True: se l'architettura nel config non e' quella con cui il
    # checkpoint e' stato salvato (es. boxhead_type diverso), si vuole un
    # errore, non un caricamento parziale silenzioso.
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  epoca {ckpt.get('epoch', '?')} | best Val IoU salvata {ckpt.get('best_iou', float('nan')):.4f}")

    t_loss, t_iou, t_miou, t_f1 = 0.0, 0.0, 0.0, 0.0
    t_boxl1, n_box = 0.0, 0
    obj_ok = obj_tot = 0
    model.eval()
    for batch in tqdm(test_loader, desc="TESTING (box predetto)"):
        imgs, masks, boxes, box_gt, has_obj = _prep_batch(batch, target, device)
        with torch.amp.autocast(device_type=device):
            all_preds, scores, box_pred, obj_logit, heat = model(imgs, use_gt_prompt=False)
            preds = get_best_mask(all_preds, scores)
            loss, _, _, _, _ = boxhead_criterion(
                preds, masks, box_pred, box_gt, obj_logit, has_obj,
                pos_weight=pos_weight,
                lambda_box=cfg["boxhead_lambda_box"],
                lambda_obj=cfg["boxhead_lambda_obj"],
                heat=heat, lambda_map=cfg["boxhead_lambda_map"])
        t_loss += loss.item()
        iou_fg, _, miou, f1, _, _ = calculate_metrics(preds.float(), masks)
        t_iou += iou_fg; t_miou += miou; t_f1 += f1
        if has_obj.any():
            t_boxl1 += F.l1_loss(box_pred[has_obj].float(), box_gt[has_obj]).item()
            n_box += 1
        obj_ok  += ((obj_logit > 0) == has_obj).sum().item()
        obj_tot += has_obj.numel()

    n = max(len(test_loader), 1)
    print("\n" + "=" * 46)
    print(f"TEST BOXHEAD ({cfg['dataset_name']}) — testa {cfg['boxhead_type']}"
          + (" + EMA" if use_ema else ""))
    print("=" * 46)
    print(f"Test Loss : {t_loss / n:.4f}")
    print(f"Test IoU  : {t_iou / n:.4f}")
    print(f"Test mIoU : {t_miou / n:.4f}")
    print(f"Test F1   : {t_f1 / n:.4f}")
    print(f"box L1    : {t_boxl1 / max(n_box, 1):.4f}   (solo positivi)")
    print(f"obj acc   : {obj_ok / max(obj_tot, 1):.4f}")
    print("=" * 46)
    return {"loss": t_loss / n, "iou": t_iou / n, "miou": t_miou / n,
            "f1": t_f1 / n, "box_l1": t_boxl1 / max(n_box, 1),
            "obj_acc": obj_ok / max(obj_tot, 1)}


def train_boxhead(cfg, use_ema=False):
    device = cfg["device"]
    target = cfg["target_size"]

    ckpt_name = cfg["checkpoint_name"]
    hist_name = cfg["history_file"]
    last_name = _last_path(ckpt_name)

    full_ds = build_dataset(cfg)
    train_subset, val_subset, _, pos_weight = build_train_val_test_subsets(full_ds, cfg)

    offline_aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    train_trans = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                             A.RandomRotate90(p=0.5), ToTensorV2()])
    val_trans   = A.Compose([ToTensorV2()])

    pos_set = set(full_ds.positive_indices)
    train_pos = [i for i in train_subset.indices if i in pos_set]
    full_ds.add_augmented_samples(indices=train_pos, transform=offline_aug,
                                  n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    aug_start = len(full_ds.data) - len(train_pos) * cfg["aug_copies"]
    train_subset = Subset(full_ds, list(train_subset.indices) + list(range(aug_start, len(full_ds.data))))

    train_loader = DataLoader(TransformSubset(train_subset, train_trans),
                              batch_size=cfg["batch_size"], shuffle=True, pin_memory=True)
    val_loader   = DataLoader(TransformSubset(val_subset, val_trans),
                              batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    model = LENTEBoxHead(cfg["sam_ckpt_path"], n_optical=cfg["n_optical"],
                         adapter_dim=cfg["adapter_dim"],
                         head_width=cfg["boxhead_width"],
                         head_work_size=cfg["boxhead_work_size"],
                         head_type=cfg["boxhead_type"],
                         head_terrain_prior=cfg["boxhead_terrain_prior"],
                         prompt_mode=cfg["boxhead_prompt"],
                         image_size=cfg["target_size"],
                         adapter_where=cfg.get("adapter_where", "attn")).to(device)

    n_head = sum(p.numel() for p in model.box_head.parameters())
    n_tr   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLENTE + box appreso su '{cfg['dataset_name']}' | EMA: {'SI' if use_ema else 'NO'}")
    print(f"  testa: {cfg['boxhead_type']}"
          + (" + prior di terreno" if cfg["boxhead_terrain_prior"] else "")
          + f" | prompt: {cfg['boxhead_prompt']}"
          + f" | {n_head/1e3:.1f}k param | trainabili totali: {n_tr/1e6:.2f}M")

    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=cfg["ema_decay"], warmup=cfg["ema_warmup"])
        print(f"  EMA attiva: decay={cfg['ema_decay']} warmup={cfg['ema_warmup']} step")

    # Due learning rate distinti. `learning_rate` (1e-5) e' tarato su adapter e
    # mask_decoder, che partono da pesi PRETRAINED e vanno solo ritoccati. La
    # testa di box invece e' una CNN inizializzata da ZERO: a 1e-5 si muove
    # 10-100 volte troppo lentamente. Misurato sul primo run: box L1 da 0.2196 a
    # 0.1902 in 20 epoche, cioe' praticamente ferma. Non era il peso della loss
    # (il termine box dominava gia' il totale) — era il passo di aggiornamento.
    # dense_log_scale vive su LENTEBoxHead, non dentro box_head: senza questa
    # riga finirebbe nel gruppo lento a 1e-5 — ma e' un parametro da zero come
    # la testa, e va alla stessa velocita'.
    head_params = list(model.box_head.parameters()) + [model.dense_log_scale]
    head_ids    = {id(p) for p in head_params}
    rest_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in head_ids]
    optimizer = optim.AdamW(
        [{"params": rest_params, "lr": cfg["learning_rate"]},
         {"params": head_params, "lr": cfg["boxhead_lr"]}],
        weight_decay=cfg["weight_decay"])
    print(f"  LR: backbone/adapter {cfg['learning_rate']:.0e} | testa di box {cfg['boxhead_lr']:.0e}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(device=device)

    tf_epochs = cfg["boxhead_teacher_forcing_epochs"]
    if cfg["boxhead_prompt"] == "dense" and tf_epochs > 0:
        # Senza canale sparse non c'e' nessun box da forzare, e l'analogo denso
        # (maschera GT come prompt) sarebbe degenere. Si parte gia' autonomi.
        print(f"  scheduled sampling disattivato: con prompt 'dense' non c'e' "
              f"box da forzare (era {tf_epochs} epoche)")
        tf_epochs = 0
    history, best_iou, patience_counter, start_epoch = [], 0.0, 0, 0

    # ── resume ESPLICITO (mai silenzioso) o archiviazione del run precedente ──
    if cfg.get("resume"):
        if os.path.exists(last_name):
            st = torch.load(last_name, map_location=device)
            model.load_state_dict(st["model_state_dict"])
            optimizer.load_state_dict(st["optimizer_state_dict"])
            scheduler.load_state_dict(st["scheduler_state_dict"])
            if ema is not None:
                if "ema_shadow" not in st:
                    raise SystemExit(
                        f"{last_name} non contiene lo stato EMA: e' di un run "
                        f"senza EMA. Riprendere mescolando i due regimi "
                        f"produrrebbe una media senza senso.")
                ema.shadow  = {k: v.to(device) for k, v in st["ema_shadow"].items()}
                ema.updates = st["ema_updates"]
            best_iou         = st["best_iou"]
            patience_counter = st["patience_counter"]
            start_epoch      = st["epoch"] + 1
            # la history riparte troncata all'epoca del resume: se il run e'
            # morto fra la scrittura della history e quella dello stato, le
            # epoche orfane vengono scartate invece di duplicarsi.
            if os.path.exists(hist_name):
                with open(hist_name) as f:
                    history = json.load(f)[:start_epoch]
            print(f"  RESUME da {last_name}: epoca {start_epoch}, "
                  f"best IoU {best_iou:.4f}, patience {patience_counter}")
        else:
            print(f"  --resume richiesto ma {last_name} non esiste: parto da zero.")
    else:
        # run fresco: i prodotti del run precedente vengono ARCHIVIATI (_bakN),
        # mai sovrascritti. Lo stato di resume invece e' scratch superato dalla
        # decisione di ripartire: si elimina con avviso.
        for p in (hist_name, ckpt_name):
            dst = _rotate(p)
            if dst:
                print(f"  Run precedente archiviato: {p} -> {dst}")
        if os.path.exists(last_name):
            os.remove(last_name)
            print(f"  Stato di resume obsoleto rimosso: {last_name}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        # Scheduled sampling: probabilita' di usare il box GT come prompt, da 1
        # a 0 in `tf_epochs` epoche. Serve solo a far partire il decoder; se
        # restasse accesa il modello imparerebbe a dipendere da un input che a
        # test time non ha.
        p_gt = max(0.0, 1.0 - epoch / max(tf_epochs, 1)) if tf_epochs > 0 else 0.0

        t_loss = t_seg = t_box = t_obj = t_iou = t_map = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['epochs']} [TRAIN]")
        for batch in pbar:
            imgs, masks, boxes, box_gt, has_obj = _prep_batch(batch, target, device)
            use_gt = (torch.rand(1).item() < p_gt)
            # Il prompt GT va sporcato; il TARGET della L1 (box_gt) resta pulito.
            prompt_boxes = (jitter_boxes(boxes, target, cfg["boxhead_gt_jitter"])
                            if use_gt else boxes)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                all_preds, scores, box_pred, obj_logit, heat = model(
                    imgs, gt_boxes=prompt_boxes, use_gt_prompt=use_gt,
                    gt_prompt_mask=has_obj)
                preds = get_best_mask(all_preds, scores)
                loss, seg, boxl, objl, mapl = boxhead_criterion(
                    preds, masks, box_pred, box_gt, obj_logit, has_obj,
                    pos_weight=pos_weight,
                    lambda_box=cfg["boxhead_lambda_box"],
                    lambda_obj=cfg["boxhead_lambda_obj"],
                    heat=heat, lambda_map=cfg["boxhead_lambda_map"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            iou_fg, _, _, _, _, _ = calculate_metrics(preds.detach().float(), masks)
            t_loss += loss.item(); t_seg += seg.item()
            t_box  += boxl.item(); t_obj += objl.item(); t_iou += iou_fg
            t_map  += mapl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", seg=f"{seg.item():.4f}",
                             box=f"{boxl.item():.4f}", map=f"{mapl.item():.4f}",
                             iou=f"{iou_fg:.4f}", p_gt=f"{p_gt:.2f}")

        # Valutazione: se l'EMA e' attiva, i pesi shadow vengono CARICATI nel
        # modello vivo (swap_in) e ripristinati subito dopo (swap_out). Un solo
        # modulo: gli hook degli Adapter restano corretti (vedi ModelEMA).
        backup = ema.swap_in(model) if ema is not None else None
        v_loss, v_iou, v_f1, v_boxl1, obj_acc, area_pred, area_gt = _validate(
            model, val_loader, target, device, pos_weight, cfg)

        improved = v_iou > best_iou
        if improved:
            best_iou = v_iou
            # con lo swap attivo qui dentro ci sono i pesi EMA: e' quello che
            # vogliamo salvare ("si valuta e si salva la media")
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, ckpt_name)

        if backup is not None:
            ema.swap_out(model, backup)      # ripristina i pesi live PRIMA di
                                             # riprendere il training
        scheduler.step(v_iou)

        nb = max(len(train_loader), 1)
        rec = {
            "epoch": epoch + 1,
            "p_gt_prompt": p_gt,
            "train_loss": t_loss / nb, "train_seg": t_seg / nb,
            "train_box_l1": t_box / nb, "train_obj": t_obj / nb,
            "train_map": t_map / nb, "train_iou": t_iou / nb,
            "val_loss": v_loss, "val_iou": v_iou, "val_f1": v_f1,
            "val_box_l1": v_boxl1, "val_obj_acc": obj_acc,
            "val_box_area_pred": area_pred, "val_box_area_gt": area_gt,
            "ema": bool(ema is not None),
        }
        history.append(rec)
        with open(hist_name, "w") as f:
            json.dump(history, f, indent=2)

        # Divario train-val: e' il segnale che dice se la CAPACITA' e' il collo
        # di bottiglia. Finora e' sempre stato negativo (val sopra train) anche
        # dopo 70 epoche, cioe' il modello non riusciva nemmeno a sfruttare i
        # dati che aveva. Se alzando adapter_dim diventa positivo, la capacita'
        # era il freno e ha senso salire ancora; se resta negativo, non lo era.
        # NB conservativo: la IoU di train e' su dati AUMENTATI, quindi piu'
        # difficili — un divario nullo e' gia' indizio di underfitting.
        gap = t_iou / nb - v_iou
        print(f"EPOCA {epoch+1} | Val IoU: {v_iou:.4f} | F1: {v_f1:.4f} | "
              f"box L1: {v_boxl1:.4f} | area {area_pred:.3f}/{area_gt:.3f} | "
              f"obj acc: {obj_acc:.3f} | gap {gap:+.3f} | p_gt={p_gt:.2f}")

        if improved:
            print(f"  Record IoU: {best_iou:.4f} — checkpoint salvato in {ckpt_name}")
            patience_counter = 0
        else:
            patience_counter += 1

        # stato di resume a fine epoca, con i pesi VIVI (lo swap_out e' gia'
        # avvenuto sopra) — scritto anche sull'ultima epoca prima del break
        _save_resume_state(last_name, model, optimizer, scheduler, ema,
                           epoch, best_iou, patience_counter)

        # Non fermarsi finche' lo scheduled sampling e' ancora acceso: il
        # modello sta ancora passando dal box GT a quello predetto, e in
        # quella transizione la IoU legittimamente non migliora.
        if (not improved and patience_counter >= cfg["patience"]
                and epoch >= tf_epochs):
            print(f"\nEARLY STOPPING — Miglior Val IoU: {best_iou:.4f}")
            break

    return best_iou


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LENTE con prompt box appreso")
    ap.add_argument("-d", "--dataset", default=None)
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--ema", action="store_true", help="attiva l'EMA dei pesi")
    ap.add_argument("--resume", action="store_true",
                    help="riprende dallo stato last_*.pth invece di partire da zero")
    a = ap.parse_args()
    from LENTE_run import prepare_boxhead_cfg
    _cfg = prepare_boxhead_cfg(load_config(a.config, dataset=a.dataset), a.ema)
    _cfg["resume"] = a.resume
    train_boxhead(_cfg, use_ema=a.ema)

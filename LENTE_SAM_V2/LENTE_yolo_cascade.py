#!/usr/bin/env python
"""
LENTE_yolo_cascade.py — detector YOLO davanti a SAM2.
=====================================================
Cascata detect-then-segment: YOLO propone i bounding box, LENTE (SAM2) segmenta
dentro ciascuno, le maschere si uniscono.

PERCHE' QUESTA STRADA. Il limite contro cui abbiamo sbattuto in tre esperimenti
diversi (testa GAP, testa heatmap, prompt denso) e' che UN rettangolo non puo'
descrivere tre frane in tre angoli diversi: su L4S `_bbox_from_mask` prende il
box dell'UNIONE e il p90 della sua area e' il 96.9% dell'immagine. La testa di
box interna si e' sempre fermata a box troppo larghi (area 0.52-0.75 contro
0.391 del vero).
Un detector risolve il problema per costruzione: emette N box, uno per istanza.

LEGITTIMITA'. Il GT box resta SUPERVISIONE (YOLO si addestra sui box veri, come
ogni detector) e non INPUT a inferenza: a test time i box li produce YOLO. E'
la stessa distinzione che regge tutto il progetto dal giorno del leak.

LA GERARCHIA DEI RISULTATI che ne esce, sullo stesso split:

    prompt-free                 IoU 0.5942   pavimento (nessun prompt)
    denso + EMA + lambda_map    IoU 0.6066   meglio attuale
    YOLO -> SAM2                IoU   ?      questa cascata
    gt_box (oracolo)            IoU 0.7501   soffitto: box perfetti

Il divario fra la cascata e 0.7501 misura ESATTAMENTE quanto e' buono YOLO, e
diventa la metrica di quanto vale la pena investire sul detector.

DUE AVVERTENZE, da tenere presenti leggendo i risultati:

  1. Il RECALL DELLA CASCATA E' LIMITATO DAL DETECTOR. Se YOLO non trova una
     frana, SAM2 non la vede proprio: nessun prompt, nessuna maschera. Il
     segmentatore non puo' recuperare cio' che il detector ha perso.
  2. Sui negativi YOLO non emette box, quindi la maschera esce vuota e quella
     patch vale IoU 1.0 con la metrica attuale. Parte del guadagno verra' da
     li' — cioe' dal comportamento da CLASSIFICATORE, non da segmentazione
     migliore. Va riportato separando i due contributi (vedi `evaluate` che
     stampa la decomposizione positivi/negativi).

USO  (tutti i parametri stanno in config.yaml sotto le chiavi yolo_*;
      i flag da riga di comando sono override opzionali)

    pip install ultralytics

    # 1) esporta il dataset in formato YOLO, RISPETTANDO lo split di LENTE
    python LENTE_yolo_cascade.py export -d l4s

    # 2) addestra il detector
    python LENTE_yolo_cascade.py train -d l4s

    # 3) misura quanto sbaglia YOLO -> da qui esce il valore di prompt_jitter
    #    ATTENZIONE: ultralytics NON sovrascrive, incrementa la cartella.
    #    Controlla con `ls -d runs/detect/*/` quale sia l'ultima.
    python LENTE_yolo_cascade.py calibrate -d l4s --yolo runs/detect/train/weights/best.pt

    # 4) riaddestra LENTE per i box di YOLO: in config.yaml metti
    #      prompt_source: "gt_box"    e    prompt_jitter: <valore dal punto 3>
    python LENTE_run.py -d l4s -m train

    # 5) valuta la cascata sul TEST set
    python LENTE_yolo_cascade.py eval -d l4s --yolo .../best.pt --lente best_lente_l4s_gtbox.pth

"""

import argparse
import os

# Backend matplotlib NON interattivo, prima che ultralytics importi pyplot.
# A fine training ultralytics disegna la curva precisione-recupero e poi
# ripristina il backend di partenza: su un server headless quello e'
# "tkagg", che non esiste, e il ripristino solleva ImportError DOPO che
# l'addestramento e' gia' finito e i pesi sono salvati. Si perde solo il
# grafico, ma il traceback fa sembrare fallito tutto il run.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets,
    build_model, get_best_mask, scale_boxes, _bbox_from_mask,
)


# ============================================================
# 1. MASCHERA -> BOX  (componenti connesse)
# ============================================================

def masks_to_boxes(mask, min_area=4):
    """
    Da una maschera binaria alla lista dei box delle sue componenti connesse.

    E' il punto in cui questa strada si separa da tutto quello provato finora:
    `LENTE_utils._bbox_from_mask` restituisce UN box, quello dell'unione di
    tutte le frane. Qui ogni componente ha il suo, che e' cio' che rende il
    bersaglio rappresentabile.

    min_area scarta le componenti minuscole: su patch 128x128 una macchia da
    pochi pixel e' piu' probabilmente rumore di annotazione che una frana, e
    come bersaglio di detection sarebbe solo confusione.

    Ritorna una lista di (x0, y0, x1, y1) in pixel, estremi inclusi.
    """
    lab, n = ndimage.label(np.asarray(mask) > 0)
    boxes = []
    for sl_y, sl_x in ndimage.find_objects(lab):
        h = sl_y.stop - sl_y.start
        w = sl_x.stop - sl_x.start
        if h * w < min_area:
            continue
        boxes.append((float(sl_x.start), float(sl_y.start),
                      float(sl_x.stop - 1), float(sl_y.stop - 1)))
    return boxes


def gt_boxes_for(mask, cfg):
    """
    I box bersaglio per il detector, secondo `yolo_box_mode`.

    "union" (default) — UN box per patch, quello dell'unione di tutte le frane.
        E' esattamente il prompt con cui LENTE ha ottenuto 0.7501 (l'oracolo),
        quindi il checkpoint gt_box esistente funziona senza riaddestramento e
        quel numero diventa il SOFFITTO MISURATO della cascata.

        La divisione dei ruoli e' questa: il detector dice "ci sono frane, piu' o
        meno qui"; SAM2 dice "esattamente questi pixel". Un box che e' un PROMPT
        non deve delimitare l'oggetto — deve indicare la regione. Che SAM2 sappia
        segmentare piu' frane dentro un riquadro largo non e' un'ipotesi: e'
        precisamente cio' che 0.7501 misura.

        In piu' sparisce il problema degli oggetti piccoli: invece di macchie
        dell'1-2% dell'immagine, il bersaglio e' una regione del ~42%.

    "component" — N box, uno per componente connessa sopra `yolo_min_area`.
        Bersagli piu' stretti, ma richiedono di RIADDESTRARE LENTE: i box
        componente valgono ~1-2% dell'immagine contro il ~42% dell'unione, cioe'
        un regime di prompt che il segmentatore non ha mai visto (misurato: 32x
        di differenza di scala). Tenuta per confronto.
    """
    if cfg.get("yolo_box_mode", "union") == "component":
        return masks_to_boxes(mask, min_area=cfg.get("yolo_min_area", 4))
    m = np.asarray(mask)
    if int(m.sum()) < cfg.get("yolo_min_area", 4):
        return []                      # patch praticamente vuota: nessun box
    b = _bbox_from_mask(torch.from_numpy(m).long()).tolist()
    return [(b[0], b[1], b[2], b[3])]


def _yolo_bands(cfg):
    return list(cfg.get("yolo_bands", cfg.get("rgb_channels", [0, 1, 2])))


def _yolo_nch(cfg):
    """Quanti canali riceve il detector. 3 = percorso storico (PNG + BGR)."""
    return len(_yolo_bands(cfg))


def _make_stack(img_np, cfg):
    """
    Immagine a N canali per il detector, uint8, (H, W, N).

    `img_np` e' lo stack COMPLETO (C,H,W): 12 bande ottiche + slope + DEM su
    L4S. Quindi `yolo_bands` puo' pescare anche fra i canali di TERRENO, non
    solo fra le ottiche — es. [7, 11, 12] = B8, B12, slope.

    Usa `yolo_bands`, NON `rgb_channels`: il primo sceglie i canali che rendono
    le frane VISIBILI, il secondo quelli che rendono l'immagine gradevole. Sono
    obiettivi diversi e su L4S danno risposte diverse (vedi config.yaml).

    Lo stretch e' PER-CANALE, mai globale. Con 3 bande ottiche serviva perche'
    il NIR e' molto piu' luminoso del blu; mescolando riflettanze e pendenze
    diventa obbligatorio, perche' le scale non hanno nemmeno la stessa unita'.
    Per la stessa ragione con canali non ottici `rgb_stretch` DEVE essere
    "minmax": il ramo "clip" taglia a [0,1] e azzererebbe uno slope in gradi.
    """
    ch = _yolo_bands(cfg)
    st = img_np[ch, :, :].transpose(1, 2, 0).astype(np.float32)
    n = st.shape[-1]
    if cfg.get("rgb_stretch", "clip") == "minmax":
        flat = st.reshape(-1, n)
        lo, hi = flat.min(0), flat.max(0)
        st = (st - lo) / (hi - lo + 1e-8)
    else:
        st = np.clip(st, 0, 1)
    return (st * 255).astype(np.uint8)


def _make_rgb(img_np, cfg):
    """Alias storico: identico a _make_stack quando i canali sono 3."""
    return _make_stack(img_np, cfg)


def _for_yolo(img_np, cfg):
    """
    Lo stack nel formato che `predict` si aspetta.

    A 3 canali serve RGB -> BGR, e non e' cosmetico: ultralytics interpreta gli
    array numpy come BGR (eredita la convenzione di cv2). In export i PNG
    vengono scritti con `cv2.imwrite` dopo RGB2BGR, quindi in training YOLO
    vede i canali giusti; passando RGB a `predict` si avrebbero rosso e blu
    scambiati rispetto all'addestramento — un errore che non da' nessun
    messaggio, solo detection peggiori.

    Oltre i 3 canali la convenzione BGR non esiste: si passa lo stack
    nell'ordine di `yolo_bands`, lo stesso in cui l'export lo ha scritto.
    """
    st = _make_stack(img_np, cfg)
    if st.shape[-1] != 3:
        return st
    import cv2
    return cv2.cvtColor(st, cv2.COLOR_RGB2BGR)


# ============================================================
# 1b. YOLO CONGELATO DENTRO IL LOOP DI TRAINING DI LENTE
# ============================================================

class YoloBoxProvider:
    """
    Detector congelato che propone i box mentre LENTE si addestra.

    PERCHE' DENTRO IL LOOP e non precalcolato. I box devono corrispondere
    all'immagine che LENTE vede davvero, e quella e' gia' AUMENTATA (flip,
    rotazioni). Precalcolando sulle immagini originali i box non seguirebbero
    l'augmentation — e' lo stesso difetto che affligge `TransformSubset`, che
    trasforma image e mask lasciando il box invariato. Facendo girare YOLO
    sull'immagine aumentata la corrispondenza e' garantita per costruzione, e
    funziona anche sulle copie aumentate offline.

    PERCHE' CONGELATO. YOLO e' gia' addestrato e va in `eval` sotto `no_grad`:
    e' un fornitore di indizi, non un pezzo da ottimizzare. Il gradiente della
    segmentazione non lo raggiunge (l'NMS non e' derivabile comunque).

    Il box e' un INDIZIO, non la risposta: viene da un modello, e' disponibile
    a inferenza, e non guarda mai la maschera del campione.
    """

    def __init__(self, weights, cfg, conf=None, imgsz=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("Serve ultralytics:  pip install ultralytics")
        self.model = YOLO(weights)
        self.cfg   = cfg
        self.conf  = cfg["yolo_conf"]  if conf  is None else conf
        self.imgsz = cfg["yolo_imgsz"] if imgsz is None else imgsz

    @torch.no_grad()
    def boxes_for(self, imgs_cpu):
        """
        imgs_cpu: (B, C, H, W) alla risoluzione NATIVA, su CPU.

        Ritorna (boxes, found):
          boxes  (B, 4) float, coordinate in pixel nativi; righe di zeri dove
                 il detector non ha proposto nulla
          found  (B,)   bool, quali campioni hanno un box
        """
        ims = [_for_yolo(x.numpy(), self.cfg) for x in imgs_cpu]
        res = self.model.predict(ims, conf=self.conf, imgsz=self.imgsz,
                                 max_det=self.cfg["yolo_max_det"], verbose=False)
        B = len(ims)
        boxes = torch.zeros(B, 4)
        found = torch.zeros(B, dtype=torch.bool)
        for i, r in enumerate(res):
            if r.boxes is not None and len(r.boxes) > 0:
                # il piu' confidente: ultralytics ordina per confidenza
                boxes[i] = torch.as_tensor(r.boxes.xyxy[0].cpu(), dtype=torch.float32)
                found[i] = True
        return boxes, found

    @torch.no_grad()
    def dense_for(self, imgs_cpu, out_size, conf=None, max_det=None):
        """
        Le proposte di YOLO come MAPPA densa (B, 1, out_size, out_size) in
        [-1, 1], da dare a SAM2 come `input_masks`.

        PERCHE' NON IL RETTANGOLO. `input_boxes` sono due token d'angolo
        accodati alla sequenza del decoder; `input_masks` e' una mappa sommata
        pixel per pixel all'image embedding. Tre ordini di grandezza di banda in
        piu', e soprattutto: nella mappa ci stanno PIU' BOX. Il limite contro
        cui si sono fermati tutti gli esperimenti sul prompt — un rettangolo non
        puo' descrivere tre macchie in tre angoli — qui non esiste.

        PERCHE' SPARISCE LA SOGLIA. Ogni box dipinge la propria CONFIDENZA
        invece di essere tenuto o buttato da un `conf >` secco. Il modello
        impara da solo quanto fidarsi di una proposta a 0.15 contro una a 0.85:
        e' la soglia, ma continua e appresa. Per questo qui `conf` va tenuta
        BASSA (0.05): non serve piu' filtrare, serve graduare.

        Codifica: -1 dove non c'e' nessuna proposta, 2*conf-1 dentro un box,
        massimo dove due box si sovrappongono. Un campione senza alcuna proposta
        riceve una mappa costante -1, che e' neutra (una costante additiva
        uniforme non cambia nulla). E' anche il motivo per cui questo canale non
        ha bisogno di `_forward_split`: box e non-box convivono nello stesso
        batch.

        Ritorna (mappa, numero di box dipinti nel batch).
        """
        ims = [_for_yolo(x.numpy(), self.cfg) for x in imgs_cpu]
        H, W = imgs_cpu.shape[-2], imgs_cpu.shape[-1]
        res = self.model.predict(
            ims,
            conf=self.cfg.get("yolo_dense_conf", 0.05) if conf is None else conf,
            imgsz=self.imgsz,
            max_det=self.cfg.get("yolo_dense_max_det", 10) if max_det is None else max_det,
            verbose=False)

        B = len(ims)
        out = torch.full((B, 1, out_size, out_size), -1.0)
        sx, sy = out_size / W, out_size / H
        n_box = 0
        for i, r in enumerate(res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.cpu().numpy(),
                                           r.boxes.conf.cpu().numpy()):
                a = max(0, min(out_size - 1, int(np.floor(x0 * sx))))
                b = max(a + 1, min(out_size, int(np.ceil(x1 * sx))))
                t = max(0, min(out_size - 1, int(np.floor(y0 * sy))))
                d = max(t + 1, min(out_size, int(np.ceil(y1 * sy))))
                out[i, 0, t:d, a:b].clamp_(min=float(2.0 * c - 1.0))
                n_box += 1
        return out, n_box


# ============================================================
# 2. EXPORT IN FORMATO YOLO
# ============================================================

def _write_multich_tiff(path, arr):
    """
    Scrive un TIFF a N bande, come N PAGINE da (H,W).

    Il layout conta. Un TIFF a pagina singola con N campioni per pixel — che e'
    quello che `tifffile.imwrite` produce da un array (H,W,N) — OpenCV non lo
    decodifica: MISURATO su 14 bande, `cv2.imread(..., IMREAD_UNCHANGED)` ne
    restituisce 1.

    Ultralytics legge invece i TIFF con `cv2.imdecodemulti`, che estrae le
    PAGINE e le impila. Quindi si scrive (N,H,W): N pagine monocanale.

    `cv2.imwrite` non e' utilizzabile in nessuno dei due casi: il suo encoder
    TIFF si ferma a 4 canali.
    """
    try:
        import tifffile
    except ImportError:
        raise SystemExit(
            "Per il multicanale serve tifffile (cv2 non scrive TIFF oltre 4 "
            "bande):  pip install tifffile")
    tifffile.imwrite(path, np.ascontiguousarray(arr.transpose(2, 0, 1)))


def _verify_multich_tiff(path, n_ch):
    """
    Rilegge il primo TIFF scritto CON IL LETTORE DI ULTRALYTICS.

    Verificare con `cv2.imread` sarebbe inutile: non e' quello che il training
    usera'. Il fallimento tipico e' silenzioso — se il decoder restituisce meno
    bande di quelle scritte non alza nessun errore, e il detector si
    addestrerebbe su una frazione dei canali credendo di averli tutti. Meglio
    saperlo al file numero 1 che dopo tre ore.
    """
    import cv2
    im = None
    try:
        from ultralytics.utils.patches import imread as _uimread
        im = _uimread(path)
    except Exception:
        pass
    if im is None:      # ripiego: la stessa cosa che fa ultralytics sui TIFF
        ok, frames = cv2.imdecodemulti(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if ok and frames:
            im = np.stack(frames, axis=2) if len(frames) > 1 else frames[0]
    got = None if im is None else (1 if im.ndim == 2 else im.shape[-1])
    if got != n_ch:
        raise SystemExit(
            f"Il TIFF a {n_ch} bande e' stato scritto, ma il lettore di "
            f"ultralytics ne restituisce {got}.\n"
            f"  file: {path}\n"
            "  Il detector vedrebbe meno canali di quelli esportati, senza "
            "nessun errore a runtime.\n"
            "  Opzione: tornare a 3 canali (es. yolo_bands: [7, 11, 12] = "
            "B8, B12, slope), che tiene anche i pesi COCO intatti.")
    print(f"  verifica: {os.path.basename(path)} riletto con {got} bande, ok")


def export(cfg, out_dir, min_area=4):
    """
    Scrive il dataset nel layout che ultralytics si aspetta:

        out_dir/images/{train,val,test}/*.png
        out_dir/labels/{train,val,test}/*.txt      # class cx cy w h  (normalizzati)
        out_dir/data.yaml

    CRITICO: lo split e' quello di LENTE, ottenuto da
    `build_train_val_test_subsets` con lo stesso `random_seed`. Se YOLO si
    addestrasse su patch che finiscono nel test di LENTE, la cascata sarebbe
    valutata su dati visti dal detector — un leak esattamente come quello del
    gt_box, solo spostato di un modulo.
    """
    import cv2

    # ── formato delle immagini ──────────────────────────────────────────
    # 3 canali -> PNG, il percorso storico.
    # N canali -> .npy, perche' in un PNG non ci stanno. Serve pero' che
    # ultralytics sappia leggerli: si verifica SUBITO invece di scoprirlo dopo
    # aver scritto 3799 file.
    n_ch = _yolo_nch(cfg)
    ext = "png" if n_ch == 3 else "tif"
    if n_ch != 3:
        if cfg.get("rgb_stretch") != "minmax":
            raise SystemExit(
                f"yolo_bands ha {n_ch} canali ma rgb_stretch e' "
                f"{cfg.get('rgb_stretch')!r}: con canali non ottici serve "
                f"'minmax', altrimenti 'clip' azzera slope e DEM.")
        # Il multicanale passa per TIFF: e' l'unico formato in IMG_FORMATS che
        # regge piu' di 4 bande (npy non c'e' in ultralytics 8.4.x).
        try:
            import inspect
            from ultralytics.data.base import BaseDataset
        except ImportError:
            raise SystemExit("Serve ultralytics:  pip install ultralytics")
        if "channels" not in inspect.signature(BaseDataset.__init__).parameters:
            raise SystemExit(
                "Questa versione di ultralytics non supporta il multispettrale: "
                "BaseDataset.__init__ non accetta `channels`, quindi leggerebbe "
                "i TIFF con IMREAD_COLOR e ne terrebbe solo 3 bande.\n"
                "  Opzioni: aggiornare ultralytics, oppure tornare a 3 canali "
                "(es. yolo_bands: [7, 11, 12] = B8, B12, slope).")

    full_ds = build_dataset(cfg)
    tr, va, te, _ = build_train_val_test_subsets(full_ds, cfg)
    splits = {"train": tr.indices, "val": va.indices, "test": te.indices}

    for sub in ("images", "labels"):
        for sp in splits:
            os.makedirs(os.path.join(out_dir, sub, sp), exist_ok=True)

    print(f"Export a {n_ch} canali (bande {_yolo_bands(cfg)}) in .{ext}")

    checked = (n_ch == 3)      # il primo TIFF scritto viene riletto e verificato
    n_box_tot = 0
    for sp, idxs in splits.items():
        n_box = 0
        for i in tqdm(idxs, desc=f"export {sp}"):
            sample = full_ds[i]
            img = sample["image"].numpy()                  # (C,H,W)
            msk = sample["mask"].numpy()
            h, w = msk.shape[-2], msk.shape[-1]

            dst = os.path.join(out_dir, "images", sp, f"{i:06d}.{ext}")
            if n_ch == 3:
                cv2.imwrite(dst, cv2.cvtColor(_make_stack(img, cfg), cv2.COLOR_RGB2BGR))
            else:
                # stesso ordine di canali che _for_yolo produce a inferenza
                _write_multich_tiff(dst, _make_stack(img, cfg))
                if not checked:
                    _verify_multich_tiff(dst, n_ch)
                    checked = True

            # Le patch NEGATIVE producono un .txt VUOTO, non vengono saltate:
            # ultralytics le usa come background esplicito, ed e' cosi' che il
            # detector impara a non sparare box dove non c'e' niente. Su L4S
            # sono il 41%, quindi e' segnale importante.
            lines = []
            for x0, y0, x1, y1 in gt_boxes_for(msk, cfg):
                cx = (x0 + x1) / 2.0 / w
                cy = (y0 + y1) / 2.0 / h
                bw = (x1 - x0 + 1) / w
                bh = (y1 - y0 + 1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_box += len(lines)
            with open(os.path.join(out_dir, "labels", sp, f"{i:06d}.txt"), "w") as f:
                f.write("\n".join(lines))
        print(f"  {sp}: {len(idxs)} patch, {n_box} box  ({n_box/max(len(idxs),1):.2f} per patch)")
        n_box_tot += n_box

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n"
                f"train: images/train\nval: images/val\ntest: images/test\n"
                f"nc: 1\nnames: ['landslide']\n")
        # `channels` e' la chiave con cui ultralytics costruisce lo stem a N
        # canali. Scritta solo quando serve, cosi' i dataset a 3 canali gia'
        # esportati restano identici.
        if n_ch != 3:
            f.write(f"channels: {n_ch}\n")
    print(f"\nScritto {yaml_path} — {n_box_tot} box totali")
    return yaml_path


# ============================================================
# 3. TRAINING DEL DETECTOR
# ============================================================

def _expand_yolo_stem(yolo_obj, n_ch):
    """
    Allarga il primo Conv2d del detector da 3 a n_ch canali TENENDO i pesi COCO.

    Senza questo, con `channels: N` ultralytics costruirebbe lo stem da zero e
    si perderebbe il pretraining — su 2280 immagini e' quasi certamente peggio
    di qualunque guadagno spettrale.

    I 3 canali originali ereditano i filtri addestrati, i restanti partono da
    N(0, 0.001). E' la stessa operazione che LENTE fa sul patch embed di SAM2
    (LENTE_utils.py), con lo stesso limite noto: i canali nuovi devono imparare
    da zero. Qui pero' il resto della rete NON e' congelato, quindi puo'
    adattarsi a loro — a differenza di LENTE.

    Il Conv2d viene modificato SUL POSTO invece che sostituito: `forward` usa
    `self.weight`, e `in_channels` e' solo metadato (serve al controllo su
    `groups`). Cosi' non c'e' da rintracciare il modulo genitore.
    """
    net = yolo_obj.model
    for mod in net.modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            w = mod.weight.data
            new = w.new_zeros((w.shape[0], n_ch, *w.shape[2:]))
            new[:, :3] = w
            nn.init.normal_(new[:, 3:], mean=0.0, std=0.001)
            mod.weight = nn.Parameter(new)
            mod.in_channels = n_ch
            if isinstance(getattr(net, "yaml", None), dict):
                net.yaml["ch"] = n_ch
            print(f"  stem del detector allargato 3 -> {n_ch} canali "
                  f"(i primi 3 tengono i pesi COCO, gli altri partono da rumore)")
            return
    raise RuntimeError(
        "Nessun Conv2d con in_channels=3 trovato nel detector: impossibile "
        "allargare lo stem. Controlla che `model` sia un .pt pretrained.")


def train_yolo(data_yaml, model="yolo11m.pt", imgsz=512, epochs=300, batch=16,
               lr0=0.002, warmup_epochs=5.0, ch=3):
    """
    Addestra il detector. Thin wrapper su ultralytics: anchor, NMS e mosaic
    sono loro; qui si fissano solo le scelte che dipendono da QUESTO problema.

    * imgsz 512 su patch native 128: le frane occupano in media il 4% della
      patch, e sotto la testa P3 di YOLO (stride 8) una cella copre
      512/8 = 64 -> 2 pixel nativi. A 256 ne copriva 4: meta' delle GT stava
      in 1-2 celle, ai limiti di cio' che la griglia puo' assegnare.
      Misurato a 256 con yolo11s: recall fermo a ~0.38.
    * epochs 300 + patience 50: dataset piccolo (2280 immagini), il detector
      converge lentamente e l'early stopping di ultralytics lo chiude da solo
      quando smette di migliorare.
    * cos_lr: schedule piu' stabile del lineare sulle code lunghe.

    Augmentation: SOLO i ribaltamenti, e per una ragione geometrica precisa.

    Le immagini satellitari non hanno orientamento canonico, quindi verrebbe
    naturale ruotarle liberamente (degrees=180). MA le etichette qui sono box
    ALLINEATI AGLI ASSI: ruotando di un angolo arbitrario, ultralytics ricalcola
    il rettangolo allineato che contiene il box ruotato, e quello e' PIU' GRANDE
    dell'oggetto. Misurato:

        aspect 1:1 ruotato di 45 gradi -> area x2.00
        aspect 3:1 ruotato di 45 gradi -> area x2.67
        aspect 5:1 ruotato di 45 gradi -> area x3.60

    Le frane sono spesso allungate lungo il versante, quindi degrees>0 significa
    addestrare su etichette sistematicamente gonfiate. E' corruzione del dato,
    non augmentation. (Su MASCHERE di segmentazione la rotazione libera va
    benissimo: e' il formato box a non tollerarla.)

    flipud + fliplr invece preservano l'allineamento ESATTAMENTE e coprono 4
    delle 8 simmetrie diedrali: sono gratis e corrette. flipud in particolare e'
    a 0.0 nei default di ultralytics perche' sulle foto naturali il cielo sta in
    alto — qui non esiste un "alto", quindi va acceso.

    mixup resta a 0 (il default): i vincitori di Landslide4Sense lo usavano su
    SEGMENTATORI; su un detector di oggetti gia' piccoli e sfumati, mescolare
    due immagini li rende ancora meno visibili.
    mosaic resta al default 1.0: per oggetti piccoli e' l'augmentation piu'
    utile che ultralytics abbia.

    LEARNING RATE ABBASSATO (0.002 contro lo 0.01 di default), warmup allungato
    a 5 epoche. Misurato in modalita' "union": con il default il modello faceva
    mAP50 0.255 all'epoca 1 (quando il warmup tiene il LR basso) e poi CROLLAVA
    a 0.008 entro l'epoca 5, con la box_loss che SALIVA — divergenza, non lenta
    convergenza.
    Il motivo e' che in modalita' union i bersagli sono radi: UNO per immagine e
    nessuno sul 41% delle patch, quindi 6-11 istanze per batch da 16. Con un
    segnale cosi' rado il gradiente e' rumoroso e il LR di default e' troppo
    alto. In modalita' component (3 bersagli per patch) il default reggeva.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Serve ultralytics:  pip install ultralytics")
    y = YOLO(model)
    if ch != 3:
        _expand_yolo_stem(y, ch)
    return y.train(
        data=data_yaml, imgsz=imgsz, epochs=epochs, batch=batch,
        single_cls=True, patience=50, cos_lr=True,
        lr0=lr0, warmup_epochs=warmup_epochs,
        degrees=0.0, flipud=0.5, fliplr=0.5, mixup=0.0,
    )


# ============================================================
# 4. LA CASCATA
# ============================================================

@torch.no_grad()
def _regime(cfg, conf, max_det):
    """
    Soglia e numero massimo di proposte, presi dal config quando non forzati.

    Prima erano cablati (0.25 e 10) nelle firme, mentre la pipeline gira a
    yolo_conf 0.10 e yolo_max_det 1. Chi chiamava queste funzioni da codice
    invece che dal CLI misurava un regime diverso da quello reale, senza alcun
    errore: la stessa classe di trappola gia' vista con yolo_imgsz.
    """
    c = cfg["yolo_conf"] if conf is None else conf
    m = cfg["yolo_max_det"] if max_det is None else max_det
    return c, m


def cascade_masks(lente, yolo, img_t, cfg, conf=None, max_det=None):
    """
    Un campione: YOLO propone i box, LENTE segmenta dentro ciascuno, si unisce.

    img_t: (C,H,W) alla risoluzione nativa. Ritorna una maschera (H,W) {0,1}.

    Si cicla sui box invece di passarli tutti insieme a SAM2: con poche
    istanze per patch il costo e' irrilevante, e ogni maschera resta
    attribuibile al suo box (utile per il debug e per contare le istanze).
    """
    conf, max_det = _regime(cfg, conf, max_det)
    device = next(lente.parameters()).device
    target = cfg["target_size"]
    H, W = img_t.shape[-2], img_t.shape[-1]

    # imgsz ESPLICITO: senza, ultralytics usa il proprio default (640), che non
    # e' la risoluzione a cui il detector e' stato addestrato. Con detector a
    # 512 e a 1024 nello stesso progetto, ometterlo significa misurarli
    # entrambi a 640 e non accorgersene.
    det = yolo.predict(_for_yolo(img_t.numpy(), cfg), conf=conf,
                       imgsz=cfg["yolo_imgsz"],
                       max_det=max_det, verbose=False)[0]
    boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else np.empty((0, 4))

    x = F.interpolate(img_t.unsqueeze(0).to(device), (target, target),
                      mode="bilinear", align_corners=False)
    union = torch.zeros(H, W, dtype=torch.long)

    # NESSUNA DETECTION -> si segmenta SENZA prompt, non si restituisce vuoto.
    #
    # Il gating "niente box = maschera vuota" sembra gratis (una patch negativa
    # predetta vuota vale IoU 1.0) ma e' una trappola: misurato sul val, YOLO
    # propone un box solo sul 52% delle patch POSITIVE, quindi il 48% dei
    # positivi prenderebbe 0 secco. Il conto dice che anche con segmentazione
    # perfetta dentro i box la cascata si fermerebbe a 0.559, sotto il
    # prompt-free (0.6066).
    #
    # Passando None il modello lavora in regime prompt-free — e se e' stato
    # addestrato con `prompt_dropout` sa farlo. Cosi' la cascata e' un
    # INSTRADAMENTO (prompt buono dove c'e', comportamento normale altrove) e
    # non puo' fare peggio della baseline.
    if len(boxes) == 0:
        with torch.amp.autocast(device_type=device.type):
            preds, scores = lente(x, None)
        m = get_best_mask(preds, scores).float()
        m = F.interpolate(m, (H, W), mode="bilinear", align_corners=False)
        return (torch.sigmoid(m) > 0.5).long().squeeze().cpu(), 0

    for x0, y0, x1, y1 in boxes:
        b = torch.tensor([[x0, y0, x1, y1]], device=device, dtype=torch.float32)
        # scale_boxes e non una moltiplicazione a mano: aggiunge lo stesso
        # margine di 3px che i box GT ricevevano in training. Senza, i prompt
        # della cascata sarebbero sistematicamente piu' stretti di quelli su
        # cui LENTE e' stato addestrato.
        b = scale_boxes(b, H, W, target)
        with torch.amp.autocast(device_type=device.type):
            preds, scores = lente(x, b)
        m = get_best_mask(preds, scores).float()
        m = F.interpolate(m, (H, W), mode="bilinear", align_corners=False)
        union |= (torch.sigmoid(m) > 0.5).long().squeeze().cpu()

    return union, len(boxes)


def calibrate(cfg, yolo_path, conf=None, max_det=None):
    """
    Misura QUANTO SBAGLIA YOLO sui box, per tarare `prompt_jitter`.

    Il rischio della cascata a due stadi non e' la mancanza di gradiente
    congiunto: e' il disallineamento delle distribuzioni. LENTE addestrato su
    box GT perfetti riceve a inferenza box imprecisi, cioe' un prompt che non
    ha mai visto. La cura non richiede end-to-end (impossibile: l'NMS non e'
    derivabile): basta addestrare LENTE su box degradati QUANTO quelli veri.

    Qui si accoppiano i box predetti a quelli GT per massima IoU, e si misura
    lo scostamento delle coordinate come frazione del lato del box — che e'
    esattamente la parametrizzazione di `jitter_boxes(rel_std=...)`.

    Gira sul VAL, non sul test: e' una calibrazione, e va fatta su dati che non
    sono quelli del risultato finale.

    Stampa anche recall e precision a livello di ISTANZA: sono il tetto della
    cascata. Se YOLO trova il 78% delle frane, SAM2 non puo' segmentare il
    restante 22% — non le vede proprio.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Serve ultralytics:  pip install ultralytics")

    conf, max_det = _regime(cfg, conf, max_det)
    print(f"regime del detector: conf {conf}  max_det {max_det}  "
          f"imgsz {cfg['yolo_imgsz']}  bande {_yolo_bands(cfg)}")

    def _iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    full_ds = build_dataset(cfg)
    _, va, _, _ = build_train_val_test_subsets(full_ds, cfg)
    yolo = YOLO(yolo_path)

    rel_err, box_iou = [], []
    n_gt = n_pred = n_match = 0
    # Copertura in PIXEL: e' il vero tetto della cascata. Il recall per istanza
    # conta allo stesso modo una frana da 50x50 e un frammento da 5x5, ma per la
    # IoU finale mancare il frammento non costa quasi nulla e mancare la frana
    # grande costa tutto. Qui si misura che frazione dei pixel-frana cade dentro
    # almeno un box proposto — cioe' quanti pixel SAM2 ha la possibilita' di
    # segmentare.
    px_cov = px_tot = 0
    for i in tqdm(va.indices, desc="calibrazione"):
        s = full_ds[i]
        msk = s["mask"].numpy()
        # STESSO criterio dell'export (yolo_box_mode + min_area): altrimenti si
        # misurerebbe il recall contro GT diverse da quelle di addestramento.
        gts = gt_boxes_for(msk, cfg)
        det = yolo.predict(_for_yolo(s["image"].numpy(), cfg), conf=conf,
                           imgsz=cfg["yolo_imgsz"],
                           max_det=max_det, verbose=False)[0]
        preds = det.boxes.xyxy.cpu().numpy().tolist() if det.boxes is not None else []
        n_gt += len(gts); n_pred += len(preds)

        # pixel-frana raggiungibili: quelli dentro l'unione dei box proposti
        if msk.sum() > 0:
            reach = np.zeros_like(msk, dtype=bool)
            for p in preds:
                x0, y0 = int(max(p[0], 0)), int(max(p[1], 0))
                x1, y1 = int(min(p[2], msk.shape[1]-1)), int(min(p[3], msk.shape[0]-1))
                reach[y0:y1+1, x0:x1+1] = True
            px_cov += int((reach & (msk > 0)).sum())
            px_tot += int((msk > 0).sum())

        used = set()
        for g in gts:
            best, bi = None, 0.0
            for j, p in enumerate(preds):
                if j in used:
                    continue
                v = _iou(g, p)
                if v > bi:
                    best, bi = j, v
            if best is not None and bi >= 0.1:      # soglia bassa: interessa
                used.add(best); n_match += 1        # l'errore, non il conteggio
                box_iou.append(bi)
                p = preds[best]
                w = max(g[2] - g[0], 1.0); h = max(g[3] - g[1], 1.0)
                rel_err += [abs(p[0]-g[0])/w, abs(p[1]-g[1])/h,
                            abs(p[2]-g[2])/w, abs(p[3]-g[3])/h]

    rel_err = np.asarray(rel_err) if rel_err else np.zeros(1)
    print("\n" + "=" * 52)
    print("CALIBRAZIONE DEI BOX YOLO (sul VAL)")
    print("=" * 52)
    print(f"COPERTURA IN PIXEL : {px_cov/max(px_tot,1):.3f}   <- IL TETTO VERO")
    print(f"  (frazione di pixel-frana dentro almeno un box proposto: e' il")
    print(f"   massimo che la cascata puo' segmentare, il resto SAM2 non lo vede)")
    print()
    print(f"istanze GT {n_gt} | proposte {n_pred} | accoppiate {n_match}")
    print(f"  recall istanze    : {n_match/max(n_gt,1):.3f}   (conta i frammenti come le frane grandi)")
    print(f"  precision istanze : {n_match/max(n_pred,1):.3f}")
    print(f"  IoU media box accoppiati: {np.mean(box_iou) if box_iou else 0:.3f}")
    print(f"\nerrore relativo per coordinata (frazione del lato):")
    print(f"  media {rel_err.mean():.3f} | mediana {np.median(rel_err):.3f} | "
          f"p90 {np.percentile(rel_err, 90):.3f}")
    print(f"\n  ->  metti  prompt_jitter: {rel_err.mean():.2f}  in config.yaml")
    print(f"      e riaddestra LENTE con prompt_source: \"gt_box\"")
    print("=" * 52)
    return float(rel_err.mean())


def evaluate(cfg, yolo_path, lente_path, conf=None, max_det=None):
    """
    Valuta la cascata sul TEST set, con la decomposizione positivi/negativi.

    La decomposizione e' obbligatoria qui, non un extra: con la metrica attuale
    una patch negativa predetta vuota vale 1.0 e i negativi sono il 41% del
    test. Senza separare i due contributi non si distingue "segmenta meglio" da
    "il detector tace sui negativi".
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Serve ultralytics:  pip install ultralytics")

    conf, max_det = _regime(cfg, conf, max_det)
    print(f"regime del detector: conf {conf}  max_det {max_det}  "
          f"imgsz {cfg['yolo_imgsz']}  bande {_yolo_bands(cfg)}")

    full_ds = build_dataset(cfg)
    _, _, te, _ = build_train_val_test_subsets(full_ds, cfg)
    pos_set = set(full_ds.positive_indices)

    yolo = YOLO(yolo_path)

    # La cascata per definizione DA' i box a SAM2, quindi il modello va
    # costruito in regime "gt_box" a prescindere da cosa dice il config: con
    # prompt_source "none" resolve_prompt IGNOREREBBE i box di YOLO in silenzio
    # e questa funzione misurerebbe il prompt-free con un gating sopra.
    cfg = dict(cfg)
    if cfg.get("prompt_source") != "gt_box":
        print(f"  NB: prompt_source del config era '{cfg.get('prompt_source')}', "
              f"forzato a 'gt_box' (la cascata usa i box di YOLO come prompt)")
        cfg["prompt_source"] = "gt_box"
    lente = build_model(cfg, cfg["device"])
    ck = torch.load(lente_path, map_location=cfg["device"])
    # strict=True: un checkpoint dell'architettura sbagliata (es. boxhead) deve
    # fallire subito, non caricarsi monco in silenzio — stessa protezione di
    # test_boxhead.
    lente.load_state_dict(ck["model_state_dict"])
    lente.eval()

    iou_pos, iou_neg, n_det = [], [], []
    TP = FP = FN = 0.0
    for i in tqdm(te.indices, desc="cascata"):
        s = full_ds[i]
        pred, k = cascade_masks(lente, yolo, s["image"], cfg, conf, max_det)
        gt = s["mask"].long()
        n_det.append(k)

        inter = float((pred & gt).sum())
        un    = float((pred | gt).sum())
        iou   = 1.0 if un == 0 else inter / un        # vuoto+vuoto = 1.0, come calculate_metrics
        (iou_pos if i in pos_set else iou_neg).append(iou)

        TP += inter
        FP += float((pred & (1 - gt)).sum())
        FN += float(((1 - pred) & gt).sum())

    allv = iou_pos + iou_neg
    print("\n" + "=" * 52)
    print(f"CASCATA YOLO -> SAM2  ({cfg['dataset_name']})")
    print("=" * 52)
    print(f"IoU macro (metrica attuale)  : {np.mean(allv):.4f}   <- confronta con 0.6066")
    print(f"  sui positivi ({len(iou_pos):3d})          : {np.mean(iou_pos):.4f}")
    print(f"  sui negativi ({len(iou_neg):3d})          : {np.mean(iou_neg):.4f}")
    print(f"IoU micro (pooled, letteratura): {TP/(TP+FP+FN+1e-9):.4f}")
    print(f"F1  micro                      : {2*TP/(2*TP+FP+FN+1e-9):.4f}")
    print(f"box proposti per patch         : {np.mean(n_det):.2f}  "
          f"(0 su {sum(1 for k in n_det if k == 0)} patch)")
    print("=" * 52)


# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Cascata YOLO -> SAM2 per LENTE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Tutti i default sono None: se non passi il flag vince il config.yaml
    # (chiavi yolo_*), come per il resto del progetto.
    p = sub.add_parser("export")
    p.add_argument("--out", default=None); p.add_argument("--min-area", type=int, default=None)
    p = sub.add_parser("train")
    p.add_argument("--data", default=None); p.add_argument("--model", default=None)
    p.add_argument("--imgsz", type=int, default=None); p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p = sub.add_parser("calibrate"); p.add_argument("--yolo", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--yolo", required=True); p.add_argument("--lente", required=True)
    for name in ("calibrate", "eval"):
        sub.choices[name].add_argument("--conf", type=float, default=None)
        sub.choices[name].add_argument("--max-det", type=int, default=None)
        # La risoluzione di inferenza DEVE combaciare con quella di
        # addestramento del detector: un modello addestrato a 1024 valutato a
        # 512 (o al default 640 di ultralytics) non e' lo stesso modello.
        sub.choices[name].add_argument("--imgsz", type=int, default=None)
        # Le bande con cui il checkpoint e' stato ADDESTRATO. yolo_bands e'
        # globale, i checkpoint no: calibrare train-N con le bande di train-M
        # da' numeri falsi senza nessun errore. E' gia' successo con train-7.
        sub.choices[name].add_argument("--bands", type=str, default=None,
                                       help="es. \"3,2,1,12,13\" per train-10")

    for s in sub.choices.values():
        s.add_argument("-d", "--dataset", default=None)
        s.add_argument("-c", "--config", default="config.yaml")
        # Devono combaciare con quelli usati per addestrare il checkpoint LENTE:
        # un modello a 12 adapter non puo' caricare pesi a 24, e viceversa.
        # Senza questi flag si prendono i valori del config, che possono essere
        # tutt'altro rispetto al checkpoint che si sta valutando.
        s.add_argument("--adapter-where", choices=["attn", "mlp", "both"], default=None)
        s.add_argument("--adapter-dim", type=int, default=None)
        s.add_argument("--target-size", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, dataset=a.dataset)

    def opt(flag, key):
        """Flag da riga di comando se presente, altrimenti il valore del config."""
        v = getattr(a, flag, None)
        return cfg[key] if v is None else v

    if getattr(a, "imgsz", None) is not None and a.cmd in ("calibrate", "eval"):
        cfg["yolo_imgsz"] = a.imgsz
    if getattr(a, "bands", None) is not None:
        cfg["yolo_bands"] = [int(x) for x in a.bands.replace(" ", "").split(",")]
    if a.adapter_where is not None: cfg["adapter_where"] = a.adapter_where
    if a.adapter_dim is not None:   cfg["adapter_dim"] = a.adapter_dim
    if a.target_size is not None:   cfg["target_size"] = a.target_size

    # Cartella di export per-dataset: due dataset non devono pestarsi i piedi.
    out_dir = a.out if getattr(a, "out", None) else f"yolo_{cfg['dataset_name']}"

    if a.cmd == "export":
        export(cfg, out_dir, min_area=opt("min_area", "yolo_min_area"))
    elif a.cmd == "train":
        data = a.data or os.path.join(f"yolo_{cfg['dataset_name']}", "data.yaml")
        train_yolo(data, model=opt("model", "yolo_model"),
                   imgsz=opt("imgsz", "yolo_imgsz"),
                   epochs=opt("epochs", "yolo_epochs"),
                   batch=opt("batch", "yolo_batch"),
                   lr0=cfg["yolo_lr0"], warmup_epochs=cfg["yolo_warmup_epochs"],
                   ch=_yolo_nch(cfg))
    elif a.cmd == "calibrate":
        calibrate(cfg, a.yolo, conf=opt("conf", "yolo_conf"),
                  max_det=opt("max_det", "yolo_max_det"))
    else:
        evaluate(cfg, a.yolo, a.lente, conf=opt("conf", "yolo_conf"),
                 max_det=opt("max_det", "yolo_max_det"))


if __name__ == "__main__":
    main()

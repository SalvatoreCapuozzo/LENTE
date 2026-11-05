#!/usr/bin/env python
"""
LENTE_sonde.py — quattro misure su checkpoint gia' addestrati, senza addestrare.

Rispondono a domande a cui finora abbiamo risposto per deduzione.

    soffitto   Quanto vale DAVVERO un box perfetto, tolta la fuga della
               sentinella. E' il numero che decide se il ramo cascata resta
               vivo: se un box perfetto non compra niente sulle patch con
               frana, nessun detector puo' comprarlo.

    soglia     La soglia 0.5 non e' mai stata ottimizzata su L4S, e con
               pos_weight 6.44 la BCE trasla i logit. Scelta sul VAL,
               riportata sul TEST, mai il contrario.

    interp     L'encoder e' sensibile a COME vengono riempiti i pixel fra un
               campione nativo e l'altro? Se cambiare kernel non muove niente,
               un modello di superrisoluzione — che e' un modo molto piu' caro
               di riempire gli stessi pixel — non ha niente da comprare.

    tta        Otto simmetrie diedrali mediate. Le immagini satellitari non
               hanno orientamento canonico e sono gia' le trasformazioni usate
               in addestramento. Va riportato come cio' che e': cambia
               l'inferenza, non il modello.

RAPPORTO CON LENTE_diagnose.py. Quello e' piu' vecchio e piu' ricco sulla sola
soglia: valuta anche il gate su area e il gate su YOLO, e li sceglie tutti sul
val. Ma richiede un provider YOLO anche in regime prompt-free e si guida
modificando la costante RUNS. Qui la soglia c'e' in versione minima, senza
dipendenze, perche' tutti i risultati recenti sono prompt-free.

USO

    python LENTE_sonde.py soglia   --ckpt best_..._1024px.pth --target-size 1024
    python LENTE_sonde.py interp   --ckpt best_..._1024px.pth --target-size 1024
    python LENTE_sonde.py tta      --ckpt best_..._1024px.pth --target-size 1024
    python LENTE_sonde.py soffitto --ckpt best_lente_l4s_gtbox.pth --prompt gt_box

I flag di architettura DEVONO combaciare con quelli del checkpoint: la
configurazione del modello vive in config.yaml e non dentro il .pth, quindi un
disallineamento non lo segnala nessuno.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (load_config, build_dataset, build_train_val_test_subsets,
                         TransformSubset, build_model, get_best_mask, scale_boxes,
                         _bbox_from_mask, as_refine_map)


# ------------------------------------------------------------------
# le otto simmetrie del quadrato, come coppie (applica, inverti)
# ------------------------------------------------------------------
# Servono entrambe: la predizione va riportata nell'orientamento originale
# prima di poter essere mediata con le altre. Le composte con la trasposta non
# sono involuzioni, quindi la loro inversa va scritta a mano.

def _id(x):   return x
def _fh(x):   return x.flip(-1)
def _fv(x):   return x.flip(-2)
def _fhv(x):  return x.flip(-1).flip(-2)
def _t(x):    return x.transpose(-1, -2)
def _tfh(x):  return x.transpose(-1, -2).flip(-1)
def _tfv(x):  return x.transpose(-1, -2).flip(-2)
def _tfhv(x): return x.transpose(-1, -2).flip(-1).flip(-2)

DIEDRALI = [
    (_id,   _id),
    (_fh,   _fh),
    (_fv,   _fv),
    (_fhv,  _fhv),
    (_t,    _t),
    (_tfh,  lambda y: y.flip(-1).transpose(-1, -2)),
    (_tfv,  lambda y: y.flip(-2).transpose(-1, -2)),
    (_tfhv, lambda y: y.flip(-1).flip(-2).transpose(-1, -2)),
]

# nearest e area non accettano align_corners: passarglielo e' un errore
CON_ALLINEAMENTO = {"bilinear", "bicubic"}

# Quale stadio di H-SAM esce dal modello: "secondo" (default, la forward),
# "primo" (il decoder di SAM2) o "media" delle due probabilita'. Impostato
# da --uscita in main(); senza H-SAM conta solo "secondo".
USCITA = "secondo"


def _su(x, size, modo):
    """
    Ingrandimento dell'INGRESSO, in un punto solo.

    Tenuto separato da ogni altro interpolate del progetto perche' gli altri
    hanno vincoli diversi: sulle etichette serve nearest, e sulle predizioni la
    metrica va calcolata sulla griglia nativa. Qui si sceglie soltanto CON COSA
    riempire i pixel fra un campione nativo e l'altro.
    """
    kw = {"align_corners": False} if modo in CON_ALLINEAMENTO else {}
    return F.interpolate(x, size, mode=modo, **kw)


def _box_vista(masks, avanti):
    """Box di ogni campione, dalla maschera vista come la vede il modello."""
    m = avanti(masks.cpu())
    return torch.stack([_bbox_from_mask(mi) for mi in m])


def _avanti(model, x, b, n_ref):
    """
    Una forward piu' `n_ref` riletture della propria maschera, con lo STESSO
    prompt: e' cio' che fanno l'addestramento, la scelta del checkpoint e
    LENTE_test.py. Prima qui c'era la sola prima forward, cioe' si valutava
    il modello a meta' del lavoro per cui era stato addestrato.
    """
    logit = get_best_mask(*model(x, b))
    for _ in range(n_ref):
        r = as_refine_map(logit, model.mask_input_size[0])
        logit = get_best_mask(*model(x, b, refine=r))
    if USCITA == "secondo" or getattr(model, "hsam", None) is None:
        return logit
    # Con H-SAM la forward restituisce il secondo stadio; il primo (decoder di
    # SAM2) dell'ULTIMO passaggio resta in model.ultimo_prior. La sonda `cella`
    # ha misurato che il primo prende piu' frane piccole e il secondo sporca
    # meno le patch vuote: qui si sceglie quale dei due, o la loro media.
    primo = model.ultimo_prior
    if USCITA == "primo":
        return primo
    # media delle PROBABILITA', riportata in logit perche' il resto della
    # sonda applica la sigmoide dopo il ricampionamento sulla griglia nativa
    p = 0.5 * (torch.sigmoid(logit.float()) + torch.sigmoid(primo.float()))
    return torch.logit(p.clamp(1e-6, 1 - 1e-6)).to(logit.dtype)


@torch.no_grad()
def probabilita(model, loader, cfg, dev, interp="bilinear", n_tta=1,
                politica_box="none", desc="misura"):
    """
    Genera, batch per batch, (probabilita' sulla griglia NATIVA, maschera).

    Separare la produzione delle probabilita' dal loro punteggio e' cio' che
    permette alle quattro sonde di condividere ESATTAMENTE lo stesso percorso
    di inferenza. Se divergessero, i loro numeri non sarebbero confrontabili
    fra loro ne' con LENTE_test.py.

    politica_box:
      "none"    nessun prompt: il regime di tutti i risultati recenti
      "gt"      i box del dataset cosi' come sono, SENTINELLA COMPRESA sulle
                maschere vuote. E' il regime che ha prodotto lo 0.7501
      "gt_pos"  il box vero sulle patch con frana, NESSUN box sulle vuote.
                E' cio' che fa la cascata a inferenza: su un negativo il
                detector semplicemente non emette niente
    """
    T = cfg["target_size"]
    n_ref = int(cfg.get("refine_passes", 0))
    for batch in tqdm(loader, desc=desc, leave=False):
        imgs = batch["image"].to(dev)
        masks = batch["mask"].to(dev)
        h, w = imgs.shape[-2], imgs.shape[-1]
        vuota = (masks.flatten(1).sum(1) == 0)

        acc = None
        for k in range(n_tta):
            avanti, indietro = DIEDRALI[k]
            x = _su(avanti(imgs), (T, T), interp)

            if politica_box == "none":
                logit = _avanti(model, x, None, n_ref)
            else:
                # Il box va trasformato INSIEME all'immagine. Prima si passava
                # batch["boxes"], calcolato sulla vista originale: con --tta 8
                # sette viste su otto ricevevano il box nel punto sbagliato —
                # lo stesso difetto corretto in TransformSubset. Si ricalcola
                # dalla maschera trasformata con la stessa funzione del
                # dataset: sulla vista identita' deve coincidere col suo box.
                bv = _box_vista(batch["mask"], avanti)
                if k == 0 and not torch.equal(bv, batch["boxes"].float()):
                    raise RuntimeError(
                        "box ricalcolato diverso da quello del dataset sulla "
                        "vista identita': convenzioni dei box non allineate")
                b = scale_boxes(bv.to(dev), h, w, T)
            if politica_box == "gt":
                logit = _avanti(model, x, b, n_ref)
            elif politica_box == "gt_pos":
                # Box e non-box non convivono in una sola forward: SAM2 vuole
                # input_boxes uniforme sul batch. Due passate e si ricuce.
                logit = None
                for sel, bb in ((~vuota, b), (vuota, None)):
                    if not sel.any():
                        continue
                    o = _avanti(model, x[sel], None if bb is None else bb[sel], n_ref)
                    if logit is None:
                        logit = o.new_zeros((x.shape[0],) + tuple(o.shape[1:]))
                    logit[sel] = o

            # STESSO ordine di LENTE_test.py: prima si scende sulla griglia
            # nativa, poi la sigmoide. Invertirli cambia il risultato.
            p = torch.sigmoid(F.interpolate(logit.float(), (h, w),
                                            mode="bilinear", align_corners=False))
            acc = indietro(p) if acc is None else acc + indietro(p)
        yield acc / n_tta, masks


def _conta(prob, masks, soglie):
    """
    tp/fp/fn PER CAMPIONE e IoU per campione, per OGNI soglia in una passata.

    Restituiti per campione e non gia' sommati perche' servono due micro
    diversi: uno su tutte le patch, uno sulle sole patch con frana. La
    seconda e' la grandezza che riportano i lavori che escludono i negativi
    dal dataset, e senza i conteggi separati non si puo' ricavare.
    """
    gt = masks.unsqueeze(1).float()
    vuota = (gt.flatten(1).sum(1) == 0).cpu()
    out = {}
    for s in soglie:
        pr = (prob > s).float()
        tp = (pr * gt).sum((2, 3)).squeeze(1).cpu()
        fp = (pr * (1 - gt)).sum((2, 3)).squeeze(1).cpu()
        fn = ((1 - pr) * gt).sum((2, 3)).squeeze(1).cpu()
        ps = (tp + 1e-7) / (tp + fp + fn + 1e-7)
        out[s] = (tp, fp, fn, ps, vuota)
    return out


class Registro:
    """Accumula i conteggi di tutte le soglie su tutto lo split."""

    def __init__(self, soglie):
        # [tp, fp, fn su TUTTE le patch, poi gli stessi tre sulle sole
        #  patch con frana, poi le IoU per campione dei due gruppi]
        self.s = {k: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], []] for k in soglie}

    def aggiungi(self, conteggi):
        for k, (tp, fp, fn, ps, vuota) in conteggi.items():
            r = self.s[k]
            pos = ~vuota
            r[0] += float(tp.sum()); r[1] += float(fp.sum()); r[2] += float(fn.sum())
            r[3] += float(tp[pos].sum()); r[4] += float(fp[pos].sum())
            r[5] += float(fn[pos].sum())
            r[6] += ps[pos].tolist(); r[7] += ps[vuota].tolist()

    def riga(self, k):
        tp, fp, fn, tpp, fpp, fnp, frane, vuote = self.s[k]
        return {"micro_iou": tp / max(tp + fp + fn, 1.0),
                "micro_f1": 2 * tp / max(2 * tp + fp + fn, 1.0),
                # Stessa formula, ma i pixel delle patch VUOTE non entrano.
                # Su una patch vuota non esistono ne' veri positivi ne' falsi
                # negativi: contribuisce SOLO falsi positivi, quindi toglierla
                # alza il micro per costruzione. E' il numero da usare per
                # confrontarsi con chi esclude i negativi dal dataset, e non
                # va mai messo nella stessa colonna dell'altro.
                "micro_pos_iou": tpp / max(tpp + fpp + fnp, 1.0),
                "micro_pos_f1": 2 * tpp / max(2 * tpp + fpp + fnp, 1.0),
                "frane": float(np.mean(frane)) if frane else 0.0,
                "vuote": float(np.mean(vuote)) if vuote else 0.0,
                # La IoU per IMMAGINE mediata su TUTTE le patch: e' quello che
                # calcola segmentation_models_pytorch con reduction
                # "micro-imagewise" e zero_division=1.0, cioe' la metrica della
                # letteratura che confrontiamo (DiGATe-UNet e simili). Una patch
                # vuota prevista vuota vale 1.0, quindi un modello che non dice
                # mai niente prende gia' la frazione di patch vuote: sul nostro
                # test 1127/2338 = 0.4821. Va riportata SEMPRE accanto a quel
                # valore di partenza, mai da sola.
                "per_immagine": ((sum(frane) + sum(vuote))
                                 / max(len(frane) + len(vuote), 1)),
                "banale": len(vuote) / max(len(frane) + len(vuote), 1),
                "n_frane": len(frane), "n_vuote": len(vuote)}


def _intestazione():
    print(f"\n{'':>26} {'micro IoU':>10} {'micro F1':>9} {'con frana':>10} "
          f"{'vuote':>8} {'micro+ IoU':>11} {'per immagine':>13}")
    print("  " + "-" * 94)


def _stampa(nome, r, base=None):
    d = f"   {r['micro_iou'] - base['micro_iou']:+.4f}" if base is not None else ""
    print(f"{nome:>26} {r['micro_iou']:>10.4f} {r['micro_f1']:>9.4f} "
          f"{r['frane']:>10.4f} {r['vuote']:>8.4f} "
          f"{r['micro_pos_iou']:>11.4f} {r['per_immagine']:>13.4f}{d}")


CODA = """
  micro+ IoU e' lo stesso micro calcolato SOLO sulle patch con frana. Sale per
  costruzione, perche' una patch vuota non porta ne' veri positivi ne' falsi
  negativi: contribuisce solo falsi positivi. Serve a confrontarsi con i lavori
  che escludono i negativi dal dataset — e va riportato dichiarando che lo e',
  mai nella stessa colonna del micro completo.

  per immagine e' la IoU di ogni patch mediata su tutte le patch: la metrica
  della letteratura (smp, reduction "micro-imagewise", zero_division=1.0). Una
  patch vuota prevista vuota vale 1.0, quindi un modello che non predice MAI
  niente prende gia' la frazione di patch vuote dello split. Quel valore di
  partenza e' stampato sotto la tabella: senza, la colonna non vuol dire niente."""


def _misura(model, loader, cfg, dev, soglie=(0.5,), extra=(), **kw):
    reg = Registro(list(soglie))
    for p, m in probabilita(model, loader, cfg, dev, **kw):
        reg.aggiungi(_conta(p, m, list(soglie)))
        for e in extra:
            e.aggiungi(p, m)
    return reg


# Classi di area in pixel NATIVI (10 m). 16 px e' la frana mediana di Sen12:
# la prima classe e' quella che a 512 di ingresso occupa una cella sola della
# griglia su cui SAM decide.
CLASSI_AREA = [(1, 16), (17, 64), (65, 256), (257, None)]


def _classe_area(a):
    for i, (lo, hi) in enumerate(CLASSI_AREA):
        if a >= lo and (hi is None or a <= hi):
            return i
    return len(CLASSI_AREA) - 1


class PerArea:
    """
    Quanto di ogni frana VERA viene preso, per classe di area.

    La IoU per campione mescola frane grandi e piccole nella stessa patch, e
    il micro e' dominato dalle grandi perche' pesa per pixel. Qui ogni frana e'
    un oggetto (componente a 8 vicini sulla griglia nativa, la stessa
    definizione della blob loss) e si contano, per classe di area:
      prese        frane con almeno un pixel previsto;
      coperte      frane con almeno meta' dei pixel previsti;
      recall px    pixel di frana previsti, sul totale della classe.
    Misura il lato del RICHIAMO. I falsi positivi non hanno una dimensione
    vera a cui attribuirli, quindi qui non compaiono: restano nelle colonne
    della tabella principale.
    """

    def __init__(self, soglia):
        self.soglia = soglia
        k = len(CLASSI_AREA)
        self.n, self.prese, self.coperte = [0] * k, [0] * k, [0] * k
        self.px, self.tp = [0.0] * k, [0.0] * k

    def aggiungi(self, prob, masks):
        pr = (prob[:, 0] > self.soglia).cpu().numpy().astype(np.float64)
        gt = masks.cpu().numpy() > 0
        for b in range(gt.shape[0]):
            if not gt[b].any():
                continue
            lab, n = ndimage.label(gt[b], structure=OTTO)
            aree = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            presi = np.bincount(lab.ravel(), weights=pr[b].ravel(), minlength=n + 1)[1:]
            for k in range(n):
                c = _classe_area(int(aree[k]))
                self.n[c] += 1
                self.px[c] += float(aree[k])
                self.tp[c] += float(presi[k])
                self.prese[c] += int(presi[k] > 0)
                self.coperte[c] += int(presi[k] >= 0.5 * aree[k])

    def stampa(self):
        print(f"\n  FRANE PER CLASSE DI AREA (test, soglia {self.soglia:.2f})")
        print(f"  {'area px':>12} {'frane':>7} {'prese':>8} {'coperte >= 50%':>15} {'recall px':>10}")
        for i, (lo, hi) in enumerate(CLASSI_AREA):
            nome = f"{lo}-{hi}" if hi is not None else f">{lo - 1}"
            n = max(self.n[i], 1)
            print(f"  {nome:>12} {self.n[i]:>7} {self.prese[i] / n:>8.1%} "
                  f"{self.coperte[i] / n:>15.1%} {self.tp[i] / max(self.px[i], 1):>10.1%}")


def _carica_modello(cfg, dev, percorso, costruttore=None):
    model = (costruttore or build_model)(cfg, dev)
    ck = torch.load(percorso, map_location=dev)
    sd = ck.get("model_state_dict", ck)
    inc = model.load_state_dict(sd, strict=False)

    # strict=False ignora in SILENZIO le chiavi mancanti. Su un parametro
    # congelato va bene: il valore giusto arriva comunque dai pesi
    # preaddestrati. Su un parametro ADDESTRABILE no — resterebbe
    # all'inizializzazione casuale, e il numero stampato in fondo sarebbe privo
    # di senso senza che niente segnali un errore. Un disallineamento di FORMA
    # invece esplode da solo, quindi il caso pericoloso e' solo questo.
    addestrabili = {n for n, q in model.named_parameters() if q.requires_grad}
    persi = sorted(addestrabili.intersection(inc.missing_keys))
    print(f"  pesi: {len(sd)} nel checkpoint | {len(inc.missing_keys)} mancanti "
          f"| {len(inc.unexpected_keys)} inattesi")
    if persi:
        for k in persi[:8]:
            print(f"    MANCANTE E ADDESTRABILE: {k}")
        raise SystemExit(f"{len(persi)} parametri addestrabili non sono nel "
                         f"checkpoint: resterebbero casuali e la misura sarebbe "
                         f"priva di senso. Controlla i flag di architettura.")
    meta = [k for k in ck if k != "model_state_dict"] if isinstance(ck, dict) else []
    if meta:
        print(f"  il checkpoint porta anche: {', '.join(map(str, meta))}")
    model.eval()
    return model


# ------------------------------------------------------------------
# Le frane come oggetti: componenti connesse a 8 vicini sulla griglia nativa,
# la stessa definizione della blob loss.
OTTO = np.ones((3, 3), dtype=int)


def _frane(sub, area_min):
    """Ogni frana del sottoinsieme: (patch, etichetta, area, riga, colonna del centro)."""
    out = []
    for i in range(len(sub)):
        m = sub[i]["mask"].numpy() > 0
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=OTTO)
        aree = ndimage.sum(m, lab, index=range(1, n + 1))
        centri = ndimage.center_of_mass(m, lab, index=range(1, n + 1))
        for k in range(n):
            if aree[k] >= area_min:
                out.append((i, k + 1, int(aree[k]),
                            int(round(centri[k][0])), int(round(centri[k][1]))))
    return out


def _piccole(a, model, guidato, cfg, dev, sub, carica, pol):
    """
    Dieci frane piccole scelte A CASO, con la mappa di probabilita' del
    modello. Prima stampa come sono distribuite le aree, perche' "piccola"
    va definita sui dati e non a occhio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fr = _frane(sub, a.area_min)
    aree = np.array([f[2] for f in fr])
    print(f"\n  {len(fr)} frane (componenti a 8 vicini, >= {a.area_min} px) in "
          f"{len(set(f[0] for f in fr))} patch del {a.split}")
    print(f"  {'percentile':>10} {'area px':>8} {'frane sotto':>12} {'pixel-frana sotto':>18}")
    for q in (10, 25, 50, 75, 90):
        v = np.percentile(aree, q)
        s = aree <= v
        print(f"  {q:>9}% {v:>8.0f} {s.mean():>12.0%} {aree[s].sum() / aree.sum():>18.1%}")

    amax = a.area_max if a.area_max is not None else int(np.percentile(aree, 25))
    cand = [f for f in fr if f[2] <= amax]
    # Una frana per patch, estratte a caso con seme fisso: nessuna scelta a
    # mano, e rilanciando si rivedono le stesse.
    rng = np.random.default_rng(a.seed)
    scelte, viste = [], set()
    for j in rng.permutation(len(cand)):
        if cand[j][0] not in viste:
            scelte.append(cand[j])
            viste.add(cand[j][0])
        if len(scelte) == a.n:
            break
    print(f"\n  'piccola' = area fra {a.area_min} e {amax} px "
          f"({'primo quartile' if a.area_max is None else 'scelta a mano'}): "
          f"{len(cand)} frane. Estratte a caso {len(scelte)}, seme {a.seed}.")

    loader = carica(Subset(sub, [f[0] for f in scelte]))
    P = torch.cat([p.cpu() for p, _ in probabilita(
        model, loader, cfg, dev, n_tta=a.tta, politica_box=pol,
        desc="piccole")])[:, 0].numpy()
    PG = None
    if guidato is not None:
        PG = torch.cat([p.cpu() for p, _ in probabilita(
            guidato, loader, cfg, dev, n_tta=a.tta, politica_box="gt_pos",
            desc="con riquadro")])[:, 0].numpy()
    sg = a.soglia if a.soglia_guidato is None else a.soglia_guidato

    ch = cfg.get("rgb_channels", [3, 2, 1])
    Z = a.zoom
    titoli = ["immagine B4 B3 B2", "verita'", "probabilita'",
              "zoom immagine", "zoom probabilita'"]
    if PG is not None:
        titoli.append("zoom, con riquadro vero")
    fig, ax = plt.subplots(len(scelte), len(titoli), squeeze=False,
                           figsize=(2.4 * len(titoli), 2.5 * len(scelte)))
    for c, t in enumerate(titoli):
        ax[0, c].set_title(t, fontsize=10)

    def _stat(p, questa, m, soglia):
        d = p[questa]
        return (d.max(), d.mean(), (d > soglia).mean(),
                int(((p > soglia) & ~m).sum()))

    print(f"\n  {'patch':>5} {'area':>5}   {'max p':>6} {'media p':>8} "
          f"{'px > soglia':>12} {'falsi px':>9}   (soglia {a.soglia:.2f})")
    for r, (i, k, area, cy, cx) in enumerate(scelte):
        s = sub[i]
        m = s["mask"].numpy() > 0
        lab, _ = ndimage.label(m, structure=OTTO)
        questa = lab == k
        rgb = s["image"].numpy()[ch].transpose(1, 2, 0).astype(np.float64)
        lo, hi = np.percentile(rgb, (2, 98))          # solo per la vista
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
        ver = np.zeros(m.shape + (3,))
        ver[m] = 0.45
        ver[questa] = (1.0, 0.85, 0.0)
        H, W = m.shape
        y0 = int(np.clip(cy - Z // 2, 0, H - Z))
        x0 = int(np.clip(cx - Z // 2, 0, W - Z))
        zs = (slice(y0, y0 + Z), slice(x0, x0 + Z))

        mx, me, fr_in, fp = _stat(P[r], questa, m, a.soglia)
        riga = (f"  {i:>5} {area:>5}   {mx:>6.2f} {me:>8.2f} {fr_in:>12.0%} {fp:>9}")
        pannelli = [(rgb, None), (ver, None), (P[r], "magma"),
                    (rgb[zs], None), (P[r][zs], "magma")]
        if PG is not None:
            gmx, gme, gfr, gfp = _stat(PG[r], questa, m, sg)
            riga += f"   | con riquadro: {gmx:.2f} {gme:.2f} {gfr:.0%} {gfp}"
            pannelli.append((PG[r][zs], "magma"))
        print(riga)

        for c, (img, cmap) in enumerate(pannelli):
            q = ax[r, c]
            kw = {"vmin": 0, "vmax": 1, "cmap": cmap} if cmap else {}
            q.imshow(img, interpolation="nearest", **kw)
            zoom = c >= 3
            q.contour(questa[zs] if zoom else questa, levels=[0.5],
                      colors="cyan", linewidths=0.9)
            if not zoom:
                q.add_patch(Rectangle((x0 - .5, y0 - .5), Z, Z, fill=False,
                                      ec="white", lw=0.8, ls="--"))
            q.set_xticks([]); q.set_yticks([])
        bx = _bbox_from_mask(s["mask"]).tolist()
        if pol == "gt_pos":
            # il modello principale e' guidato: si mostra il riquadro che ha
            # ricevuto davvero, cioe' l'unione di TUTTE le frane della patch
            for c in range(3):
                ax[r, c].add_patch(Rectangle((bx[0] - .5, bx[1] - .5),
                                             bx[2] - bx[0] + 1, bx[3] - bx[1] + 1,
                                             fill=False, ec="lime", lw=1.2))
        if PG is not None:
            ax[r, 5].add_patch(Rectangle((bx[0] - x0 - .5, bx[1] - y0 - .5),
                                         bx[2] - bx[0] + 1, bx[3] - bx[1] + 1,
                                         fill=False, ec="lime", lw=1.2))
            # il riquadro puo' uscire dal ritaglio: senza questo gli assi si
            # allargano per contenerlo e l'immagine si rimpicciolisce
            ax[r, 5].set_xlim(-.5, Z - .5)
            ax[r, 5].set_ylim(Z - .5, -.5)
            ax[r, 5].set_xlabel(f"max {gmx:.2f} · media {gme:.2f}", fontsize=8)
        ax[r, 0].set_ylabel(f"patch {i}\n{area} px", fontsize=9)
        ax[r, 4].set_xlabel(f"max {mx:.2f} · media {me:.2f}", fontsize=8)

    fig.suptitle(f"{a.ckpt}\nfrane piccole ({a.area_min}-{amax} px) estratte a caso "
                 f"dal {a.split}, seme {a.seed} · {a.tta} viste · probabilita' nero 0 "
                 f"-> giallo 1 · contorno ciano = la frana scelta · tratteggio bianco = "
                 f"finestra dello zoom"
                 + (" · verde = riquadro dato a SAM" if pol == "gt_pos" or PG is not None else ""),
                 fontsize=9)
    alto = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / alto))   # spazio per il titolo
    fig.savefig(a.out, dpi=110)
    print(f"""
  figura salvata in {a.out}

  Sono dieci esempi, non una misura: servono a VEDERE se le frane piccole
  vengono mancate (probabilita' vicina a zero dentro il contorno) o trovate e
  disegnate male (probabilita' alta ma che deborda o si spezza). Il numero
  vero e' la IoU per classe di dimensione su tutto il {a.split}.""")


# ------------------------------------------------------------------
# Modo `cella`: la decisione di SAM2 vive sulla griglia a stride 16 (le mappe
# a S/4 e S/8 non ricevono semantica dall'alto: fpn_top_down_levels = [2, 3]).
# Qui si misura se la presa di una frana dipende da QUANTE celle di quella
# griglia tocca, e se il secondo stadio di H-SAM recupera frane che il primo
# aveva perso.
CLASSI_CELLE = [(1, 1), (2, 2), (3, 4), (5, None)]


def _celle_toccate(comp, lato_cella):
    """Quante celle distinte della griglia S/16 tocca una componente (maschera bool nativa)."""
    r, c = np.nonzero(comp)
    return len(set(zip((r // lato_cella).tolist(), (c // lato_cella).tolist())))


class PerCella:
    """Presa delle frane per numero di celle S/16 toccate, a una soglia fissa."""

    def __init__(self, soglia, lato_cella):
        self.soglia, self.lato = soglia, lato_cella
        k = len(CLASSI_CELLE)
        self.n, self.prese, self.piccole_una = [0] * k, [0] * k, [0, 0]

    def aggiungi(self, prob, masks):
        pr = (prob[:, 0] > self.soglia).cpu().numpy()
        gt = masks.cpu().numpy() > 0
        for b in range(gt.shape[0]):
            if not gt[b].any():
                continue
            lab, n = ndimage.label(gt[b], structure=OTTO)
            for k in range(1, n + 1):
                comp = lab == k
                nc = _celle_toccate(comp, self.lato)
                i = next(j for j, (lo, hi) in enumerate(CLASSI_CELLE)
                         if nc >= lo and (hi is None or nc <= hi))
                self.n[i] += 1
                self.prese[i] += int(pr[b][comp].any())
                if comp.sum() <= 16:
                    self.piccole_una[0] += 1
                    self.piccole_una[1] += int(nc == 1)

    def stampa(self, titolo):
        print(f"\n  {titolo}: presa per celle S/16 toccate "
              f"(cella = {self.lato}x{self.lato} px nativi, soglia {self.soglia:.2f})")
        print(f"  {'celle':>8} {'frane':>7} {'prese':>8}")
        for i, (lo, hi) in enumerate(CLASSI_CELLE):
            nome = str(lo) if hi == lo else (f"{lo}-{hi}" if hi else f">={lo}")
            print(f"  {nome:>8} {self.n[i]:>7} {self.prese[i] / max(self.n[i], 1):>8.1%}")
        t, u = self.piccole_una
        print(f"  frane da 1-16 px che stanno in UNA sola cella: {u} su {t} "
              f"({u / max(t, 1):.1%})")


@torch.no_grad()
def _cella(a, model, cfg, dev, sub, carica):
    """Primo e secondo stadio sulla stessa patch, alla stessa soglia, senza TTA."""
    T = cfg["target_size"]
    n_ref = int(cfg.get("refine_passes", 0))
    # px nativi per cella S/16: il lato nativo e' load_size (256 su Bijie), 128 su L4S
    lato = max(1, round(16 * int(cfg.get("load_size") or 128) / T))
    hsam = getattr(model, "hsam", None) is not None
    fin_a, fin_c = PerArea(a.soglia), PerCella(a.soglia, lato)
    pri_a, pri_c = PerArea(a.soglia), PerCella(a.soglia, lato)
    for batch in tqdm(carica(sub), desc="cella", leave=False):
        imgs, masks = batch["image"].to(dev), batch["mask"].to(dev)
        h, w = imgs.shape[-2:]
        logit = _avanti(model, _su(imgs, (T, T), "bilinear"), None, n_ref)
        nat = lambda lg: torch.sigmoid(F.interpolate(lg.float(), (h, w), mode="bilinear",
                                                     align_corners=False))
        p = nat(logit)
        fin_a.aggiungi(p, masks); fin_c.aggiungi(p, masks)
        if hsam:
            q = nat(model.ultimo_prior)
            pri_a.aggiungi(q, masks); pri_c.aggiungi(q, masks)
    print(f"\n  ingresso {T} px -> una cella S/16 copre {lato}x{lato} px nativi")
    if hsam:
        print("\n  PRIMO STADIO (decoder di SAM2), ultimo passaggio")
        pri_a.stampa()
        pri_c.stampa("primo stadio")
        print("\n  SECONDO STADIO (uscita del modello)")
    fin_a.stampa()
    fin_c.stampa("uscita")
    print("""
  Lettura. Se la presa crolla nella riga `1` cella e sale con le celle, il
  limite e' la griglia su cui SAM2 decide, non il decoder. Se il secondo
  stadio non prende piu' frane del primo nella stessa riga, H-SAM rifinisce
  cio' che il primo stadio ha gia' trovato ma non recupera cio' che ha perso.""")


# ------------------------------------------------------------------
# Modo `fusione`: LENTE dice SE nella patch c'e' una frana (e' forte sulle
# vuote), la rete nativa (rete_nativa.py, U-Net a 128 px) dice DOVE sono
# quelle piccole, che la griglia a stride 16 di SAM2 diluisce.
#
#     P = max( P_LENTE ,  pi * P_rete )      pi = max della mappa di LENTE
#
# Dove LENTE vede la patch vuota pi e' basso e la rete nativa viene spenta,
# quindi i suoi falsi positivi sulle vuote non contano; dove LENTE vede
# qualcosa, la rete nativa puo' aggiungere le frane piccole. Nessun
# addestramento: la fusione avviene tutta qui.

@torch.no_grad()
def _tre_mappe(model, rete, loader, cfg, dev):
    T = cfg["target_size"]
    n_ref = int(cfg.get("refine_passes", 0))
    for batch in tqdm(loader, desc="fusione", leave=False):
        imgs, masks = batch["image"].to(dev), batch["mask"].to(dev)
        h, w = imgs.shape[-2:]
        la = _avanti(model, _su(imgs, (T, T), "bilinear"), None, n_ref)
        pa = torch.sigmoid(F.interpolate(la.float(), (h, w), mode="bilinear",
                                         align_corners=False))
        pb = torch.sigmoid(rete(imgs).float())
        pi = pa.amax(dim=(2, 3), keepdim=True)
        yield {"LENTE": pa, "rete nativa": pb, "fusione": torch.maximum(pa, pi * pb)}, masks


def _fusione(a, model, cfg, dev, va, te, carica):
    from rete_nativa import costruisci
    ck = torch.load(a.ckpt_nativa, map_location=dev)
    rete = costruisci(ck["n_canali"], ck.get("encoder", "resnet18")).to(dev)
    rete.load_state_dict(ck["model_state_dict"])
    rete.eval()
    print(f"  rete nativa {a.ckpt_nativa}: epoca {ck.get('epoch')}, "
          f"micro val {ck.get('micro', float('nan')):.4f}")

    nomi = ("LENTE", "rete nativa", "fusione")
    soglie = [round(float(x), 2) for x in np.arange(a.soglia_min, a.soglia_max + 1e-9, 0.05)]
    rv = {n: Registro(soglie) for n in nomi}
    for mappe, m in _tre_mappe(model, rete, carica(va), cfg, dev):
        for n in nomi:
            rv[n].aggiungi(_conta(mappe[n], m, soglie))
    chiave = {"micro": "micro_iou", "micropiu": "micro_pos_iou",
              "per_immagine": "per_immagine", "frane": "frane"}[a.scegli_su]
    best = {n: max(soglie, key=lambda s, n=n: rv[n].riga(s)[chiave]) for n in nomi}
    print(f"\n  soglie scelte sul val su {a.scegli_su}: "
          + ", ".join(f"{n} {best[n]:.2f}" for n in nomi))

    lato = max(1, round(16 * int(cfg.get("load_size") or 128) / cfg["target_size"]))
    s_te = {n: sorted({0.5, best[n]}) for n in nomi}
    rt = {n: Registro(s_te[n]) for n in nomi}
    per_area = {n: PerArea(0.5) for n in nomi}
    per_cella = {n: PerCella(0.5, lato) for n in nomi}
    for mappe, m in _tre_mappe(model, rete, carica(te), cfg, dev):
        for n in nomi:
            rt[n].aggiungi(_conta(mappe[n], m, s_te[n]))
            per_area[n].aggiungi(mappe[n], m)
            per_cella[n].aggiungi(mappe[n], m)

    print("\nTEST — qui si RIPORTA")
    _intestazione()
    for n in nomi:
        _stampa(f"{n} 0.50", rt[n].riga(0.5))
        if best[n] != 0.5:
            _stampa(f"{n} {best[n]:.2f}", rt[n].riga(best[n]))
    r = rt["LENTE"].riga(0.5)
    print(f"\n  predittore banale, colonna 'per immagine': {r['banale']:.4f}")
    for n in nomi:
        print(f"\n  ===== {n} (soglia 0.50, per confrontare a parita' di soglia) =====")
        per_area[n].stampa()
        per_cella[n].stampa(n)
    print("""
  Criterio fissato prima: la fusione vale se alza le frane da 1-16 px e quelle
  in UNA cella rispetto a LENTE, senza abbassare le vuote sotto 0.89.""")


# ------------------------------------------------------------------
# Modo `bootstrap`: due checkpoint sullo STESSO test. Lo scarto fra i due e'
# piu' grande del caso dovuto a quali patch sono capitate nel test?
#
# Si ricampionano le patch di test con reinserimento, N volte, e su ogni
# ricampionamento si misurano ENTRAMBI i modelli sulle stesse patch (bootstrap
# appaiato). Si riporta lo scarto osservato, l'intervallo al 95% (percentili
# 2.5 e 97.5) e la frazione di ricampionamenti in cui B non e' sotto A.
# Copre la variabilita' del TEST, non quella dell'addestramento: per quella
# servono piu' run, cioe' i fold.

@torch.no_grad()
def _per_patch(model, loader, cfg, dev, soglie, desc):
    """tp, fp, fn e IoU di ogni patch del loader, per ogni soglia, in ordine."""
    acc = {s: [[], [], [], []] for s in soglie}
    for p, m in probabilita(model, loader, cfg, dev, desc=desc):
        for s, (tp, fp, fn, ps, _) in _conta(p, m, soglie).items():
            for lista, v in zip(acc[s], (tp, fp, fn, ps)):
                lista.append(v.numpy().astype(np.float64))
    return {s: [np.concatenate(l) for l in acc[s]] for s in soglie}


def _bootstrap(a, model_a, carica_b, cfg, dev, va, te, carica):
    soglie = [round(float(x), 2) for x in np.arange(a.soglia_min, a.soglia_max + 1e-9, 0.05)]
    chiave = {"micro": "micro_iou", "micropiu": "micro_pos_iou",
              "per_immagine": "per_immagine", "frane": "frane"}[a.scegli_su]

    def misura(model, nome):
        # soglia scelta sul val, per ciascun modello, come nel modo `soglia`
        rv = _misura(model, carica(va), cfg, dev, soglie=soglie, desc=f"val {nome}")
        best = max(soglie, key=lambda s: rv.riga(s)[chiave])
        if best in (soglie[0], soglie[-1]):
            raise SystemExit(f"{nome}: la soglia migliore {best:.2f} cade sul bordo, "
                             f"allarga --soglia-min/--soglia-max")
        print(f"  {nome}: soglia scelta sul val su {a.scegli_su} = {best:.2f}")
        return best, _per_patch(model, carica(te), cfg, dev, sorted({0.5, best}), f"test {nome}")

    sa, pa = misura(model_a, "A")
    sb, pb = misura(carica_b(), "B")

    # stesso test, stesso ordine: i pixel di frana per patch devono coincidere
    ga, gb = pa[0.5][0] + pa[0.5][2], pb[0.5][0] + pb[0.5][2]
    if not np.array_equal(ga, gb):
        raise SystemExit("le patch di test dei due modelli non coincidono: split diversi?")

    n = len(ga)
    rng = np.random.default_rng(a.seed)
    righe = [("soglia scelta sul val", pa[sa], pb[sb]), ("soglia 0.50", pa[0.5], pb[0.5])]

    def micro(c, idx):
        tp = c[0][idx].sum(-1)
        return tp / np.maximum(tp + c[1][idx].sum(-1) + c[2][idx].sum(-1), 1.0)

    def per_imm(c, idx):
        return c[3][idx].mean(-1)

    print(f"\nBOOTSTRAP APPAIATO — {n} patch di test, {a.n_boot} ricampionamenti, seme {a.seed}")
    print(f"  A = {a.ckpt} (soglia {sa:.2f})\n  B = {a.ckpt_b} (soglia {sb:.2f})")
    print(f"\n  {'':>24} {'metrica':>13} {'A':>8} {'B':>8} {'B - A':>8} "
          f"{'IC 95%':>20} {'B >= A':>8}")
    tutti = np.arange(n)
    for nome, ca, cb in righe:
        for metrica, f in (("micro IoU", micro), ("per immagine", per_imm)):
            va_, vb_ = f(ca, tutti), f(cb, tutti)
            d = []
            for inizio in range(0, a.n_boot, 1000):          # a blocchi: memoria limitata
                idx = rng.integers(0, n, size=(min(1000, a.n_boot - inizio), n))
                d.append(f(cb, idx) - f(ca, idx))
            d = np.concatenate(d)
            lo, hi = np.percentile(d, (2.5, 97.5))
            print(f"  {nome:>24} {metrica:>13} {va_:>8.4f} {vb_:>8.4f} {vb_ - va_:>+8.4f} "
                  f"  [{lo:+.4f}, {hi:+.4f}] {np.mean(d >= 0):>8.1%}")
    print("""
  Lettura. IC tutto sopra lo zero: B e' meglio di A su questo test, e non per
  quali patch sono capitate. IC che attraversa lo zero: lo scarto e' compatibile
  con il caso, si scrive "alla pari". La variabilita' dell'ADDESTRAMENTO non e'
  qui dentro: un altro seme o un altro fold possono spostare entrambi i modelli.
  Le soglie scelte sul val sono trattate come fisse: la loro incertezza non
  entra nell'intervallo.""")


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["soffitto", "soglia", "interp", "tta", "piccole",
                                     "cella", "fusione", "bootstrap"])
    ap.add_argument("--ckpt-b", default=None,
                    help="solo per `bootstrap`: il secondo checkpoint, stessa architettura "
                         "salvo --multi-b")
    ap.add_argument("--multi-b", action="store_true",
                    help="solo per `bootstrap`: il secondo checkpoint e' di LENTE_multi_v2.py")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--vecchia-scelta-a", action="store_true",
                    help="riaccende su --ckpt la scelta dinamica fra le maschere di SAM2, "
                         "per misurare il vecchio comportamento contro quello corretto")
    ap.add_argument("--salva", default=None,
                    help="solo per `soglia`: scrive val e test in questo file JSON")
    ap.add_argument("--fold", type=int, default=0,
                    help="fold dello split a rotazione (0-4): deve essere quello del checkpoint")
    ap.add_argument("--ckpt-nativa", default="best_rete_nativa_l4s.pth",
                    help="solo per `fusione`: il checkpoint di rete_nativa.py")
    ap.add_argument("--ckpt", required=True)
    # ---- solo per il modo `piccole`
    ap.add_argument("--n", type=int, default=10, help="quante frane piccole")
    ap.add_argument("--area-min", type=int, default=4,
                    help="sotto questa area una macchia e' piu' probabilmente "
                         "rumore di annotazione che una frana")
    ap.add_argument("--area-max", type=int, default=None,
                    help="default: il primo quartile delle aree, misurato")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--zoom", type=int, default=32, help="lato del ritaglio, px nativi")
    ap.add_argument("--soglia", type=float, default=0.5,
                    help="la soglia scelta sul val per QUESTO checkpoint")
    ap.add_argument("--soglia-guidato", type=float, default=None)
    ap.add_argument("--ckpt-guidato", type=str, default=None,
                    help="checkpoint addestrato con il riquadro vero, stessa "
                         "architettura: aggiunge la colonna con il riquadro")
    ap.add_argument("--out", type=str, default="frane_piccole.png")
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--prompt", choices=["none", "gt_box"], default="none")
    ap.add_argument("--adapter-where", choices=["attn", "mlp", "both"], default=None)
    ap.add_argument("--adapter-dim", type=int, default=None)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--encoder",
                    choices=["hiera", "unet", "prithvi", "terracodec"],
                    default=None)
    ap.add_argument("--norm-ingresso", action="store_true",
                    help="inizializza l'affine d'ingresso di TerraCodec e "
                         "Prithvi con le statistiche del train, cosi' ricevono "
                         "ingressi standardizzati come in addestramento. L4S "
                         "ha media ~1 e deviazione ~0.45: all'identita' sono "
                         "fuori scala.")
    ap.add_argument("--tec-variante", type=str, default=None)
    ap.add_argument("--tec-compression", type=int, default=None)
    ap.add_argument("--tec-size", type=int, default=None)
    ap.add_argument("--prithvi-size", type=int, default=None)
    ap.add_argument("--prithvi-model", type=str, default=None)
    ap.add_argument("--prithvi-adapter-dim", type=int, default=None)
    ap.add_argument("--prithvi-lora", type=int, default=None)
    ap.add_argument("--unfreeze-blocks", type=int, default=None)
    ap.add_argument("--gsrm", action="store_true")
    ap.add_argument("--hsam", action="store_true",
                    help="deve combaciare con il checkpoint")
    ap.add_argument("--fpn-completo", action="store_true",
                    help="deve combaciare con il checkpoint")
    ap.add_argument("--multi", action="store_true",
                    help="checkpoint di LENTE_multi_v2.py: si misura il dominio -d")
    ap.add_argument("--uscita", choices=["secondo", "primo", "media"], default="secondo",
                    help="solo con --hsam: quale stadio misurare. secondo = uscita "
                         "del modello; primo = decoder di SAM2; media = media delle "
                         "due probabilita'")
    ap.add_argument("--fusione", choices=["none", "mgem", "dgfm", "entrambi"],
                    default=None,
                    help="deve combaciare con il checkpoint: senza, lo stream "
                         "ausiliario e i moduli di fusione non vengono nemmeno "
                         "costruiti e i loro pesi risulterebbero inattesi")
    ap.add_argument("--fusione-width", type=int, default=None)
    ap.add_argument("--fusione-aux", choices=["terreno", "prithvi"], default=None)
    # Se l'ottimo cade sull'ESTREMO dell'intervallo non e' un ottimo: e' il
    # bordo della griglia. Va allargato e rimisurato, altrimenti si riporta un
    # valore troncato senza saperlo.
    # Su COSA si sceglie la soglia sul val. Il micro completo include le
    # patch vuote: per un modello addestrato solo sulle positive — che le vuote
    # non le ha mai viste — li' produce spazzatura, e la soglia verrebbe
    # scelta per contenere quella invece che per segmentare bene le frane.
    ap.add_argument("--scegli-su",
                    choices=["micro", "micropiu", "frane", "per_immagine"],
                    default="micro",
                    help="micro (default) sul val completo; micropiu sulle "
                         "sole patch con frana, da usare per l'oracolo e per "
                         "ogni modello addestrato con --solo-positive; "
                         "per_immagine e' la metrica della letteratura, "
                         "dove una patch vuota prevista vuota vale 1.0")
    ap.add_argument("--soglia-min", type=float, default=0.20)
    ap.add_argument("--soglia-max", type=float, default=0.80)
    # Senza questo il modello viene ricostruito a 12 canali ottici e il patch
    # embed non combacia col checkpoint: stessa trappola dei flag mancanti che
    # ci e' costata tre volte. La configurazione del modello vive in
    # config.yaml e non dentro il .pth.
    ap.add_argument("--bands", type=str, default=None,
                    help="le bande OTTICHE con cui il checkpoint e' stato "
                         "addestrato, es. \"3,7\"")
    ap.add_argument("--batch-size", type=int, default=4)
    # La TTA cambia la DISTRIBUZIONE delle probabilita': mediare otto viste
    # smussa i picchi, quindi la soglia scelta senza TTA non e' piu' quella
    # giusta con la TTA. Applicarla sopra una soglia scelta a parte sarebbe
    # esattamente l'errore che il protocollo val->test serve a evitare.
    ap.add_argument("--refine-passes", type=int, default=None,
                    help="default: quello del config, cioe' quello con cui il "
                         "modello e' stato addestrato. 0 riproduce le misure "
                         "vecchie, fatte sulla sola prima forward")
    ap.add_argument("--tta", type=int, default=1, choices=[1, 4, 8],
                    help="viste diedrali nel modo `soglia`: lo sweep e la "
                         "scelta avvengono nello stesso regime in cui si "
                         "riporta il risultato")
    a = ap.parse_args()

    cfg = load_config(a.config, dataset=a.dataset, prompt_source=a.prompt)
    for k, v in (("adapter_where", a.adapter_where), ("adapter_dim", a.adapter_dim),
                 ("target_size", a.target_size), ("encoder", a.encoder),
                 ("unfreeze_blocks", a.unfreeze_blocks)):
        if v is not None:
            cfg[k] = v
    if a.norm_ingresso:
        cfg["norm_ingresso"] = True
    if a.tec_variante is not None:
        cfg["tec_variante"] = a.tec_variante
    if a.tec_compression is not None:
        cfg["tec_compression"] = a.tec_compression
    if a.tec_size is not None:
        cfg["tec_size"] = a.tec_size
    if a.prithvi_size is not None:
        cfg["prithvi_size"] = a.prithvi_size
    if a.prithvi_model is not None:
        cfg["prithvi_model"] = a.prithvi_model
    if a.prithvi_adapter_dim is not None:
        cfg["prithvi_adapter_dim"] = a.prithvi_adapter_dim
    if a.prithvi_lora is not None:
        cfg["prithvi_lora"] = a.prithvi_lora
    if a.gsrm:
        cfg["gsrm"] = True
    if a.hsam:
        cfg["hsam"] = True
    if a.fpn_completo:
        cfg["fpn_completo"] = True
    if a.uscita != "secondo" and not a.hsam:
        raise SystemExit("--uscita primo/media ha senso solo con --hsam")
    global USCITA
    USCITA = a.uscita
    if a.hsam:
        print(f"  H-SAM, stadio misurato: {USCITA}")
    if a.fusione is not None:
        cfg["fusione"] = a.fusione
    if a.fusione_width is not None:
        cfg["fusione_width"] = a.fusione_width
    if a.fusione_aux is not None:
        cfg["fusione_aux"] = a.fusione_aux
    if a.bands is not None:
        cfg["optical_bands"] = [int(x) for x in a.bands.replace(" ", "").split(",")]
        cfg["n_optical"] = len(cfg["optical_bands"])
    if a.refine_passes is not None:
        cfg["refine_passes"] = a.refine_passes
    print(f"  passaggi di raffinamento: {int(cfg.get('refine_passes', 0))}"
          + ("   (FORZATO: non e' quello dell'addestramento)"
             if a.refine_passes is not None else "   (come in addestramento)"))
    dev = cfg["device"]
    # Con gt_box, fuori dal modo `soffitto`, si valuta SEMPRE senza la
    # sentinella: e' l'unico regime il cui numero significhi qualcosa. Quello
    # con la sentinella resta disponibile solo nel confronto a tre di
    # `soffitto`, dove serve proprio a mostrare la fuga.
    pol = "gt_pos" if a.prompt == "gt_box" else "none"

    cfg["fold"] = a.fold
    full = build_dataset(cfg)
    _, va, te, _ = build_train_val_test_subsets(full, cfg)

    def carica(sub):
        return DataLoader(TransformSubset(sub, A.Compose([ToTensorV2()])),
                          batch_size=a.batch_size, shuffle=False, pin_memory=True)

    def modello(percorso, multi):
        if multi:
            from LENTE_multi_v2 import costruisci_multi
            m = _carica_modello(cfg, dev, percorso, costruttore=costruisci_multi)
            m.imposta_dominio(a.dataset)
            print(f"  modello multi-dominio, dominio misurato: {a.dataset}")
            return m
        return _carica_modello(cfg, dev, percorso)

    model = modello(a.ckpt, a.multi)
    if a.vecchia_scelta_a:
        # il comportamento di prima del 2026-11-04 (vedi LENTE.__init__)
        model.model.mask_decoder.dynamic_multimask_via_stability = True
        print("  A con la scelta dinamica fra le maschere ACCESA (comportamento vecchio)")
    print(f"\ncheckpoint {a.ckpt}   ingresso {cfg['target_size']}   prompt {a.prompt}")

    # --------------------------------------------------------------
    if a.modo == "soffitto":
        print("""
Il 0.7501 dell'oracolo gt_box conteneva la fuga della sentinella: sulle
maschere vuote _bbox_from_mask restituisce [0,0,10,10] e il modello aveva
imparato a riconoscerla, prendendo IoU 1.0000 esatto sul 40% del test.

Qui la stessa cosa su griglia NATIVA e nei tre regimi. La colonna che conta e'
`con frana`: dove una frana c'e', la sentinella non esiste, quindi quel numero
e' pulito in tutte le righe ed e' esattamente cio' che un detector perfetto
potrebbe comprare.""")
        _intestazione()
        ris = {}
        for nome, p in (("box veri + sentinella", "gt"),
                        ("box veri, vuote senza", "gt_pos"),
                        ("nessun prompt", "none")):
            ris[p] = _misura(model, carica(te), cfg, dev,
                             politica_box=p, desc=nome).riga(0.5)
            _stampa(nome, ris[p])
        print(f"""
  ATTENZIONE alla terza riga. Quel `nessun prompt` NON e' una linea di base:
  e' QUESTO checkpoint, addestrato a ricevere sempre un box, privato
  dell'ingresso da cui dipende. Misura la sua dipendenza dal prompt, non
  quanto vale un prompt. Confrontarci il guadagno darebbe un numero gonfiato.

  L'unica cifra pulita di questa tabella e' `con frana` = {ris['gt']['frane']:.4f},
  identica nelle due righe con i box perche' sui positivi il box e' lo stesso.
  Va confrontata con il `con frana` di un modello addestrato SENZA prompt, allo
  stesso valore di soglia e con lo stesso numero di viste. La differenza e' il
  soffitto della cascata: quanto varrebbe un detector PERFETTO, che nessun
  detector reale puo' essere.

  Anche la colonna `vuote` va letta cosi': 1.0000 esatto con la sentinella e
  {ris['gt_pos']['vuote']:.4f} senza. Quel salto e' la fuga misurata, non una
  prestazione persa — togliendo la sentinella il modello si trova senza box su
  patch dove ne ha sempre ricevuto uno, e si accende.""")

    # --------------------------------------------------------------
    elif a.modo == "soglia":
        soglie = [round(float(x), 2) for x in
                  np.arange(a.soglia_min, a.soglia_max + 1e-9, 0.05)]
        if a.tta > 1:
            print(f"  sweep CON {a.tta} viste diedrali: la soglia viene scelta "
                  f"nello stesso regime in cui verra' riportata")
        rv = _misura(model, carica(va), cfg, dev, soglie=soglie,
                     politica_box=pol, n_tta=a.tta, desc="val")
        print("\nVALIDAZIONE — qui si SCEGLIE, e solo qui")
        _intestazione()
        for s in soglie:
            _stampa(f"soglia {s:.2f}", rv.riga(s))
        chiave = {"micro": "micro_iou", "micropiu": "micro_pos_iou",
                  "per_immagine": "per_immagine",
                  "frane": "frane"}[a.scegli_su]
        best = max(soglie, key=lambda s: rv.riga(s)[chiave])
        if a.scegli_su != "micro":
            print(f"\n  soglia scelta su {a.scegli_su}, non sul micro completo")
        print(f"\n  scelta sul val: {best:.2f}   "
              f"(a 0.50 dava {rv.riga(0.5)['micro_iou']:.4f})")
        if best in (soglie[0], soglie[-1]):
            raise SystemExit(
                f"\n  L'OTTIMO CADE SUL BORDO della griglia "
                f"[{soglie[0]:.2f}, {soglie[-1]:.2f}]. Non e' un massimo,"
                f"\n  e' un troncamento: il valore vero sta oltre. "
                f"Rilancia allargando,"
                f"\n  per esempio con  --soglia-max {min(soglie[-1] + 0.15, 0.98):.2f}\n")

        per_area = PerArea(best)
        rt = _misura(model, carica(te), cfg, dev, soglie=[0.5, best],
                     politica_box=pol, n_tta=a.tta, desc="test", extra=[per_area])
        print("\nTEST — qui si RIPORTA")
        _intestazione()
        _stampa("soglia 0.50", rt.riga(0.5))
        _stampa(f"soglia {best:.2f}", rt.riga(best), base=rt.riga(0.5))
        r = rt.riga(best)
        print(f"\n  predittore banale (mai una frana), colonna 'per immagine': "
              f"{r['banale']:.4f}  = {r['n_vuote']} vuote su "
              f"{r['n_vuote'] + r['n_frane']} patch di test")
        per_area.stampa()
        if a.salva:
            import json
            with open(a.salva, "w") as fh:
                json.dump({"ckpt": a.ckpt, "dataset": a.dataset, "fold": a.fold,
                           "multi": a.multi, "scegli_su": a.scegli_su, "tta": a.tta,
                           "soglia_val": best, "val": rv.riga(best),
                           "test_05": rt.riga(0.5), "test": rt.riga(best)}, fh, indent=1)
            print(f"\n  risultati salvati in {a.salva}")
        print("""
  Va dichiarato in tabella: la soglia e' scelta sulla validazione mentre i
  benchmark pubblicati usano 0.5 fissa. E' un vantaggio di protocollo, non di
  modello.""" + CODA)

    # --------------------------------------------------------------
    elif a.modo == "fusione":
        _fusione(a, model, cfg, dev, va, te, carica)

    # --------------------------------------------------------------
    elif a.modo == "bootstrap":
        if a.ckpt_b is None:
            raise SystemExit("bootstrap: serve --ckpt-b")
        _bootstrap(a, model, lambda: modello(a.ckpt_b, a.multi_b), cfg, dev, va, te, carica)

    # --------------------------------------------------------------
    elif a.modo == "cella":
        _cella(a, model, cfg, dev, va if a.split == "val" else te, carica)

    # --------------------------------------------------------------
    elif a.modo == "piccole":
        guidato = None
        if a.ckpt_guidato is not None:
            # stesso modello, ma costruito per LEGGERE il riquadro: con
            # prompt_source none resolve_prompt lo scarterebbe in silenzio
            cfg_g = dict(cfg, prompt_source="gt_box")
            print(f"\ncheckpoint con riquadro {a.ckpt_guidato}")
            guidato = _carica_modello(cfg_g, dev, a.ckpt_guidato)
        _piccole(a, model, guidato, cfg, dev, va if a.split == "val" else te,
                 carica, pol)

    # --------------------------------------------------------------
    elif a.modo == "interp":
        print("""
Il modello e' addestrato su bilineare, quindi OGNI scambio peggiora: non si
misura quale kernel sia migliore, si misura QUANTO l'encoder sia sensibile a
cio' che sta fra un campione nativo e l'altro.

  tutti dentro +-0.0106  ->  indifferente. Un modello di superrisoluzione e' un
                             modo piu' caro di riempire gli stessi pixel:
                             ipotesi chiusa.
  crollo su nearest      ->  sensibile all'alta frequenza. Il gradino dopo e'
                             riaddestrare con kernel bicubico, non la
                             superrisoluzione.""")
        _intestazione()
        base = None
        for modo in ("bilinear", "bicubic", "area", "nearest"):
            r = _misura(model, carica(te), cfg, dev, interp=modo,
                        politica_box=pol, desc=modo).riga(0.5)
            _stampa(modo, r, base=base)
            base = base or r

    # --------------------------------------------------------------
    elif a.modo == "tta":
        if a.prompt != "none":
            raise SystemExit("La TTA qui vale solo prompt-free: con un box "
                             "andrebbe trasformato anche quello, e su un box "
                             "allineato agli assi la trasposta non e' gratis.")
        _intestazione()
        base = None
        for n in (1, 4, 8):
            r = _misura(model, carica(te), cfg, dev, n_tta=n,
                        desc=f"{n} viste").riga(0.5)
            _stampa(f"{n} viste", r, base=base)
            base = base or r
        print("""
  Le prime 4 sono i ribaltamenti, le altre 4 aggiungono la trasposta. Il costo
  e' lineare nel numero di viste e non tocca il modello: se il salto sta fra 1
  e 4, le altre quattro non servono.""")
    print()


if __name__ == "__main__":
    main()

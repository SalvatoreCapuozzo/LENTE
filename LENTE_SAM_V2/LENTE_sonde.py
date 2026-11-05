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
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (load_config, build_dataset, build_train_val_test_subsets,
                         TransformSubset, build_model, get_best_mask, scale_boxes)


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
                logit = get_best_mask(*model(x, None))
            elif politica_box == "gt":
                b = scale_boxes(batch["boxes"].to(dev), h, w, T)
                logit = get_best_mask(*model(x, b))
            else:
                # Box e non-box non convivono in una sola forward: SAM2 vuole
                # input_boxes uniforme sul batch. Due passate e si ricuce.
                b = scale_boxes(batch["boxes"].to(dev), h, w, T)
                logit = None
                for sel, bb in ((~vuota, b), (vuota, None)):
                    if not sel.any():
                        continue
                    o = get_best_mask(*model(x[sel], None if bb is None else bb[sel]))
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
                "n_frane": len(frane), "n_vuote": len(vuote)}


def _intestazione():
    print(f"\n{'':>26} {'micro IoU':>10} {'micro F1':>9} {'con frana':>10} "
          f"{'vuote':>8} {'micro+ IoU':>11}")
    print("  " + "-" * 80)


def _stampa(nome, r, base=None):
    d = f"   {r['micro_iou'] - base['micro_iou']:+.4f}" if base is not None else ""
    print(f"{nome:>26} {r['micro_iou']:>10.4f} {r['micro_f1']:>9.4f} "
          f"{r['frane']:>10.4f} {r['vuote']:>8.4f} "
          f"{r['micro_pos_iou']:>11.4f}{d}")


CODA = """
  micro+ IoU e' lo stesso micro calcolato SOLO sulle patch con frana. Sale per
  costruzione, perche' una patch vuota non porta ne' veri positivi ne' falsi
  negativi: contribuisce solo falsi positivi. Serve a confrontarsi con i lavori
  che escludono i negativi dal dataset — e va riportato dichiarando che lo e',
  mai nella stessa colonna del micro completo."""


def _misura(model, loader, cfg, dev, soglie=(0.5,), **kw):
    reg = Registro(list(soglie))
    for p, m in probabilita(model, loader, cfg, dev, **kw):
        reg.aggiungi(_conta(p, m, list(soglie)))
    return reg


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["soffitto", "soglia", "interp", "tta"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--prompt", choices=["none", "gt_box"], default="none")
    ap.add_argument("--adapter-where", choices=["attn", "mlp", "both"], default=None)
    ap.add_argument("--adapter-dim", type=int, default=None)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--encoder",
                    choices=["hiera", "unet", "prithvi", "terracodec"],
                    default=None)
    ap.add_argument("--tec-variante", type=str, default=None)
    ap.add_argument("--tec-compression", type=int, default=None)
    ap.add_argument("--tec-size", type=int, default=None)
    ap.add_argument("--prithvi-size", type=int, default=None)
    ap.add_argument("--prithvi-model", type=str, default=None)
    ap.add_argument("--prithvi-adapter-dim", type=int, default=None)
    ap.add_argument("--prithvi-lora", type=int, default=None)
    ap.add_argument("--unfreeze-blocks", type=int, default=None)
    ap.add_argument("--gsrm", action="store_true")
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
    if a.fusione is not None:
        cfg["fusione"] = a.fusione
    if a.fusione_width is not None:
        cfg["fusione_width"] = a.fusione_width
    if a.fusione_aux is not None:
        cfg["fusione_aux"] = a.fusione_aux
    if a.bands is not None:
        cfg["optical_bands"] = [int(x) for x in a.bands.replace(" ", "").split(",")]
        cfg["n_optical"] = len(cfg["optical_bands"])
    dev = cfg["device"]
    pol = "gt" if a.prompt == "gt_box" else "none"

    full = build_dataset(cfg)
    _, va, te, _ = build_train_val_test_subsets(full, cfg)

    def carica(sub):
        return DataLoader(TransformSubset(sub, A.Compose([ToTensorV2()])),
                          batch_size=a.batch_size, shuffle=False, pin_memory=True)

    model = build_model(cfg, dev)
    ck = torch.load(a.ckpt, map_location=dev)
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
        best = max(soglie, key=lambda s: rv.riga(s)["micro_iou"])
        print(f"\n  scelta sul val: {best:.2f}   "
              f"(a 0.50 dava {rv.riga(0.5)['micro_iou']:.4f})")
        if best in (soglie[0], soglie[-1]):
            raise SystemExit(
                f"\n  L'OTTIMO CADE SUL BORDO della griglia "
                f"[{soglie[0]:.2f}, {soglie[-1]:.2f}]. Non e' un massimo,"
                f"\n  e' un troncamento: il valore vero sta oltre. "
                f"Rilancia allargando,"
                f"\n  per esempio con  --soglia-max {min(soglie[-1] + 0.15, 0.98):.2f}\n")

        rt = _misura(model, carica(te), cfg, dev, soglie=[0.5, best],
                     politica_box=pol, n_tta=a.tta, desc="test")
        print("\nTEST — qui si RIPORTA")
        _intestazione()
        _stampa("soglia 0.50", rt.riga(0.5))
        _stampa(f"soglia {best:.2f}", rt.riga(best), base=rt.riga(0.5))
        print("""
  Va dichiarato in tabella: la soglia e' scelta sulla validazione mentre i
  benchmark pubblicati usano 0.5 fissa. E' un vantaggio di protocollo, non di
  modello.""" + CODA)

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

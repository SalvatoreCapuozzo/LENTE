#!/usr/bin/env python
"""
LENTE_bande.py — quali bande contano, misurato in due modi indipendenti.

LA DOMANDA. Oggi entrano tutte e 12 le bande ottiche piu' pendenza e DEM. Le
prime tre ereditano i pesi RGB preaddestrati, le altre nove partono da rumore
N(0, 1e-3) e devono imparare da zero con l'encoder a valle congelato: e' il
caso peggiore, molte bande e poca capacita' per sfruttarle. In letteratura un
sottoinsieme di 8 canali eguaglia o supera configurazioni fino a 30, perche' le
bande ridondanti DEGRADANO attivamente (fenomeno di Hughes).

DUE MISURE CHE NON DICONO LA STESSA COSA.

  separabilita   Quanta informazione c'e' NEL DATO, senza alcun modello. Per
                 ogni canale, quanto bene un singolo valore di pixel separa
                 frana da non-frana. Non dipende da come e' fatta LENTE, e non
                 puo' essere confusa da un difetto architetturale.

  permutazione   Quanta informazione il MODELLO ADDESTRATO usa davvero. Si
                 mescola un canale alla volta fra i campioni e si guarda quanto
                 crolla la micro IoU. Non richiede riaddestramento.

Incrociarle e' il punto: le due misure creano quattro caselle, e tre di esse
sono azioni diverse.

    separabile + usato        tenere
    separabile + NON usato    informazione lasciata sul tavolo: il problema non
                              e' la banda, e' lo stem che non sa leggerla
    non separabile + usato    il modello ci si appoggia per correlazione
                              spuria: candidato forte alla rimozione
    non separabile + non usato   togliere, e' solo rumore in ingresso

METODO. La separabilita' si misura SOLO sul train: scegliere le bande e' una
decisione di modellazione, e guardarla su val o test sarebbe la stessa colpa
del box della ground truth spostata di un modulo.

USO

    python LENTE_bande.py separabilita
    python LENTE_bande.py permutazione --ckpt best_..._1024px.pth \\
           --adapter-where both --adapter-dim 128 --target-size 1024
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (load_config, build_dataset, build_train_val_test_subsets,
                         TransformSubset, build_model, get_best_mask)

# L4S: 12 bande Sentinel-2 (B8A esclusa) piu' pendenza e DEM da ALOS PALSAR.
NOMI = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
        "B9", "B10", "B11", "B12", "slope", "DEM"]

# Risoluzione NATIVA in metri. Riportata perche' e' la variabile su cui il
# miglior metodo pubblicato su L4S costruisce la sua architettura: un ramo
# dedicato alle bande a 10 m invece di impilarle tutte in una convoluzione.
RISOLUZIONE = {"B1": 60, "B2": 10, "B3": 10, "B4": 10, "B5": 20, "B6": 20,
               "B7": 20, "B8": 10, "B9": 60, "B10": 60, "B11": 20, "B12": 20,
               "slope": 12, "DEM": 12}

# A cosa serve ciascuna, in una riga. Serve a leggere i numeri, non a produrli.
RUOLO = {
    "B1": "aerosol costieri",       "B2": "blu",
    "B3": "verde",                  "B4": "rosso",
    "B5": "bordo rosso 1",          "B6": "bordo rosso 2",
    "B7": "bordo rosso 3",          "B8": "vicino infrarosso",
    "B9": "vapore acqueo",          "B10": "cirri",
    "B11": "SWIR 1, umidita'",      "B12": "SWIR 2, suolo nudo",
    "slope": "pendenza",            "DEM": "quota",
}

MAX_PX = 300      # pixel-frana campionati per patch, con altrettanti di sfondo


def _auc(pos, neg):
    """
    Area sotto la curva ROC di un classificatore a soglia su UN valore.

    Calcolata come statistica di Mann-Whitney sui ranghi: non assume alcuna
    distribuzione, a differenza della distanza di Jeffries-Matusita che le
    vuole gaussiane — e le riflettanze non lo sono.

    0.5 = il canale non distingue niente. Il verso non conta: 0.2 separa
    quanto 0.8, in direzione opposta, quindi si riporta max(a, 1-a).
    """
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return 0.5
    tutti = np.concatenate([pos, neg])
    # ranghi medi, cosi' i pareggi non falsano il conto
    ordine = np.argsort(tutti, kind="mergesort")
    ranghi = np.empty(len(tutti), dtype=np.float64)
    ranghi[ordine] = np.arange(1, len(tutti) + 1)
    _, inv, cnt = np.unique(tutti, return_inverse=True, return_counts=True)
    somma = np.zeros(len(cnt)); np.add.at(somma, inv, ranghi)
    ranghi = (somma / cnt)[inv]
    a = (ranghi[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)
    return max(a, 1.0 - a)


def _campiona(cfg, cache=None):
    """
    Pixel campionati dal solo TRAIN: (P, N, pid).

      P, N   (C, n)  valori dei canali su pixel-frana e su pixel di sfondo,
                     bilanciati per costruzione
      pid    (n,)    indice della patch di provenienza di ciascuna colonna

    `pid` serve alla selezione in avanti: i pixel della stessa patch sono
    fortemente correlati, quindi dividere per PIXEL fra addestramento e
    valutazione darebbe punteggi gonfiati. La divisione va fatta per PATCH.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"  campionamento riletto da {cache}")
        return z["P"], z["N"], z["pid"]

    full = build_dataset(cfg)
    tr, _, _, _ = build_train_val_test_subsets(full, cfg)
    rng = np.random.default_rng(0)
    pos, neg, pid = [], [], []
    for i in tqdm(tr.indices, desc="  raccolta"):
        s = full[i]
        img = s["image"].numpy()                      # (C,H,W)
        m = s["mask"].numpy().reshape(-1) > 0
        x = img.reshape(img.shape[0], -1)             # (C, H*W)
        idx_p = np.flatnonzero(m)
        if idx_p.size == 0:
            continue                                  # patch vuota: nessun positivo
        idx_n = np.flatnonzero(~m)
        k = min(MAX_PX, idx_p.size, idx_n.size)
        pos.append(x[:, rng.choice(idx_p, k, replace=False)])
        neg.append(x[:, rng.choice(idx_n, k, replace=False)])
        pid.append(np.full(k, i, dtype=np.int32))

    P = np.concatenate(pos, axis=1)
    N = np.concatenate(neg, axis=1)
    pid = np.concatenate(pid)
    print(f"  {P.shape[1]:,} pixel-frana e altrettanti di sfondo da "
          f"{len(set(pid.tolist())):,} patch, bilanciati per costruzione")
    if cache:
        np.savez_compressed(cache, P=P, N=N, pid=pid)
    return P, N, pid


def separabilita(cfg, a):
    print(f"\nseparabilita' sul TRAIN, nessun modello")
    P, N, _ = _campiona(cfg, a.cache)

    righe = []
    for c in range(P.shape[0]):
        p, n = P[c].astype(np.float64), N[c].astype(np.float64)
        sd = np.sqrt((p.var() + n.var()) / 2) or 1e-9
        righe.append((NOMI[c] if c < len(NOMI) else f"ch{c}", c,
                      _auc(p, n), (p.mean() - n.mean()) / sd))

    print(f"\n{'banda':>7} {'idx':>4} {'m':>4} {'ruolo':<20} {'AUC':>7} {'d di Cohen':>11}")
    print("  " + "-" * 60)
    for nome, c, auc, d in sorted(righe, key=lambda r: -r[2]):
        print(f"{nome:>7} {c:>4} {RISOLUZIONE.get(nome,0):>4} "
              f"{RUOLO.get(nome,''):<20} {auc:>7.4f} {d:>+11.3f}")
    print("""
  AUC 0.50 = il canale non distingue nulla da solo. Attenzione pero': un
  canale con AUC bassa puo' comunque servire IN COMBINAZIONE — l'NDVI vive
  nella differenza fra B8 e B4, non in nessuno dei due. Questa misura ordina i
  candidati, non decide da sola.
  Il d di Cohen da' il VERSO: negativo = la frana e' piu' scura in quella banda.""")


def combinazione(cfg, a):
    """
    Ridondanza e selezione in avanti: quante bande servono DAVVERO, e quali.

    PERCHE' SERVE. L'AUC valuta ogni banda da sola e non vede le combinazioni;
    la permutazione vede il modello ma SOTTOSTIMA le variabili correlate,
    perche' mescolarne una non fa danno finche' un'altra porta la stessa
    informazione. Su meta' della tabella le due misure lasciano in gara due
    ipotesi opposte — "sostituibile" contro "il modello non sa leggerla" — e
    solo la ridondanza le distingue.

    METODO. Un classificatore lineare per pixel, addestrato su meta' delle
    patch e valutato sull'altra meta', sempre dentro il TRAIN. A ogni passo si
    aggiunge la banda che alza di piu' l'AUC DATE quelle gia' scelte: se B8
    entra nonostante il suo 0.587 da sola, significa che serve in differenza
    con un'altra, cioe' l'NDVI, e non va tolta.

    IL LIMITE, dichiarato. E' un modello LINEARE e PER PIXEL: non vede il
    contesto spaziale, che e' esattamente cio' che rende utile la pendenza.
    Quindi sottostima slope e DEM per costruzione, come gia' l'AUC. Serve a
    ordinare le bande spettrali fra loro, non a decidere sul terreno.
    """
    print(f"\nridondanza e combinazioni sul TRAIN, nessun modello LENTE")
    P, N, pid = _campiona(cfg, a.cache)
    C = P.shape[0]
    nomi = [NOMI[c] if c < len(NOMI) else f"ch{c}" for c in range(C)]

    X = np.concatenate([P, N], axis=1).T.astype(np.float64)   # (n, C)
    y = np.r_[np.ones(P.shape[1]), np.zeros(N.shape[1])]
    g = np.r_[pid, pid]

    # --- 1. correlazione fra bande -------------------------------------
    R = np.corrcoef(X, rowvar=False)
    print("\ncorrelazione fra bande (x100, valore assoluto)")
    print("        " + " ".join(f"{n[:4]:>4}" for n in nomi))
    for i in range(C):
        celle = " ".join(f"{abs(R[i, j]) * 100:>4.0f}" if i != j else "   ."
                         for j in range(C))
        print(f"{nomi[i]:>7} {celle}")

    # Numero EFFETTIVO di assi indipendenti: rapporto di partecipazione degli
    # autovalori. Con 14 bande perfettamente scorrelate darebbe 14, con 14
    # copie della stessa banda darebbe 1. E' la risposta quantitativa a
    # "quante bande servono", prima ancora di sapere quali.
    ev = np.clip(np.linalg.eigvalsh(R), 0, None)
    eff = (ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12)
    q = np.cumsum(np.sort(ev)[::-1]) / ev.sum()
    print(f"\n  assi effettivi: {eff:.2f} su {C} bande")
    print(f"  componenti per il 95% della varianza: "
          f"{int(np.searchsorted(q, 0.95)) + 1}")

    # --- 2. selezione in avanti ----------------------------------------
    # Divisione per PATCH: i pixel della stessa patch sono quasi copie.
    patch = np.unique(g)
    rng = np.random.default_rng(0)
    tenute = set(rng.choice(patch, len(patch) // 2, replace=False).tolist())
    fit = np.array([p in tenute for p in g])
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-9
    Z = (X - mu) / sd

    try:
        from sklearn.linear_model import LogisticRegression

        def punteggio(cols):
            m = LogisticRegression(max_iter=200, C=1.0)
            m.fit(Z[fit][:, cols], y[fit])
            s = Z[~fit][:, cols] @ m.coef_[0]
            return _auc(s[y[~fit] == 1], s[y[~fit] == 0])
        print("\n  classificatore: regressione logistica (sklearn)")
    except ImportError:
        def punteggio(cols):
            # discriminante di Fisher in forma chiusa: w = S^-1 (m1 - m0).
            # L'assunzione gaussiana tocca solo la scelta dei pesi, non la
            # valutazione, che resta l'AUC non parametrica sulla proiezione.
            A, B = Z[fit][y[fit] == 1][:, cols], Z[fit][y[fit] == 0][:, cols]
            S = (np.cov(A, rowvar=False) + np.cov(B, rowvar=False)) / 2
            S = np.atleast_2d(S) + np.eye(len(cols)) * 1e-6
            w = np.linalg.solve(S, np.atleast_1d(A.mean(0) - B.mean(0)))
            s = Z[~fit][:, cols] @ w
            return _auc(s[y[~fit] == 1], s[y[~fit] == 0])
        print("\n  sklearn assente: discriminante di Fisher in forma chiusa")

    scelte, resto, prec = [], list(range(C)), 0.5
    print(f"\n{'passo':>5} {'banda':>7} {'AUC insieme':>12} {'guadagno':>9} "
          f"{'max corr con le scelte':>23}")
    print("  " + "-" * 62)
    for passo in range(1, C + 1):
        val = {c: punteggio(scelte + [c]) for c in resto}
        c = max(val, key=val.get)
        mc = max((abs(R[c, s]) for s in scelte), default=0.0)
        print(f"{passo:>5} {nomi[c]:>7} {val[c]:>12.4f} {val[c] - prec:>+9.4f} "
              f"{mc:>23.2f}")
        scelte.append(c); resto.remove(c); prec = val[c]

    print("""
  Il guadagno crolla dove le bande smettono di aggiungere informazione: le
  righe da li' in poi sono candidate alla rimozione, e la colonna di destra
  dice perche' — correlazione alta significa "ridondante", correlazione bassa
  con guadagno nullo significa "davvero senza informazione".

  Una banda con AUC bassa da sola che entra PRESTO qui serve in combinazione:
  e' il caso che l'analisi per banda singola non poteva vedere.""")


@torch.no_grad()
def permutazione(cfg, a):
    dev = cfg["device"]
    T = cfg["target_size"]
    full = build_dataset(cfg)
    _, va, _, _ = build_train_val_test_subsets(full, cfg)
    # shuffle=True: mescolare con campioni ADIACENTI non basta, le tessere di
    # L4S sono contigue e due indici vicini sono spesso la stessa scena.
    loader = DataLoader(TransformSubset(va, A.Compose([ToTensorV2()])),
                        batch_size=a.batch_size, shuffle=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(0))

    model = build_model(cfg, dev)
    ck = torch.load(a.ckpt, map_location=dev)
    sd = ck.get("model_state_dict", ck)
    inc = model.load_state_dict(sd, strict=False)
    persi = sorted({n for n, q in model.named_parameters()
                    if q.requires_grad}.intersection(inc.missing_keys))
    if persi:
        raise SystemExit(f"{len(persi)} parametri addestrabili non sono nel "
                         f"checkpoint (primo: {persi[0]}). Controlla i flag.")
    model.eval()

    n_ch = None
    conta = {}
    print(f"\npermutazione sul VAL ({len(va.indices)} patch), soglia {a.soglia}")
    for batch in tqdm(loader, desc="  passate"):
        imgs = batch["image"].to(dev)
        masks = batch["mask"].to(dev)
        if n_ch is None:
            n_ch = imgs.shape[1]
            conta = {k: [0.0, 0.0, 0.0] for k in [-1] + list(range(n_ch))}

        for c in [-1] + list(range(n_ch)):
            x = imgs
            if c >= 0:
                x = imgs.clone()
                # roll lungo il batch: ogni campione riceve il canale c di un
                # ALTRO campione. Distrugge la corrispondenza spaziale ma
                # conserva la distribuzione marginale della banda, che e' cio'
                # che distingue una permutazione da un azzeramento.
                x[:, c] = imgs[:, c].roll(1, dims=0)
            p = torch.sigmoid(F.interpolate(
                get_best_mask(*model(F.interpolate(
                    x, (T, T), mode="bilinear", align_corners=False), None)).float(),
                masks.shape[-2:], mode="bilinear", align_corners=False))
            pr = (p > a.soglia).float(); gt = masks.unsqueeze(1).float()
            r = conta[c]
            r[0] += float((pr * gt).sum())
            r[1] += float((pr * (1 - gt)).sum())
            r[2] += float(((1 - pr) * gt).sum())

    def iou(c):
        tp, fp, fn = conta[c]
        return tp / max(tp + fp + fn, 1.0)

    base = iou(-1)
    print(f"\n  micro IoU intatta: {base:.4f}")
    print(f"\n{'banda':>7} {'idx':>4} {'m':>4} {'ruolo':<20} "
          f"{'micro IoU':>10} {'calo':>9}")
    print("  " + "-" * 62)
    for c in sorted(range(n_ch), key=lambda k: iou(k)):
        nome = NOMI[c] if c < len(NOMI) else f"ch{c}"
        print(f"{nome:>7} {c:>4} {RISOLUZIONE.get(nome,0):>4} "
              f"{RUOLO.get(nome,''):<20} {iou(c):>10.4f} {base - iou(c):>+9.4f}")
    print("""
  Calo grande = il modello si appoggia a quella banda.
  Calo NULLO = il modello la ignora: toglierla non costa niente, e liberare
  quei nove canali inizializzati a rumore puo' persino aiutare.
  Calo NEGATIVO = mescolarla MIGLIORA. Sta facendo danno.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["separabilita", "combinazione", "permutazione"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("-d", "--dataset", default="l4s")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--adapter-where", choices=["attn", "mlp", "both"], default=None)
    ap.add_argument("--adapter-dim", type=int, default=None)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--unfreeze-blocks", type=int, default=None)
    ap.add_argument("--soglia", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache", default="campioni_bande.npz",
                    help="i pixel campionati, riusati fra `separabilita` e "
                         "`combinazione` cosi' le due misure girano sugli "
                         "STESSI dati e restano confrontabili")
    a = ap.parse_args()

    cfg = load_config(a.config, dataset=a.dataset, prompt_source="none")
    for k, v in (("adapter_where", a.adapter_where), ("adapter_dim", a.adapter_dim),
                 ("target_size", a.target_size),
                 ("unfreeze_blocks", a.unfreeze_blocks)):
        if v is not None:
            cfg[k] = v

    if a.modo == "separabilita":
        separabilita(cfg, a)
    elif a.modo == "combinazione":
        combinazione(cfg, a)
    else:
        if a.ckpt is None:
            raise SystemExit("--ckpt e' obbligatorio per la permutazione")
        permutazione(cfg, a)
    print()


if __name__ == "__main__":
    main()

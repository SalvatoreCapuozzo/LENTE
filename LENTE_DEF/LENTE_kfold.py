#!/usr/bin/env python
"""
LENTE_kfold.py — la cross-validation a 5 fold, con un solo comando.

Tre modelli, stessa architettura (unf2, fusione entrambi sul terreno, 512 px),
tutti scelti sul micro del val:

    l4s     LENTE singolo su Landslide4Sense        (LENTE_run.py)
    bijie   LENTE singolo su Bijie                  (LENTE_run.py)
    multi   LENTE congiunto L4S + Bijie             (LENTE_multi_v2.py)

Per ogni fold, nell'ordine: addestra i modelli che mancano, poi li misura sul
test del fold con `LENTE_sonde.py soglia` (soglia scelta sul val del fold,
una vista). Alla fine stampa media e deviazione standard sui fold e la
differenza congiunto - singolo fold per fold.

Gli split vengono da `--fold` in build_train_val_test_subsets: il fold 0 e'
lo split di sempre, gli altri ruotano la stessa permutazione. I test dei
fold sono disgiunti.

SI PUO' INTERROMPERE E RILANCIARE. Ogni addestramento finito lascia un file
cv_kfold/<checkpoint>.fatto; ogni misura lascia il suo JSON. Al rilancio:
  - checkpoint con .fatto        -> saltato
  - checkpoint senza .fatto      -> ripreso con --resume (run interrotto)
  - checkpoint assente           -> addestrato da capo
  - JSON della misura presente   -> misura saltata

I due checkpoint del fold 0 gia' addestrati e testati (Bijie singolo e
congiunto) vengono segnati come finiti alla prima esecuzione: sono il fold 0.
L4S singolo scelto sul micro non esiste ancora, quindi anche il suo fold 0 va
addestrato.

USO (da LENTE_SAM_V2)

    python LENTE_kfold.py                        # tutto: 5 fold x 3 modelli
    python LENTE_kfold.py --folds 0,1            # solo alcuni fold
    python LENTE_kfold.py --solo-riepilogo       # solo la tabella finale
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

CARTELLA = "cv_kfold"
ARCH = ["--unfreeze-blocks", "2", "--fusione", "entrambi", "--fusione-aux", "terreno"]

MODELLI = {
    "l4s": {
        "train": ["LENTE_run.py", "-d", "l4s", "-m", "train", "--prompt", "none", *ARCH,
                  "--batch-size", "4", "--accum-steps", "2", "--select-on", "micro"],
        "ckpt": "best_lente_l4s_free_selmicro_unf2_entrambi24{f}.pth",
        "domini": ["l4s"],
    },
    "bijie": {
        "train": ["LENTE_run.py", "-d", "bijie", "-m", "train", "--prompt", "none", *ARCH,
                  "--batch-size", "4", "--accum-steps", "2", "--select-on", "micro",
                  "--target-size", "512"],
        "ckpt": "best_lente_bijie_free_selmicro_unf2_entrambi24{f}.pth",
        "domini": ["bijie"],
    },
    "multi": {
        "train": ["LENTE_multi_v2.py", *ARCH],
        "ckpt": "best_lente_multi_v3_unf2_entrambi_512px_selmedia{f}.pth",
        "domini": ["l4s", "bijie"],
    },
}

# Addestrati e testati prima che esistesse questo script, sullo split del fold 0.
GIA_FINITI = ["best_lente_bijie_free_selmicro_unf2_entrambi24.pth",
              "best_lente_multi_v3_unf2_entrambi_512px_selmedia.pth"]


def nome_ckpt(modello, fold):
    return MODELLI[modello]["ckpt"].format(f=f"_fold{fold}" if fold else "")


def segno(ckpt):
    return os.path.join(CARTELLA, ckpt + ".fatto")


def file_test(modello, dominio, fold):
    return os.path.join(CARTELLA, f"test_{modello}_{dominio}_fold{fold}.json")


def lancia(cmd, env):
    print("\n" + "=" * 100 + "\n  " + " ".join(cmd) + "\n" + "=" * 100, flush=True)
    return subprocess.run(cmd, env=env).returncode


def addestra(py, modello, fold, env):
    ckpt = nome_ckpt(modello, fold)
    if os.path.exists(segno(ckpt)):
        print(f"  [fold {fold}] {modello}: {ckpt} gia' finito, salto")
        return True
    cmd = [py, *MODELLI[modello]["train"], "--fold", str(fold), "--checkpoint", ckpt]
    if os.path.exists(ckpt):
        print(f"  [fold {fold}] {modello}: {ckpt} esiste ma non e' segnato finito -> --resume")
        cmd.append("--resume")
    if lancia(cmd, env) != 0 or not os.path.exists(ckpt):
        print(f"  [fold {fold}] {modello}: ADDESTRAMENTO FALLITO")
        return False
    open(segno(ckpt), "w").close()
    return True


def misura(py, modello, fold, env):
    ckpt = nome_ckpt(modello, fold)
    ok = True
    for dominio in MODELLI[modello]["domini"]:
        out = file_test(modello, dominio, fold)
        if os.path.exists(out):
            continue
        cmd = [py, "LENTE_sonde.py", "soglia", "-d", dominio, "--target-size", "512", *ARCH,
               "--tta", "1", "--scegli-su", "micro", "--soglia-max", "0.95",
               "--ckpt", ckpt, "--fold", str(fold), "--salva", out]
        if modello == "multi":
            cmd.append("--multi")
        if lancia(cmd, env) != 0 or not os.path.exists(out):
            print(f"  [fold {fold}] misura {modello} su {dominio}: FALLITA")
            ok = False
    return ok


def _ms(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return "       -       "
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{v.mean():.4f} ± {sd:.4f}"


def riepilogo(folds):
    righe = []
    def p(s=""):
        print(s)
        righe.append(s)

    p("\n" + "=" * 100)
    p("  CROSS-VALIDATION — test di ogni fold, soglia scelta sul val del fold, una vista")
    p("=" * 100)
    tutto = {}
    for dominio, singolo in (("l4s", "l4s"), ("bijie", "bijie")):
        r = {}
        for m in (singolo, "multi"):
            for f in folds:
                fn = file_test(m, dominio, f)
                if os.path.exists(fn):
                    r[(m, f)] = json.load(open(fn))
        p(f"\n  {dominio.upper()}")
        p(f"  {'fold':>5} {'modello':>9} {'soglia':>7} {'micro':>8} {'micro 0.5':>10} "
          f"{'per immagine':>13} {'banale':>8}")
        for f in folds:
            for m in (singolo, "multi"):
                if (m, f) in r:
                    j = r[(m, f)]
                    p(f"  {f:>5} {m:>9} {j['soglia_val']:>7.2f} {j['test']['micro_iou']:>8.4f} "
                      f"{j['test_05']['micro_iou']:>10.4f} {j['test']['per_immagine']:>13.4f} "
                      f"{j['test']['banale']:>8.4f}")
        # media di ogni modello sui fold che ha, anche senza l'altro
        p("")
        for m, nome_m in ((singolo, "singolo"), ("multi", "congiunto")):
            fm = [f for f in folds if (m, f) in r]
            if not fm:
                continue
            p(f"  {nome_m:>10} su {len(fm)} fold {fm}:   "
              f"micro {_ms([r[(m, f)]['test']['micro_iou'] for f in fm])}   "
              f"micro 0.5 {_ms([r[(m, f)]['test_05']['micro_iou'] for f in fm])}   "
              f"per immagine {_ms([r[(m, f)]['test']['per_immagine'] for f in fm])}")
        # differenza appaiata, solo sui fold dove ci sono entrambi
        comuni = [f for f in folds if (singolo, f) in r and ("multi", f) in r]
        p(f"\n  fold con entrambi i modelli: {comuni}")
        sintesi = {}
        if not comuni:
            tutto[dominio] = {"fold": []}
            continue
        for chiave, nome, campo in (("micro", "micro (soglia val)", "test"),
                                    ("micro_05", "micro (soglia 0.5)", "test_05"),
                                    ("per_immagine", "per immagine (soglia val)", "test")):
            k = "per_immagine" if chiave == "per_immagine" else "micro_iou"
            a = [r[(singolo, f)][campo][k] for f in comuni]
            b = [r[("multi", f)][campo][k] for f in comuni]
            d = [y - x for x, y in zip(a, b)]
            sintesi[chiave] = {"singolo": a, "multi": b, "diff": d}
            p(f"  {nome:>26}   singolo {_ms(a)}   congiunto {_ms(b)}   "
              f"congiunto - singolo {_ms(d)}   meglio in {sum(x > 0 for x in d)}/{len(d)} fold")
        tutto[dominio] = {"fold": comuni, **sintesi}
    p("""
  Lettura. La deviazione standard misura quanto il risultato cambia con lo
  split: e' la variabilita' che un solo test non mostra. La colonna della
  differenza e' appaiata (stesso fold, stesso test): se la media e' piccola
  rispetto alla sua deviazione standard, o il congiunto vince solo in alcuni
  fold, i due modelli vanno scritti come alla pari. Con 5 fold non si fa un
  test di significativita' credibile: si riportano media, deviazione e fold
  vinti. La soglia e' scelta sul val, i benchmark pubblicati usano 0.5 fissa:
  per questo c'e' anche la riga a 0.5.""")
    with open(os.path.join(CARTELLA, "riepilogo.json"), "w") as fh:
        json.dump(tutto, fh, indent=1)
    with open(os.path.join(CARTELLA, "riepilogo.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(righe) + "\n")
    print(f"\n  salvato in {CARTELLA}/riepilogo.txt e riepilogo.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--modelli", default="l4s,bijie,multi")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES per tutti i processi")
    ap.add_argument("--solo-riepilogo", action="store_true")
    a = ap.parse_args()

    folds = [int(x) for x in a.folds.split(",")]
    modelli = a.modelli.split(",")
    for m in modelli:
        if m not in MODELLI:
            raise SystemExit(f"modello sconosciuto: {m} (scegli fra {', '.join(MODELLI)})")
    os.makedirs(CARTELLA, exist_ok=True)

    if not a.solo_riepilogo:
        for ckpt in GIA_FINITI:
            if os.path.exists(ckpt) and not os.path.exists(segno(ckpt)):
                open(segno(ckpt), "w").close()
                print(f"  {ckpt}: gia' addestrato prima di questo script, segnato come fold 0")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
        falliti = []
        for f in folds:
            for m in modelli:
                if not addestra(sys.executable, m, f, env):
                    falliti.append(f"addestramento {m} fold {f}")
                    continue
                if not misura(sys.executable, m, f, env):
                    falliti.append(f"misura {m} fold {f}")
            riepilogo(folds)          # aggiornato a ogni fold: i parziali si leggono subito
        if falliti:
            print("\n  NON COMPLETATI (rilancia lo stesso comando dopo aver guardato l'errore):")
            for x in falliti:
                print(f"    - {x}")
    else:
        riepilogo(folds)


if __name__ == "__main__":
    main()

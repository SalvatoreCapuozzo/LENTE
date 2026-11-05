#!/usr/bin/env python
"""
ordina_prove.py — raccoglie ogni prova in una cartella sua.

Un run lascia dietro tre file con lo stesso nome di base:

    best_lente_l4s_..._1024px.pth            i pesi
    history_best_lente_l4s_..._1024px.json   la curva per epoca
    results_best_lente_l4s_..._1024px.json   il risultato sul test

Qui vengono spostati insieme in  prove/<nome della prova>/ , cosi' che a un
checkpoint corrisponda sempre la sua storia e non si debba piu' indovinare
quale JSON appartenga a quale .pth.

DUE CAUTELE, entrambe necessarie.

  * NON SPOSTA NIENTE senza --esegui. Di default stampa soltanto cosa farebbe.
  * SALTA i file scritti di recente, perche' un addestramento in corso
    riscrive il proprio checkpoint a ogni epoca e spostarlo lo romperebbe a
    meta'. La soglia e' --minuti, di default 90: piu' di un'epoca lunga.

    python ordina_prove.py                 # mostra e basta
    python ordina_prove.py --esegui        # sposta davvero
    python ordina_prove.py --esegui --minuti 200
"""

import argparse
import glob
import os
import shutil
import time

PREFISSI = ("history_", "results_", "curva_", "figura_")


def nome_prova(base):
    """
    Dal nome del checkpoint al nome della cartella.

    Si toglie solo la parte comune a tutti — `best_lente_` — e si tiene il
    resto, che e' gia' la descrizione della configurazione: dataset, regime di
    prompt, adapter, criterio di selezione, scongelamento, fusione,
    risoluzione. Riscriverla in altro modo significherebbe reinventare una
    convenzione che i nomi dei file gia' contengono.
    """
    n = base
    for p in ("best_lente_", "best_"):
        if n.startswith(p):
            n = n[len(p):]
            break
    return n or "senza_nome"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dove", default=".", help="cartella da riordinare")
    ap.add_argument("--dest", default="prove", help="cartella di destinazione")
    ap.add_argument("--minuti", type=float, default=90.0,
                    help="salta i file toccati da meno di questi minuti: e' la "
                         "protezione contro un addestramento in corso")
    ap.add_argument("--esegui", action="store_true",
                    help="senza questo non sposta niente, mostra soltanto")
    a = ap.parse_args()

    ora = time.time()
    limite = a.minuti * 60

    ckpt = sorted(glob.glob(os.path.join(a.dove, "best_*.pth")))
    if not ckpt:
        raise SystemExit(f"nessun best_*.pth in {os.path.abspath(a.dove)}")

    gruppi, vivi, orfani = {}, [], []
    usati = set()

    for c in ckpt:
        base = os.path.splitext(os.path.basename(c))[0]
        compagni = [c]
        for p in PREFISSI:
            for e in (".json", ".png", ".csv"):
                q = os.path.join(a.dove, f"{p}{base}{e}")
                if os.path.exists(q):
                    compagni.append(q)
        # Un solo file recente basta a fermare l'intero gruppo: spostare la
        # storia lasciando indietro i pesi, o viceversa, e' peggio che non
        # spostare niente.
        eta = min(ora - os.path.getmtime(f) for f in compagni)
        if eta < limite:
            vivi.append((base, eta / 60.0, len(compagni)))
            continue
        gruppi[base] = compagni
        usati.update(compagni)

    # JSON che non appartengono a nessun checkpoint: restano dove sono, ma
    # vanno segnalati, perche' di solito significa un .pth cancellato o
    # rinominato a mano.
    for p in PREFISSI:
        for q in sorted(glob.glob(os.path.join(a.dove, f"{p}*.json"))):
            if q not in usati and all(q not in v for v in gruppi.values()):
                atteso = os.path.basename(q)[len(p):].rsplit(".", 1)[0] + ".pth"
                if not os.path.exists(os.path.join(a.dove, atteso)):
                    orfani.append((os.path.basename(q), atteso))

    print(f"\n{'PROVE DA SPOSTARE' if not a.esegui else 'SPOSTAMENTO'}"
          f"   ({os.path.abspath(a.dove)} -> {a.dest}/)")
    print("-" * 78)
    tot = 0
    for base, compagni in gruppi.items():
        cart = os.path.join(a.dove, a.dest, nome_prova(base))
        mb = sum(os.path.getsize(f) for f in compagni) / 1e6
        print(f"  {nome_prova(base):<52} {len(compagni)} file  {mb:>7.1f} MB")
        if a.esegui:
            os.makedirs(cart, exist_ok=True)
            for f in compagni:
                d = os.path.join(cart, os.path.basename(f))
                if os.path.exists(d):
                    print(f"      esiste gia', saltato: {os.path.basename(f)}")
                    continue
                shutil.move(f, d)
        tot += len(compagni)

    if vivi:
        print(f"\n  SALTATI perche' scritti da poco "
              f"(soglia {a.minuti:.0f} minuti):")
        for base, minuti, n in vivi:
            print(f"    {nome_prova(base):<52} {minuti:>6.1f} min fa, {n} file")
        print("    Se e' un addestramento in corso, e' giusto cosi'. Se e' finito,")
        print("    rilancia fra un po' oppure alza --minuti.")

    if orfani:
        print(f"\n  JSON senza checkpoint, lasciati dove sono:")
        for j, atteso in orfani:
            print(f"    {j}   (manca {atteso})")

    print("-" * 78)
    if a.esegui:
        print(f"  spostati {tot} file in {len(gruppi)} cartelle\n")
    else:
        print(f"  {tot} file in {len(gruppi)} cartelle. "
              f"NIENTE E' STATO SPOSTATO: rilancia con --esegui\n")
        print("  Nota: i comandi di test cercano il checkpoint nella cartella")
        print("  corrente. Dopo lo spostamento va passato il percorso completo,")
        print(f"  per esempio  --ckpt {a.dest}/<prova>/<nome>.pth\n")


if __name__ == "__main__":
    main()

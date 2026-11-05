"""
Stato di un run L4S dalla sua storia su disco. Funziona anche mentre gira:
il JSON viene riscritto a ogni epoca, in modo atomico.

    python stato_run.py best_lente_l4s_free_ad128both_selmicro_unf2_1024px.pth
    python stato_run.py                     # il piu' recente

Il checkpoint salvato E' gia' quello dell'epoca migliore secondo `select_on`.
Questo script serve a sapere QUALE epoca sia, con che numeri, e se il run sta
ancora migliorando o sta solo consumando la pazienza.
"""

import glob
import json
import os
import sys

CRITERI = {"val_micro_iou": "micro IoU", "val_iou_frane": "frane",
           "val_iou": "macro totale"}


def main():
    # Secondo argomento: il criterio, per i run vecchi il cui history non lo
    # registra (prima che _jsonable lo includesse). Senza, si ricade su
    # "totale" e viene indicata l'epoca sbagliata come contenuto del .pth.
    forzato = None
    argv = [a for a in sys.argv[1:]]
    for c in ("micro", "frane", "totale"):
        if c in argv:
            forzato = c; argv.remove(c)
    arg = argv[0] if argv else None
    if arg and arg.endswith(".json"):
        path = arg
    elif arg:
        base = os.path.splitext(os.path.basename(arg))[0]
        path = f"history_{base}.json"
    else:
        cand = sorted(glob.glob("history_*.json"), key=os.path.getmtime)
        if not cand:
            raise SystemExit("nessun history_*.json in questa cartella")
        path = cand[-1]

    r = json.load(open(path))
    ep = r["epoche"]
    if not ep:
        raise SystemExit(f"{path}: nessuna epoca registrata")
    cfg = r.get("config", {})
    # Il criterio VERO e' quello che il training ha registrato a ogni epoca.
    # `select_on` del config puo' mentire: con --solo-positive la selezione
    # e' forzata su `frane` qualunque cosa dica, e qui si riportava l'epoca
    # migliore secondo un criterio che il training non ha mai usato.
    registrato = ep[-1].get("criterio")
    if forzato:
        sel = forzato
        chiave = {"micro": "val_micro_iou", "frane": "val_iou_frane"}.get(sel, "val_iou")
    elif registrato in CRITERI:
        chiave = registrato
        sel = {"val_micro_iou": "micro", "val_iou_frane": "frane"}.get(chiave, "totale")
    else:
        sel = cfg.get("select_on")
        if sel is None:
            sel = "totale"
            print()
            print("  NB: questo history non registra il criterio (run precedente "
                  "alla correzione).")
            print("      Se il run usava --select-on micro, rilancia aggiungendo:"
                  "  micro")
        chiave = {"micro": "val_micro_iou", "frane": "val_iou_frane"}.get(sel, "val_iou")

    print(f"\n{r['checkpoint']}")
    print(f"  criterio di selezione: {sel}   ({chiave})")
    print(f"  epoche registrate: {len(ep)}   ultima: {ep[-1]['epoca']}"
          f"   stop: {r.get('stop', 'in corso')}")

    # --- migliore secondo OGNI criterio, non solo quello attivo -------------
    print(f"\n{'criterio':>14} {'epoca':>6} {'micro IoU':>11} {'micro F1':>10} "
          f"{'frane':>8} {'vuote':>8}")
    for k, nome in CRITERI.items():
        vivi = [e for e in ep if k in e]
        if not vivi:
            continue
        b = max(vivi, key=lambda e: e[k])
        marca = "  <- attivo" if k == chiave else ""
        print(f"{nome:>14} {b['epoca']:>6} {b.get('val_micro_iou', 0):>11.4f} "
              f"{b.get('val_micro_f1', 0):>10.4f} {b.get('val_iou_frane', 0):>8.4f} "
              f"{b.get('val_iou_vuote', 0):>8.4f}{marca}")

    # --- il checkpoint su disco corrisponde all'epoca migliore del criterio --
    # Le epoche che hanno SCRITTO il file sono marcate `record`: l'ultima e'
    # quella dentro il .pth, senza dover ricostruire il criterio.
    salvate = [e for e in ep if e.get("record")]
    if salvate and not forzato:
        best = salvate[-1]
    else:
        vivi = [e for e in ep if chiave in e]
        best = max(vivi, key=lambda e: e[chiave])
    print(f"\n  il file .pth contiene l'epoca {best['epoca']}, "
          f"{CRITERI[chiave]} = {best.get(chiave, float('nan')):.4f}")
    da_allora = ep[-1]["epoca"] - best["epoca"]
    pat = int(cfg.get("patience", 15))
    print(f"  epoche senza miglioramento: {da_allora}/{pat}"
          f"{'   EARLY STOP IMMINENTE' if da_allora >= pat - 3 else ''}")

    # --- sovradattamento: il divario train/val ------------------------------
    # Su tutti i run precedenti era NULLO. Se si apre, la capacita' aggiunta
    # sta memorizzando invece di generalizzare, ed e' il segnale per fermarsi.
    print(f"\n{'ep':>4} {'train IoU':>10} {'val IoU':>9} {'divario':>9} "
          f"{'micro':>8} {'frane':>8}")
    passo = max(1, len(ep) // 10)
    for e in ep[::passo] + ([ep[-1]] if len(ep) % passo else []):
        d = e["train_iou"] - e["val_iou"]
        print(f"{e['epoca']:>4} {e['train_iou']:>10.4f} {e['val_iou']:>9.4f} "
              f"{d:>+9.4f} {e.get('val_micro_iou', 0):>8.4f} "
              f"{e.get('val_iou_frane', 0):>8.4f}")

    ultimi = ep[-5:]
    if len(ultimi) >= 2:
        dm = sum(e["train_iou"] - e["val_iou"] for e in ultimi) / len(ultimi)
        print(f"\n  divario medio sulle ultime {len(ultimi)} epoche: {dm:+.4f}")
        # Il segno CONTA. Un divario negativo (val migliore di train) e' la
        # norma qui: il train e' misurato sotto augmentation e il val no,
        # quindi il train e' il compito piu' difficile dei due. Solo un
        # divario POSITIVO e ampio e' memorizzazione. Usare abs() qui
        # segnalava come sovradattamento il caso piu' sano che esista.
        if dm < 0.05:
            esito = ("nessun sovradattamento, c'e' margine per piu' capacita'"
                     if dm > -0.03 else
                     "val piu' facile del train (augmentation): normale, "
                     "e c'e' margine per piu' capacita'")
        else:
            esito = ("APERTO: il modello sta memorizzando, aggiungere "
                     "capacita' peggiorerebbe")
        print("  " + esito)
    print()


if __name__ == "__main__":
    main()

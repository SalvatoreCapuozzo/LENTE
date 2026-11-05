#!/usr/bin/env python
"""
LENTE_run.py — entry point.
============================
Single entry point to decide HOW to run training
(or test / visualization) on any of the configured datasets.

Two usage modes:

1) Interactive (no arguments) — asks for dataset and mode via menu:
       python LENTE_run.py

2) Command line (for scripts/cluster):
       python LENTE_run.py --dataset bijie --mode train
       python LENTE_run.py --dataset l4s   --mode test
       python LENTE_run.py --dataset bijie --mode visualize
       python LENTE_run.py --dataset bijie --mode plot      # plot history (train/val/test)
       python LENTE_run.py --dataset bijie --mode ablation  # ablazione (no DFDS / no DFDS+adapter)
       python LENTE_run.py --dataset bijie --mode cv        # 5-fold cross-validation
       python LENTE_run.py --dataset bijie --mode all     # train -> test -> visualize -> plot -> ablation

Quick config overrides (optional):
       python LENTE_run.py -d bijie -m train --epochs 50 --batch-size 4 --lr 5e-5

Available datasets are those defined under `datasets:` in config.yaml.
"""

import os
import sys
import argparse
import yaml

from LENTE_utils import load_config
from LENTE_train import train
from LENTE_test import test
from LENTE_visualize_results import visualize_test_set
from LENTE_plot_history import plot_history
from LENTE_ablation import run_ablation, VARIANTS
from LENTE_cv import cross_validate

MODES = ["train", "test", "visualize", "plot", "ablation", "cv",
         "boxhead", "boxhead_ema", "boxhead_test", "boxhead_ema_test", "all"]

# Modi in cui il prompt e' un box APPRESO invece che preso dalla ground truth.
# valore = usa l'EMA.
BOXHEAD_MODES = {"boxhead": False, "boxhead_ema": True}
# Test dei corrispondenti: derivano il nome del checkpoint esattamente come il
# training, cosi' i due combaciano sempre senza toccare il config a mano.
BOXHEAD_TEST_MODES = {"boxhead_test": False, "boxhead_ema_test": True}


def prepare_boxhead_cfg(cfg, use_ema):
    """
    Adatta il config ai modi boxhead: il prompt non e' piu' ne' 'gt_box' ne'
    'none' ma un box predetto, quindi checkpoint e history prendono un nome
    dedicato. Senza questo i tre regimi si sovrascriverebbero a vicenda e
    load_checkpoint riprenderebbe in silenzio dall'esperimento sbagliato.
    """
    tag = "boxhead_ema" if use_ema else "boxhead"
    cfg = dict(cfg)
    # Il canale di prompt entra nel nome solo se non e' quello storico ("box"),
    # cosi' i run gia' fatti mantengono i loro file e i tre regimi non si
    # sovrascrivono fra loro.
    mode = cfg.get("boxhead_prompt", "box")
    if mode != "box":
        tag = f"{tag}_{mode}"
    cfg["prompt_source"] = "learned_box"
    for key, default in (("checkpoint_name", "best_lente.pth"),
                         ("history_file", "history_lente.json")):
        base, ext = os.path.splitext(cfg.get(key, default))
        # toglie l'eventuale suffisso di regime gia' applicato da load_config
        for old in ("_free", "_gtbox", "_imgbox"):
            if base.endswith(old):
                base = base[: -len(old)]
        cfg[key] = f"{base}_{tag}{ext}"
    return cfg


def available_datasets(config_path="config.yaml"):
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    names = list(raw.get("datasets", {}).keys())
    # "multi" = training congiunto L4S+Bijie (blocco `multi:` separato).
    if "multi" in raw:
        names.append("multi")
    return names


def _menu(title, options):
    options = list(options) + ["exit"]
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        choice = input("Choice > ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            selected = options[int(choice) - 1]
        elif choice in options:
            selected = choice
        else:
            print("  Invalid choice, try again.")
            continue
        if selected == "exit":
            print("Exiting.")
            sys.exit(0)
        return selected


def interactive(config_path):
    datasets = available_datasets(config_path)
    dataset = _menu("Which dataset?", datasets)
    mode    = _menu("What do you want to do?", MODES)
    return dataset, mode


def apply_overrides(cfg, args):
    """Overrides some config fields with optional CLI flags."""
    if args.epochs is not None:       cfg["epochs"] = args.epochs
    if args.batch_size is not None:   cfg["batch_size"] = args.batch_size
    if args.lr is not None:           cfg["learning_rate"] = args.lr
    if args.checkpoint is not None:   cfg["checkpoint_name"] = args.checkpoint
    if args.adapter_where is not None: cfg["adapter_where"] = args.adapter_where
    if args.adapter_dim is not None:   cfg["adapter_dim"] = args.adapter_dim
    if args.select_on is not None:     cfg["select_on"] = args.select_on
    if args.target_size is not None:   cfg["target_size"] = args.target_size
    if args.focal_gamma is not None:   cfg["loss_focal_gamma"] = args.focal_gamma
    if args.accum_steps is not None:  cfg["accum_steps"] = args.accum_steps
    if args.unfreeze_blocks is not None: cfg["unfreeze_blocks"] = args.unfreeze_blocks
    if args.encoder is not None:       cfg["encoder"] = args.encoder
    if args.unet_width is not None:    cfg["unet_width"] = args.unet_width
    if args.unet_lr is not None:       cfg["unet_lr"] = args.unet_lr
    if args.encoder_lr_mult is not None: cfg["encoder_lr_mult"] = args.encoder_lr_mult
    if args.tec_variante is not None:    cfg["tec_variante"] = args.tec_variante
    if args.tec_compression is not None: cfg["tec_compression"] = args.tec_compression
    if args.tec_size is not None:        cfg["tec_size"] = args.tec_size
    if args.tec_unfreeze:                cfg["tec_frozen"] = False
    if args.prithvi_size is not None:  cfg["prithvi_size"] = args.prithvi_size
    if args.prithvi_model is not None: cfg["prithvi_model"] = args.prithvi_model
    if args.prithvi_adapter_dim is not None:
        cfg["prithvi_adapter_dim"] = args.prithvi_adapter_dim
    if args.prithvi_lora is not None:  cfg["prithvi_lora"] = args.prithvi_lora
    if args.prithvi_unfreeze:          cfg["prithvi_frozen"] = False
    if args.fusione is not None:      cfg["fusione"] = args.fusione
    if args.fusione_aux is not None:  cfg["fusione_aux"] = args.fusione_aux
    if args.fusione_width is not None: cfg["fusione_width"] = args.fusione_width
    if args.fusione_lr is not None:    cfg["fusione_lr"] = args.fusione_lr
    if args.gsrm:                   cfg["gsrm"] = True
    if args.gsrm_lr is not None:    cfg["gsrm_lr"] = args.gsrm_lr
    if args.bands is not None:
        cfg["optical_bands"] = [int(x) for x in args.bands.replace(" ", "").split(",")]
        # n_optical viene DERIVATO da optical_bands dentro load_config: se non
        # lo si ricalcola qui, il patch embed viene costruito con il numero di
        # canali vecchio e il modello legge oltre la fine dello stack.
        cfg["n_optical"] = len(cfg["optical_bands"])
    # I checkpoint con posizionamenti diversi NON sono intercambiabili (12
    # adapter contro 24), e caricarne uno sbagliato con --resume darebbe un
    # errore di shape lontano dalla causa. Il tag nel nome li tiene separati.
    marche = []
    if cfg.get("adapter_where", "attn") != "attn":
        marche.append(f"_ad{cfg['adapter_dim']}{cfg['adapter_where']}")
    if cfg.get("select_on", "totale") == "micro":
        marche.append("_selmicro")
    if float(cfg.get("loss_focal_gamma", 0.0)) > 0:
        marche.append(f"_focal{cfg['loss_focal_gamma']:g}")
    if int(cfg.get("unfreeze_blocks", 0)) != 0:
        marche.append(f"_unf{cfg['unfreeze_blocks']}")
    if cfg.get("encoder", "hiera") == "unet":
        marche.append(f"_unet{cfg.get('unet_width', 48)}")
    elif cfg.get("encoder", "hiera") == "prithvi":
        marche.append(f"_prithvi{cfg.get('prithvi_size', 512)}")
        if int(cfg.get("prithvi_lora", 0)) > 0:
            marche.append(f"_lora{cfg['prithvi_lora']}")
        if int(cfg.get("prithvi_adapter_dim", 0)) > 0:
            marche.append(f"_pad{cfg['prithvi_adapter_dim']}")
    elif cfg.get("encoder", "hiera") == "terracodec":
        marche.append(f"_tec{cfg.get('tec_variante', 'fp')}"
                      f"{cfg.get('tec_compression', 10)}")
    if args.bands is not None:
        marche.append("_b" + "-".join(str(b) for b in cfg["optical_bands"]))
    if cfg.get("gsrm", False):
        marche.append("_gsrm")
    if cfg.get("fusione", "none") != "none":
        if cfg.get("fusione_aux", "terreno") == "prithvi":
            marche.append(f"_{cfg['fusione']}prith"
                          f"{cfg.get('prithvi_size', 512)}")
        else:
            marche.append(f"_{cfg['fusione']}{cfg.get('fusione_width', 24)}")
    if args.target_size is not None and args.target_size != 512:
        marche.append(f"_{args.target_size}px")
    if marche:
        base, ext = os.path.splitext(cfg["checkpoint_name"])
        marca = "".join(marche)
        if not base.endswith(marca):
            cfg["checkpoint_name"] = f"{base}{marca}{ext}"
            hb, he = os.path.splitext(cfg["history_file"])
            cfg["history_file"] = f"{hb}{marca}{he}"
    if args.viz_samples is not None:  cfg["viz_num_samples"] = args.viz_samples
    cfg["resume"] = bool(getattr(args, "resume", False))
    return cfg


def run_mode(cfg, mode, folds=5):
    if mode in BOXHEAD_MODES:
        # percorso dedicato: non rientra in "all", perche' addestra un modello
        # diverso (con la testa di box) e non e' un passo della stessa pipeline.
        from LENTE_boxhead import train_boxhead
        use_ema = BOXHEAD_MODES[mode]
        train_boxhead(prepare_boxhead_cfg(cfg, use_ema), use_ema=use_ema)
        return
    if mode in BOXHEAD_TEST_MODES:
        # NON si puo' usare il modo "test" normale: quello costruisce un LENTE
        # senza testa di box e carica con strict=False, scartando in silenzio
        # le chiavi box_head.* e valutando un modello monco in regime diverso.
        from LENTE_boxhead import test_boxhead
        use_ema = BOXHEAD_TEST_MODES[mode]
        test_boxhead(prepare_boxhead_cfg(cfg, use_ema), use_ema=use_ema)
        return
    if mode in ("train", "all"):
        train(cfg)
    if mode in ("test", "all"):
        test(cfg)
    if mode in ("visualize", "all"):
        visualize_test_set(cfg)
    if mode in ("plot", "all"):
        plot_history(cfg["history_file"])
    if mode in ("ablation", "all"):
        run_ablation(cfg, list(VARIANTS.keys()))
    if mode == "cv":   # esclusa da "all": allena K modelli da zero
        cross_validate(cfg, k=folds)


def main():
    ap = argparse.ArgumentParser(description="LENTE launcher (train/test/visualize/ablation on configurable datasets)")
    ap.add_argument("-d", "--dataset", default=None, help="dataset name (see config.yaml). If absent: interactive menu")
    ap.add_argument("-m", "--mode", default=None, choices=MODES, help="train | test | visualize | plot | ablation | all")
    ap.add_argument("-c", "--config", default="config.yaml", help="config path (default: config.yaml)")
    # optional overrides
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--adapter-where", choices=["attn", "mlp", "both"], default=None,
                    help="dove agganciare gli Adapter: attn (storico), mlp (Pfeiffer), "
                         "both (Houlsby). Hiera-tiny ha 12 blocchi: attn=12 punti, both=24")
    ap.add_argument("--adapter-dim", type=int, default=None,
                    help="bottleneck degli Adapter; 0 li disattiva")
    ap.add_argument("--encoder",
                    choices=["hiera", "unet", "prithvi", "terracodec"],
                    default=None,
                    help="hiera = encoder di SAM2 (default). unet = percorso "
                         "contrattivo a U-Net addestrato da zero, stesso decoder")
    ap.add_argument("--unet-lr", type=float, default=None,
                    help="learning rate dell'encoder U-Net (default 3e-4). Il "
                         "learning_rate del config e' 1e-5 e serve a pesi "
                         "preaddestrati: su un encoder nuovo non muove nulla")
    ap.add_argument("--unet-width", type=int, default=None,
                    help="larghezza base dell'encoder U-Net (default 48)")
    ap.add_argument("--unfreeze-blocks", type=int, default=None,
                    help="quanti blocchi FINALI di Hiera scongelare (dei 12). "
                         "0 = nessuno (default), -1 = tutti. Va con --encoder-lr-mult")
    ap.add_argument("--encoder-lr-mult", type=float, default=None,
                    help="moltiplicatore del learning rate per l'encoder scongelato "
                         "(default 0.1)")
    ap.add_argument("--tec-variante", type=str, default=None,
                    help="variante di TerraCodec: fp (accetta qualunque "
                         "dimensione, default), elic, tt, flex. Le ultime tre "
                         "possono pretendere 256x256.")
    ap.add_argument("--tec-compression", type=int, default=None,
                    help="livello di compressione del codec (default 10). Piu' "
                         "alto = latente piu' povero: e' la manopola che decide "
                         "quanta informazione sopravvive.")
    ap.add_argument("--tec-size", type=int, default=None,
                    help="lato dato a TerraCodec. 0 (default) = la stessa "
                         "risoluzione del decoder, che si puo' permettere "
                         "perche' e' una rete convoluzionale.")
    ap.add_argument("--tec-unfreeze", action="store_true",
                    help="rende addestrabile anche il codec")
    ap.add_argument("--prithvi-size", type=int, default=None,
                    help="lato dell'ingresso dato a Prithvi (default 512). "
                         "L'attenzione e' quadratica nei token: a 1024 pixel "
                         "sono 4096 token contro i 196 su cui e' addestrato, e "
                         "su una 3090 non ci sta. La sua mappa viene poi "
                         "riportata alla griglia che il decoder si aspetta.")
    ap.add_argument("--prithvi-model", type=str, default=None,
                    help="identificativo su HuggingFace (default il 300M; il "
                         "600M nel loro stesso studio va PEGGIO su L4S)")
    ap.add_argument("--prithvi-adapter-dim", type=int, default=None,
                    help="Adapter dentro Prithvi. ViT-L e' largo 1024 su 24 "
                         "blocchi: a dim 128 su attenzione e MLP sono 12,6 "
                         "milioni di parametri. Meglio 16-32.")
    ap.add_argument("--prithvi-lora", type=int, default=None,
                    help="rango LoRA su qkv e proj di Prithvi. E' la via che "
                         "il loro paper ha usato su L4S: a rango 8 costa 0,8 "
                         "milioni di parametri.")
    ap.add_argument("--prithvi-unfreeze", action="store_true",
                    help="rende addestrabile anche il backbone Prithvi")
    ap.add_argument("--fusione", choices=["none", "mgem", "dgfm", "entrambi"],
                    default=None,
                    help="fusione a due stream in stile TriGEFNet: un encoder "
                         "ausiliario sul terreno accanto a Hiera, poi MGEM "
                         "(arricchimento reciproco) e/o DGFM (fusione guidata "
                         "dall'ottico). La DSDF resta attiva.")
    ap.add_argument("--fusione-aux", choices=["terreno", "prithvi"],
                    default=None,
                    help="chi fa da stream ausiliario. 'terreno' (default) e' "
                         "una U-Net su pendenza e DEM, su tutti e tre i "
                         "livelli. 'prithvi' e' lo schema C: Prithvi sulle sei "
                         "bande HLS, e i moduli agiscono SOLO a stride 16, "
                         "dove un ViT a patch 16 vive nativamente. Il pavimento "
                         "e' il modello senza fusione, perche' MGEM e DGFM "
                         "partono dall'identita' esatta.")
    ap.add_argument("--fusione-width", type=int, default=None,
                    help="larghezza dell'encoder ausiliario (default 24)")
    ap.add_argument("--fusione-lr", type=float, default=None,
                    help="learning rate della fusione (default 3e-4)")
    ap.add_argument("--gsrm", action="store_true",
                    help="filtra le due skip connection del decoder di SAM2 "
                         "(stride 4 e 8) con un gate ricavato dalla mappa piu' "
                         "profonda. Oggi quelle skip vengono sommate senza "
                         "alcun filtro: la tessitura superficiale contribuisce "
                         "anche dove il livello semantico non vede niente.")
    ap.add_argument("--gsrm-lr", type=float, default=None,
                    help="learning rate dei gate GSRM (default 3e-4). Sono "
                         "moduli nuovi, il base 1e-5 non li muoverebbe.")
    ap.add_argument("--bands", type=str, default=None,
                    help="bande OTTICHE da usare, per indice: es. \"3,7\" per "
                         "B4+B8 (la coppia NDVI). Le bande di terreno restano "
                         "quelle del config. Misurato: 14 bande contengono 3.14 "
                         "assi effettivi, e B4+B8 da sole fanno il 96.6% del "
                         "lavoro di tutte e quattordici.")
    ap.add_argument("--accum-steps", type=int, default=None,
                    help="passi di accumulo del gradiente. Il batch efficace e' "
                         "batch_size * accum_steps: serve a tenere lo stesso "
                         "batch efficace dei run precedenti quando la memoria "
                         "costringe ad abbassare --batch-size. NON entra nel nome "
                         "del checkpoint, perche' non cambia il modello.")
    ap.add_argument("--focal-gamma", type=float, default=None,
                    help="gamma della focal loss, che SOSTITUISCE la BCE. 0 = BCE "
                         "esatta (default). Valori tipici 1-2: tolgono peso ai pixel "
                         "gia' classificati bene e lo spostano su quelli difficili")
    ap.add_argument("--target-size", type=int, default=None,
                    help="risoluzione a cui SAM2 riceve le patch (native 128x128). "
                         "512: la mappa profonda e' 32x32, una cella copre 4x4 pixel "
                         "nativi. 1024: mappa 64x64, una cella copre 2x2 -- e 1024 e' "
                         "la risoluzione nativa del checkpoint SAM2, quindi nessun "
                         "campo di config viene riscritto")
    ap.add_argument("--select-on", choices=["totale", "frane", "micro"], default=None,
                    help="metrica su cui scegliere il checkpoint. 'micro' e' quella "
                         "confrontabile con la letteratura ed e' quella che si riporta: "
                         "selezionare su un'altra ottimizza una cosa e ne dichiara un'altra")
    ap.add_argument("--viz-samples", type=int, default=None)
    ap.add_argument("--prompt", choices=["gt_box", "image_box", "none", "yolo_box", "clf_box"],
                    default=None,
                    help="regime di prompt del mask decoder (default: quello in config.yaml). "
                         "Il nome di checkpoint e history ne eredita il suffisso.")
    ap.add_argument("--resume", action="store_true",
                    help="riprende il training boxhead dallo stato last_*.pth. "
                         "Senza questo flag un run fresco ARCHIVIA (mai sovrascrive) "
                         "checkpoint e history del run precedente.")
    ap.add_argument("-k", "--folds", type=int, default=5, help="numero di fold per il mode 'cv' (default: 5)")
    args = ap.parse_args()

    # Resolve dataset+mode: CLI if provided, otherwise interactive menu
    if args.dataset is None or args.mode is None:
        dataset, mode = interactive(args.config)
        args.dataset = args.dataset or dataset
        args.mode    = args.mode or mode

    # "multi": training congiunto su un solo modello (percorso dedicato).
    if args.dataset == "multi":
        from LENTE_multi_utils import load_multi_config
        from LENTE_multi_train import train_multi
        if args.mode not in ("train", "all"):
            ap.error("Per 'multi' e' supportato solo --mode train.")
        mcfg = load_multi_config(args.config)
        mcfg = apply_overrides(mcfg, args)
        print("\n" + "=" * 50)
        print(f"  Dataset : multi (training congiunto {mcfg.get('members')})")
        print(f"  Device  : {mcfg['device']} | epochs={mcfg['epochs']} | batch={mcfg['batch_size']} | lr={mcfg['learning_rate']}")
        print(f"  Ckpt    : {mcfg['checkpoint_name']}")
        print("=" * 50)
        train_multi(mcfg)
        return

    cfg = load_config(args.config, dataset=args.dataset, prompt_source=args.prompt)
    cfg = apply_overrides(cfg, args)

    print("\n" + "=" * 50)
    print(f"  Dataset : {cfg['dataset_name']}  (n_optical={cfg['n_optical']}, type={cfg['dataset_type']})")
    print(f"  Mode    : {args.mode}")
    print(f"  Device  : {cfg['device']}  | epochs={cfg['epochs']} | batch={cfg['batch_size']} | lr={cfg['learning_rate']}")
    print(f"  Prompt  : {cfg.get('prompt_source', 'none')}"
          + ("   <-- ATTENZIONE: il box viene dalla ground truth" if cfg.get("prompt_source") == "gt_box" else ""))
    print(f"  Ckpt    : {cfg['checkpoint_name']}")
    print("=" * 50)

    run_mode(cfg, args.mode, folds=args.folds)


if __name__ == "__main__":
    main()

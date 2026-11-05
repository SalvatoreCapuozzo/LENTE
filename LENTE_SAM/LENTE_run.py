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

MODES = ["train", "test", "visualize", "plot", "ablation", "cv", "all"]


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
    if args.viz_samples is not None:  cfg["viz_num_samples"] = args.viz_samples
    return cfg


def run_mode(cfg, mode, folds=5):
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
    ap.add_argument("--viz-samples", type=int, default=None)
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

    cfg = load_config(args.config, dataset=args.dataset)
    cfg = apply_overrides(cfg, args)

    print("\n" + "=" * 50)
    print(f"  Dataset : {cfg['dataset_name']}  (n_optical={cfg['n_optical']}, type={cfg['dataset_type']})")
    print(f"  Mode    : {args.mode}")
    print(f"  Device  : {cfg['device']}  | epochs={cfg['epochs']} | batch={cfg['batch_size']} | lr={cfg['learning_rate']}")
    print(f"  Ckpt    : {cfg['checkpoint_name']}")
    print("=" * 50)

    run_mode(cfg, args.mode, folds=args.folds)


if __name__ == "__main__":
    main()

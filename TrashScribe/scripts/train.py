"""Train one TrashScribe configuration, then evaluate its best checkpoint on test.

    python scripts/train.py configs/exp/multi_domain.yaml [key=value ...]
"""
import _bootstrap  # noqa: F401
import argparse
import json

from trashscribe.engine import Trainer, load_config
from evaluate import evaluate_run

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("overrides", nargs="*")
ap.add_argument("--no-test", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
tr = Trainer(cfg)
tr.train()
if not a.no_test:
    res = evaluate_run(tr.run_dir, split="test")
    print(json.dumps(res.get("summary", {}), indent=1))

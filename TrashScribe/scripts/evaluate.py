"""Evaluate a run directory on a split.

Reports: waste segmentation per dataset; descriptor mAP vs prevalence floor;
naming under the generalised zero-shot protocol (seen vs held-out classes);
retrieval by atlas entry for held-out classes (AP, lift, CI, permutation p);
environment leakage of object embeddings. Novel atlases (classes absent from
every dataset) are loaded through an EXTENDED schema: no retraining.

    python scripts/evaluate.py runs/<run> [--split test]
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from trashscribe.data.canonical import Canonicalizer, collate
from trashscribe.data.mixing import MultiDataset
from trashscribe.data.registry import build_split
from trashscribe.engine import _p, evaluate, load_run
from trashscribe.naming import Namer, build_cache, calibrate_seen_penalty
from trashscribe.schema import Atlas


def evaluate_run(run_dir, split="test", n_perm=None, with_novel=True):
    run_dir = Path(run_dir)
    model, cfg, schema, atlas, tb = load_run(run_dir)
    dev = next(model.parameters()).device
    root = Path(cfg.get("data_root", ".")).expanduser()
    holdout = set(cfg.get("holdout") or [])
    names = cfg.get("test_datasets") if split == "test" else cfg.get("val_datasets") or cfg.get("train_datasets")
    adapters = build_split(cfg.get("datasets", []), names, split, atlas, root, set(), cfg.get("max_items"),
                           strict=cfg.get("strict_atlas", True))
    train_ad = build_split(cfg.get("datasets", []), cfg.get("train_datasets"), "train", atlas, root, holdout,
                           cfg.get("max_items"), strict=cfg.get("strict_atlas", True))
    seen = set().union(*[a.entries_present() for a in train_ad]) if train_ad else set()
    present = set().union(*[a.entries_present() for a in adapters]) if adapters else set()
    unseen = holdout | (present - seen)          # LODO: classes of an unseen dataset are unseen too
    canon = Canonicalizer(schema, atlas, model.cfg.image_size, False)
    dl = DataLoader(MultiDataset(adapters, canon), batch_size=cfg["train"].get("val_batch_size", 8),
                    collate_fn=collate, num_workers=cfg["train"].get("num_workers", 4))
    # naming atlas = training atlases (+ novel ones), schema extended with their new values
    atlases = [atlas] + ([Atlas.load(_p(p)) for p in cfg.get("novel_atlases", [])] if with_novel else [])
    full = Atlas.merge(atlases, "eval")
    ext = schema.extended([full])
    cache = build_cache(model, ext, full, tb)
    tb.save()
    cache.save(run_dir / "atlas_cache.pt")
    nc = cfg.get("eval", {}).get("naming", {})
    namer = Namer(cache, nc.get("weights"), nc.get("conf_thr", 0.0), nc.get("margin_thr", 0.0))
    # seen-class penalty calibrated on the VALIDATION split (no held-out class in it)
    val_ad = build_split(cfg.get("datasets", []), cfg.get("val_datasets") or cfg.get("train_datasets"), "val", atlas,
                         root, holdout, cfg.get("max_items"), strict=cfg.get("strict_atlas", True))
    dl_val = DataLoader(MultiDataset(val_ad, canon), batch_size=cfg["train"].get("val_batch_size", 8),
                        collate_fn=collate, num_workers=cfg["train"].get("num_workers", 4))
    raw = evaluate(model, dl_val, schema, dev, amp=cfg["train"].get("amp", True), return_raw=True).get("_raw")
    calib = {"gamma": 0.0}
    if raw is not None:
        calib = calibrate_seen_penalty(namer, raw["z"], raw["zc"], raw["entries"], seen,
                                       keep=cfg.get("eval", {}).get("calib_keep", 0.9))
    namer.set_seen(seen, 0.0)
    res0 = evaluate(model, dl, schema, dev, namer=namer, seen=seen, unseen=unseen, n_perm=10,
                    amp=cfg["train"].get("amp", True))
    namer.set_seen(seen, calib["gamma"])
    res = evaluate(model, dl, schema, dev, namer=namer, seen=seen, unseen=unseen,
                   retrieval_entries=sorted(unseen | seen), n_perm=n_perm or cfg.get("eval", {}).get("n_perm", 2000),
                   amp=cfg["train"].get("amp", True))
    res["summary"] = {
        "seen_classes": len(seen), "unseen_classes": sorted(unseen & present),
        "waste_iou": res["seg"].get("_all", {}).get("iou"),
        "waste_bf1": res["seg"].get("_all", {}).get("bf1"),
        "desc_map": res["desc"].get("_macro_intrinsic", {}).get("map"),
        "desc_floor": res["desc"].get("_macro_intrinsic", {}).get("floor"),
        "naming": res.get("naming"),
        "naming_uncalibrated": res0.get("naming"),
        "calibration": calib,
        "naming_ausuc": {k: v for k, v in (res.get("naming_ausuc") or {}).items() if k != "curve"},
        "naming_by_stream": {k: res.get(f"naming_{k}", {}).get("unseen") for k in ("dap", "proto", "prose")},
        "heldout_retrieval": {e: {k: v for k, v in r.items() if k in ("ap", "lift", "p")}
                              for e, r in res.get("retrieval", {}).items() if e in unseen},
        "leak_intrinsic": res.get("leakage_intrinsic"), "leak_context": res.get("leakage_context"),
        "extended_values": [v for v in ext.flat if v not in schema.flat],
    }
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-perm", type=int, default=None)
    a = ap.parse_args()
    r = evaluate_run(a.run_dir, a.split, a.n_perm)
    print(json.dumps(r["summary"], indent=1, default=float))

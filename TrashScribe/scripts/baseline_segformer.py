"""Fully fine-tuned SegFormer baseline for binary waste segmentation, trained on
exactly the same data, splits, augmentation and sampler as the TrashScribe config.

    python scripts/baseline_segformer.py configs/exp/e3_lodo_taco.yaml [--model nvidia/mit-b2] [--transplant]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from trashscribe import device as D
from trashscribe.data.canonical import Canonicalizer, collate, to_device
from trashscribe.data.mixing import MultiDataset
from trashscribe.data.registry import build_split
from trashscribe.engine import _p, load_config, load_vocab
from trashscribe.losses import mask_loss
from trashscribe.metrics import SegMeter

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--model", default="nvidia/mit-b2")
ap.add_argument("--transplant", action="store_true", help="also use the cross-dataset transplant")
ap.add_argument("--size", type=int, default=None)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
tc = cfg["train"]
dev = D.pick_device(cfg.get("device", "auto"))
D.setup(dev, cfg.get("seed", 42))
size = a.size or cfg["model"]["image_size"]
schema, atlas = load_vocab(cfg)
root = _p(".").__class__(cfg.get("data_root", ".")).expanduser()
hold = set(cfg.get("holdout") or [])
mk = lambda names, split, h: build_split(cfg["datasets"], names, split, atlas, root, h, cfg.get("max_items"),  # noqa
                                         strict=cfg.get("strict_atlas", True))
tr = MultiDataset(mk(cfg.get("train_datasets"), "train", hold), Canonicalizer(schema, atlas, size, True, tc.get("aug")),
                  tc.get("transplant_p", 0.0) if a.transplant else 0.0, tc.get("transplant"))
va = MultiDataset(mk(cfg.get("val_datasets") or cfg.get("train_datasets"), "val", hold),
                  Canonicalizer(schema, atlas, size, False))
te = MultiDataset(mk(cfg.get("test_datasets"), "test", set()), Canonicalizer(schema, atlas, size, False))
nw = int(tc.get("num_workers", 4))
kw = dict(collate_fn=collate, num_workers=nw, pin_memory=dev.type == "cuda", persistent_workers=nw > 0)
dl_tr = DataLoader(tr, batch_size=tc["batch_size"], drop_last=True, **kw,
                   sampler=WeightedRandomSampler(tr.sample_weights(tc.get("dataset_alpha", 0.5)),
                                                 int(tc.get("epoch_size") or len(tr)), replacement=True))
dl_va = DataLoader(va, batch_size=tc.get("val_batch_size", 8), **kw)
dl_te = DataLoader(te, batch_size=tc.get("val_batch_size", 8), **kw)

from transformers import SegformerForSemanticSegmentation  # noqa: E402
net = SegformerForSemanticSegmentation.from_pretrained(a.model, num_labels=1, ignore_mismatched_sizes=True).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=6e-5, weight_decay=1e-2)
steps = tc["epochs"] * len(dl_tr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * s / steps)))
name = f"{cfg['run_name']}__segformer_{a.model.split('/')[-1]}{'_tp' if a.transplant else ''}"
out = _p(cfg.get("output_dir", "runs")) / name
out.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def run_eval(dl):
    net.eval()
    m = SegMeter()
    for b in dl:
        b = to_device(b, dev)
        with D.autocast(dev, tc.get("amp", True)):
            lg = net(pixel_values=b["image"]).logits
        m.update(torch.sigmoid(F.interpolate(lg.float(), size=b["waste"].shape[-2:], mode="bilinear")),
                 b["waste"], b["valid"], b["dataset"])
    return m.result()


best, hist = -1, []
for ep in range(1, tc["epochs"] + 1):
    net.train()
    t = time.time()
    tot = 0.0
    for b in dl_tr:
        b = to_device(b, dev)
        with D.autocast(dev, tc.get("amp", True)):
            lg = net(pixel_values=b["image"]).logits
        loss = mask_loss(lg, b["waste"], b["valid"], tc["loss"].get("pos_weight", 2.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += float(loss)
    v = run_eval(dl_va)["_all"]
    hist.append({"epoch": ep, "loss": tot / len(dl_tr), "val_iou": v["iou"], "time_s": time.time() - t})
    print(hist[-1])
    if v["iou"] > best:
        best = v["iou"]
        torch.save(net.state_dict(), out / "best.pt")
net.load_state_dict(torch.load(out / "best.pt", map_location=dev))
res = {"seg": run_eval(dl_te), "params": sum(p.numel() for p in net.parameters()), "model": a.model}
res["summary"] = {"waste_iou": res["seg"]["_all"]["iou"], "waste_bf1": res["seg"]["_all"]["bf1"]}
(out / "history.json").write_text(json.dumps(hist, indent=1))
(out / "test_metrics.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res["summary"], indent=1))

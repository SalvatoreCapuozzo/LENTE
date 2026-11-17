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
from trashscribe.progress import Progress

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
kw = dict(collate_fn=collate, num_workers=nw, pin_memory=dev.type == "cuda" and bool(tc.get("pin_memory", True)), persistent_workers=nw > 0)
dl_tr = DataLoader(tr, batch_size=tc["batch_size"], drop_last=True, **kw,
                   sampler=WeightedRandomSampler(tr.sample_weights(tc.get("dataset_alpha", 0.5)),
                                                 int(tc.get("epoch_size") or len(tr)), replacement=True))
kwv = dict(collate_fn=collate, num_workers=int(tc.get("val_num_workers", nw)), pin_memory=dev.type == "cuda" and bool(tc.get("pin_memory", True)))
dl_va = DataLoader(va, batch_size=tc.get("val_batch_size", 8), **kwv)
dl_te = DataLoader(te, batch_size=tc.get("val_batch_size", 8), **kwv)

from transformers import SegformerForSemanticSegmentation  # noqa: E402
from transformers.models.segformer import modeling_segformer as _ms  # noqa: E402


def _sdpa_attention(self, hidden_states, height, width, output_attentions=False):
    """Same maths as the eager SegFormer attention (max |diff| ~3e-5), through
    F.scaled_dot_product_attention: the eager path stores the full 65k x 1k
    attention matrix of stage 1 at 1024 px and does not fit batch 2 in 8 GB."""
    b, n, _ = hidden_states.shape
    h, d = self.num_attention_heads, self.attention_head_size
    q = self.query(hidden_states).view(b, -1, h, d).transpose(1, 2)
    if self.sr_ratio > 1:
        c = hidden_states.shape[-1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(b, c, height, width)
        hidden_states = self.layer_norm(self.sr(hidden_states).reshape(b, c, -1).permute(0, 2, 1))
    k = self.key(hidden_states).view(b, -1, h, d).transpose(1, 2)
    v = self.value(hidden_states).view(b, -1, h, d).transpose(1, 2)
    ctx = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
    return (ctx.permute(0, 2, 1, 3).reshape(b, n, self.all_head_size),)


_ms.SegformerEfficientSelfAttention.forward = _sdpa_attention
net = SegformerForSemanticSegmentation.from_pretrained(a.model, num_labels=1, ignore_mismatched_sizes=True).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=6e-5, weight_decay=1e-2)
accum = max(1, int(tc.get("accum_steps", 1)))      # micro-batch = batch_size, optimizer batch x accum
steps = tc["epochs"] * (len(dl_tr) // accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * s / steps)))
name = f"{cfg['run_name']}__segformer_{a.model.split('/')[-1]}{'_tp' if a.transplant else ''}"
out = _p(cfg.get("output_dir", "runs")) / name
out.mkdir(parents=True, exist_ok=True)
prog = Progress(out)
prog(phase="setup", epochs=int(tc["epochs"]), patience=None)


@torch.no_grad()
def run_eval(dl):
    net.eval()
    if dev.type == "cuda":
        # 15 Nov: the first validation OOMed in all 5 LODO jobs (val batch 4 at 1024 px: the
        # decode head concatenates 4 x 768 maps at 256 x 256, 1.5 GiB per tensor, on top of the
        # blocks the allocator still caches from training). Free them; the queue now runs val at batch 1.
        torch.cuda.empty_cache()
    m = SegMeter()
    for b in dl:
        b = to_device(b, dev)
        with D.autocast(dev, tc.get("amp", True)):
            lg = net(pixel_values=b["image"]).logits
        m.update(torch.sigmoid(F.interpolate(lg.float(), size=b["waste"].shape[-2:], mode="bilinear")),
                 b["waste"], b["valid"], b["dataset"])
    return m.result()


best, hist, bad, start = -1, [], 0, 1
patience = int(tc.get("patience") or 0)          # same early stop as TrashScribe (base.yaml: 8)
if (out / "last.pt").exists():                   # resume after a stop of the queue
    ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
    best, hist, bad, start = ck["best"], ck["hist"], ck["bad"], ck["epoch"] + 1
    print(f"resumed after epoch {ck['epoch']} (best val IoU {best:.4f})", flush=True)
for ep in range(start, tc["epochs"] + 1):
    if patience and bad >= patience:
        break
    net.train()
    t = time.time()
    prog(phase="train", epoch=ep, step=0, steps=len(dl_tr) // accum, epoch_started=t)
    tot = 0.0
    opt.zero_grad(set_to_none=True)
    for mi, b in enumerate(dl_tr):
        b = to_device(b, dev)
        with D.autocast(dev, tc.get("amp", True)):
            lg = net(pixel_values=b["image"]).logits
        loss = mask_loss(lg, b["waste"], b["valid"], tc["loss"].get("pos_weight", 2.0))
        (loss / accum).backward()
        tot += float(loss.detach())
        if (mi + 1) % accum:
            continue
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        prog(step=(mi + 1) // accum)
    prog(phase="val")
    v = run_eval(dl_va)["_all"]
    hist.append({"epoch": ep, "loss": tot / len(dl_tr), "val_iou": v["iou"], "time_s": time.time() - t})
    print(hist[-1])
    if v["iou"] > best:
        best, bad = v["iou"], 0
        torch.save(net.state_dict(), out / "best.pt")
    else:
        bad += 1
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                "best": best, "hist": hist, "bad": bad}, out / "last.pt.tmp")
    (out / "last.pt.tmp").replace(out / "last.pt")
net.load_state_dict(torch.load(out / "best.pt", map_location=dev))
prog(phase="test", eval_started=time.time())
res = {"seg": run_eval(dl_te), "params": sum(p.numel() for p in net.parameters()), "model": a.model}
res["summary"] = {"waste_iou": res["seg"]["_all"]["iou"], "waste_bf1": res["seg"]["_all"]["bf1"]}
(out / "history.json").write_text(json.dumps(hist, indent=1))
(out / "test_metrics.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res["summary"], indent=1))
prog(phase="done", eval_done=time.time())

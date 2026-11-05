"""Export a trained run for the edge.

    python scripts/export_onnx.py runs/<run>            # trained weights
    python scripts/export_onnx.py --untrained --size 640  # graph check without a run

Writes <run>/export/trashscribe_<S>.onnx (+ atlas_cache.pt, export_meta.json). LoRA is
folded into the frozen Linears first; the zero-init adapters stay as small
modules. The ONNX outputs are checked against PyTorch on CPU.
Then, ON THE JETSON:  bash scripts/build_trt.sh <run>/export/trashscribe_<S>.onnx
"""
import _bootstrap  # noqa: F401
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from trashscribe.engine import _p, build_model, load_config, load_run, load_vocab
from trashscribe.models.trashscribe import TrashScribeEdge
from trashscribe.text_bank import TextBank

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", nargs="?")
ap.add_argument("--untrained", action="store_true")
ap.add_argument("--size", type=int, default=None)
ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--out", default=None)
a = ap.parse_args()

if a.untrained:
    cfg = load_config(_p("configs/base.yaml"), [f"model.image_size={a.size or 640}"])
    schema, _ = load_vocab(cfg)
    tb = TextBank(cfg["text_model"], "cpu", cache_path=_p(cfg["text_cache"]))
    model = build_model(cfg, schema, tb, torch.device("cpu")).eval()
    out_dir = Path(a.out or _p("export/untrained"))
else:
    model, cfg, schema, atlas, tb = load_run(a.run_dir, device="cpu")
    out_dir = Path(a.out or Path(a.run_dir) / "export")
out_dir.mkdir(parents=True, exist_ok=True)
S = model.cfg.image_size
if a.size and a.size != S:
    raise SystemExit(f"--size {a.size} != trained size {S}: SAM2 positional grids are built for {S}")
model.backbone.merge_lora_()
edge = TrashScribeEdge(model).eval()
x = torch.randn(1, 3, S, S)
with torch.no_grad():
    ref = [t.numpy() for t in edge(x)]
path = out_dir / f"trashscribe_{S}.onnx"
torch.onnx.export(edge, (x,), str(path), input_names=["image"], output_names=["waste_prob", "contour_prob", "desc"],
                  opset_version=a.opset, do_constant_folding=True, dynamo=False)
print("exported", path, f"{path.stat().st_size / 2**20:.1f} MB")

import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
got = sess.run(None, {"image": x.numpy()})
diff = {n: float(np.abs(r - g).max()) for n, r, g in zip(["waste_prob", "contour_prob", "desc"], ref, got)}
print("max |onnx - torch|:", diff)
meta = {"image_size": S, "opset": a.opset, "outputs": {n: list(r.shape) for n, r in
                                                        zip(["waste_prob", "contour_prob", "desc"], ref)},
        "max_abs_diff_vs_torch": diff, "params": sum(p.numel() for p in model.parameters()),
        "ring_cells": model.cfg.ring_cells, "naming": cfg.get("eval", {}).get("naming", {})}
(out_dir / "export_meta.json").write_text(json.dumps(meta, indent=1))
if not a.untrained:
    cache = Path(a.run_dir) / "atlas_cache.pt"
    if cache.exists():
        shutil.copy(cache, out_dir / "atlas_cache.pt")
    else:
        print("note: run scripts/evaluate.py first to produce atlas_cache.pt for naming on device")
assert max(diff.values()) < 1e-3, "ONNX output deviates from PyTorch"

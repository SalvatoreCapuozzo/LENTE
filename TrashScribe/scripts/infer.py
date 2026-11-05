"""Offline monitoring on images, a folder, a video file or a camera.

    python scripts/infer.py --run runs/<run> --source photos/ --out out/
    python3 scripts/infer.py --engine trashscribe_640_fp16.engine --cache atlas_cache.pt --source 0   # CSI/USB cam
    python scripts/infer.py ... --atlas atlas/my_city.yaml   # NEW classes / streams as text, no retraining*

Writes, per frame: an overlay JPEG and one JSON line (coverage, count, objects
with descriptors, names or "set aside", composition by stream and material),
plus a readable report. *A new atlas needs its phrases embedded once: run with
--run on a machine with the text encoder, which refreshes atlas_cache.pt.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import cv2

from trashscribe.edge_runtime import EdgePipeline, OrtBackend, TorchBackend, TrtBackend, overlay
from trashscribe.naming import AtlasCache

ap = argparse.ArgumentParser()
ap.add_argument("--run")
ap.add_argument("--engine")
ap.add_argument("--onnx")
ap.add_argument("--cache")
ap.add_argument("--atlas", nargs="*", help="extra atlas files (requires --run to embed them)")
ap.add_argument("--source", required=True)
ap.add_argument("--out", default="runs/infer")
ap.add_argument("--every", type=int, default=1, help="process one frame every N (video/camera)")
ap.add_argument("--max-frames", type=int, default=0)
a = ap.parse_args()

if a.run:
    from trashscribe.engine import _p, load_run
    from trashscribe.naming import build_cache
    from trashscribe.schema import Atlas
    model, cfg, schema, atlas, tb = load_run(a.run)
    extra = [Atlas.load(p) for p in (a.atlas or [])] + [Atlas.load(_p(p)) for p in cfg.get("novel_atlases", [])]
    full = Atlas.merge([atlas] + extra, "infer")
    cache = build_cache(model, schema.extended([full]), full, tb)
    tb.save()
    backend, size = TorchBackend(model), model.cfg.image_size
    naming = cfg.get("eval", {}).get("naming")
else:
    cache = AtlasCache.load(a.cache)
    backend = TrtBackend(a.engine) if a.engine else OrtBackend(a.onnx)
    size = backend.bufs["image"].shape[-1] if a.engine else backend.s.get_inputs()[0].shape[-1]
    naming = None
pipe = EdgePipeline(backend, cache, size, naming)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def frames():
    src = Path(a.source)
    if a.source.isdigit() or src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                yield f"frame{i:06d}", f
            i += 1
    else:
        files = [src] if src.is_file() else sorted(p for p in src.iterdir()
                                                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for p in files:
            yield p.stem, cv2.imread(str(p))


with open(out / "reports.jsonl", "w") as fj, open(out / "report.txt", "w") as ft:
    for n, (name, bgr) in enumerate(frames()):
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rep = pipe(rgb)
        cv2.imwrite(str(out / f"{name}_overlay.jpg"), cv2.cvtColor(overlay(rgb, rep, size), cv2.COLOR_RGB2BGR))
        rep.pop("_maps")
        fj.write(json.dumps({"frame": name, **rep}, default=float) + "\n")
        ft.write(f"== {name}: coverage {100 * rep['coverage']:.1f}%, {rep['count']} objects, "
                 f"{rep['timing']['total_ms']:.0f} ms\n" + "\n".join("  " + s for s in rep["lines"]) + "\n")
        print(name, f"{rep['count']} objects", f"{rep['timing']['total_ms']:.0f} ms")
        if a.max_frames and n + 1 >= a.max_frames:
            break
print("written to", out)

"""Shrink a COCO dataset to the model resolution, straight from its archive.

The pipeline letterboxes every image to `model.image_size` (<= 1024) anyway, so
storing 4000x3000 originals only costs disk and decode time. This writes images
with longest side <= --max-side (JPEG) and an annotation file whose polygons,
RLE masks, boxes and areas are rescaled to the new frame. Images are decoded
with EXIF orientation applied (as cv2 does at load time); an image whose
decoded frame does not match its annotation frame is reported, not guessed.

    python scripts/prepare_coco.py --src TACO.zip --ann TACO/data/annotations.json \
        --img-root TACO/data --out ~/data/waste/TACO/data
`--src` may be a .zip or a folder; --ann / --img-root are paths inside it.
"""
import argparse
import json
import os
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

_ZIP = {}


def _read(src: str, name: str) -> bytes:
    if src.endswith(".zip"):
        key = (src, os.getpid())          # one handle per process: a forked handle shares its offset
        z = _ZIP.get(key)
        if z is None:
            z = _ZIP[key] = zipfile.ZipFile(src)
        try:
            return z.read(name)
        except zipfile.BadZipFile:
            # Python >= 3.12 rejects some archives (TACO, ZeroWaste) as "overlapped
            # entries"; Info-ZIP reads them and verifies their CRCs
            return subprocess.run(["unzip", "-p", src, name], check=True, capture_output=True).stdout
    return (Path(src) / name).read_bytes()


def _shrink(job):
    src, member, dst, max_side, quality, want = job
    buf = np.frombuffer(_read(src, member), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)            # applies EXIF orientation
    if img is None:
        return member, None
    if want[0] and img.shape[:2] != tuple(want):
        # annotations drawn on the stored (un-rotated) frame
        raw = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if raw is not None and raw.shape[:2] == tuple(want):
            img = raw
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    if k < 1:
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return member, (h, w, nh, nw)


def _scale_ann(a, h, w, nh, nw):
    sx, sy = nw / w, nh / h
    seg = a.get("segmentation")
    if isinstance(seg, list):
        a["segmentation"] = [[round(v * (sx if i % 2 == 0 else sy), 2) for i, v in enumerate(poly)]
                             for poly in seg if len(poly) >= 6]
        if a["segmentation"]:
            rle = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], nh, nw))
            a["area"] = float(mask_utils.area(rle))
    elif isinstance(seg, dict):
        rle = mask_utils.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
        if isinstance(rle.get("counts"), str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode()}
        m = mask_utils.decode(rle).astype(np.float32)
        m = (cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
        r = mask_utils.encode(np.asfortranarray(m))
        a["area"] = float(mask_utils.area(r))
        r["counts"] = r["counts"].decode()
        a["segmentation"] = r
    if "bbox" in a:
        x, y, bw, bh = a["bbox"]
        a["bbox"] = [x * sx, y * sy, bw * sx, bh * sy]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ann", required=True, help="annotation json, path inside --src")
    ap.add_argument("--img-root", default="", help="folder of image file_names, path inside --src")
    ap.add_argument("--out", required=True, help="output folder: images + annotations.json")
    ap.add_argument("--ann-name", default="annotations.json")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--split-json", default=None,
                    help="official split {split: [file_name, ...]} (path inside --src): also writes "
                         "<ann-name stem>_<split>.json")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    ds = json.loads(_read(a.src, a.ann))
    root = a.img_root.rstrip("/") + "/" if a.img_root else ""
    jobs, new_name = [], {}
    for im in ds["images"]:
        fn = im["file_name"]
        nfn = str(Path(fn).with_suffix(".jpg"))
        new_name[im["id"]] = nfn
        jobs.append((a.src, root + fn, str(out / nfn), a.max_side, a.quality,
                     (im.get("height"), im.get("width"))))
    done = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (member, r) in enumerate(ex.map(_shrink, jobs, chunksize=8)):
            done[member] = r
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    by_id = {im["id"]: im for im in ds["images"]}
    missing, mismatch, keep_ids = [], [], set()
    for im, job in zip(ds["images"], jobs):
        r = done[job[1]]
        if r is None:
            missing.append(im["file_name"])
            continue
        h, w, nh, nw = r
        if (im.get("height"), im.get("width")) not in ((h, w), (None, None)):
            mismatch.append((im["file_name"], (im.get("height"), im.get("width")), (h, w)))
            continue
        im["orig_height"], im["orig_width"], im["orig_file_name"] = h, w, im["file_name"]
        im["height"], im["width"], im["file_name"] = nh, nw, new_name[im["id"]]
        keep_ids.add(im["id"])
    ds["images"] = [im for im in ds["images"] if im["id"] in keep_ids]
    anns = []
    for an in ds["annotations"]:
        if an["image_id"] not in keep_ids:
            continue
        im = by_id[an["image_id"]]
        anns.append(_scale_ann(an, im["orig_height"], im["orig_width"], im["height"], im["width"]))
    ds["annotations"] = anns
    (out / a.ann_name).write_text(json.dumps(ds))
    if a.split_json:
        sp = json.loads(_read(a.src, a.split_json))
        stem = Path(a.ann_name).stem
        name_of = {im["id"]: Path(im["orig_file_name"]).name for im in ds["images"]}
        for split, files in sp.items():
            files = {Path(f).name for f in files}
            ids = {i for i, n in name_of.items() if n in files}
            sub = {**ds, "images": [im for im in ds["images"] if im["id"] in ids],
                   "annotations": [an for an in ds["annotations"] if an["image_id"] in ids]}
            (out / f"{stem}_{split}.json").write_text(json.dumps(sub))
            print(f"split {split}: {len(ids)} images (listed {len(files)})")
    print(f"images kept {len(keep_ids)}, unreadable {len(missing)}, frame mismatch {len(mismatch)}; "
          f"annotations {len(anns)}")
    for m in mismatch[:20]:
        print("  mismatch", m)
    for m in missing[:20]:
        print("  unreadable", m)


if __name__ == "__main__":
    main()

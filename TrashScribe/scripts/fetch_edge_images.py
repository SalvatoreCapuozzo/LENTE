"""Fetch a few TEST-split images per environment for the on-device benchmark, without
downloading whole archives: the public zips are read over HTTP range requests (only the
central directory, the split files and the chosen images are transferred).

    python scripts/fetch_edge_images.py --out ~/data/waste/_edge_test --n 30
    python scripts/fetch_edge_images.py --list taco          # show an archive's members

Images are shrunk to longest side <= 1024 px, as on the training workstation
(scripts/prepare_coco.py), so pre-processing times match what the model saw. The selection
is deterministic (seed 0 over the sorted test ids). Not used for any accuracy number.

Splits reproduce configs/datasets.yaml:
  TACO        random split seed 42 (70/10/20) over the image ids of annotations.json
  ZeroWaste-f official test/ folder
  UAVVaste    official test list (annotations/train_val_test_distribution_file.json)
  MJU-Waste   Google Drive only (no range requests): not fetched, copy it from the workstation
  BePLi v2    SEANOE archive if it is reachable, else skipped
"""
import argparse
import io
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ARCHIVES = {
    "taco": "https://zenodo.org/api/records/3587843/files/TACO.zip/content",
    "zerowaste": "https://zenodo.org/api/records/6412647/files/zerowaste-f-final.zip/content",
    "uavvaste": "https://zenodo.org/api/records/8214061/files/UAVVasteDataset.zip/content",
}
UA = {"User-Agent": "trashscribe-edge-bench/1.0"}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; 1 MB blocks are cached."""

    BLOCK = 1 << 20

    def __init__(self, url):
        self.src, self.pos, self.cache = url, 0, {}
        self._resolve()
        self.fetched = 0

    def _resolve(self):
        req = urllib.request.Request(self.src, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
            self.url = r.geturl()                 # follow the redirect (re-resolved after errors)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            a, b = i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK) - 1
            for attempt in range(8):
                try:
                    req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={a}-{b}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        self.cache[i] = r.read()
                    break
                except Exception:  # noqa: BLE001  (Zenodo answers 504 now and then)
                    if attempt == 7:
                        raise
                    time.sleep(min(60, 2 ** (attempt + 1)))
                    try:
                        self._resolve()
                    except Exception:  # noqa: BLE001
                        pass
            self.fetched += len(self.cache[i])
            if len(self.cache) > 64:              # keep at most 64 MB in memory
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n:
            i, o = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_zip(name):
    f = HttpRangeFile(ARCHIVES[name])
    return zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20)), f


def find(z, suffix):
    hits = [n for n in z.namelist() if n.endswith(suffix) and "__MACOSX" not in n]
    if not hits:
        raise SystemExit(f"{suffix} not found in archive")
    return sorted(hits, key=len)[0]


def shrink_save(data: bytes, dst: Path, max_side=1024):
    import cv2
    import numpy as np
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print("  unreadable image skipped:", dst.name)
        return False
    h, w = im.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        im = cv2.resize(im, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst.with_suffix(".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def pick(ids, n):
    ids = sorted(ids)
    random.Random(0).shuffle(ids)
    return ids[:n]


def split_ids(ids, fractions, seed):               # copy of trashscribe.data.adapters.split_ids
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    a = round(len(ids) * fractions["train"])
    b = a + round(len(ids) * fractions["val"])
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def test_members(name, z):
    """Archive members of the test split (list of (member, output stem))."""
    if name == "taco":
        ann_m = find(z, "annotations.json")
        ann = json.loads(z.read(ann_m))
        root = ann_m.rsplit("/", 1)[0] + "/" if "/" in ann_m else ""
        ids = split_ids([im["id"] for im in ann["images"]], {"train": 0.7, "val": 0.1, "test": 0.2}, 42)["test"]
        by_id = {im["id"]: im["file_name"] for im in ann["images"]}
        return [(root + by_id[i], by_id[i].replace("/", "_").rsplit(".", 1)[0]) for i in ids]
    if name == "zerowaste":
        lab = find(z, "test/labels.json")
        ann = json.loads(z.read(lab))
        root = lab.rsplit("/", 1)[0] + "/data/"
        names = set(z.namelist())
        out = []
        for im in ann["images"]:
            m = root + im["file_name"]
            if m not in names:
                m = lab.rsplit("/", 1)[0] + "/" + im["file_name"]
            out.append((m, Path(im["file_name"]).stem))
        return out
    if name == "uavvaste":
        dist = json.loads(z.read(find(z, "train_val_test_distribution_file.json")))
        test = dist.get("test") or dist.get("Test")
        by_base = {Path(n).name: n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        return [(by_base[t], Path(t).stem) for t in test if t in by_base]
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "data/waste/_edge_test"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--datasets", nargs="*", default=list(ARCHIVES))
    ap.add_argument("--list", metavar="DATASET")
    a = ap.parse_args()
    if a.list:
        z, _ = open_zip(a.list)
        for n in z.namelist()[:4000]:
            print(n)
        return
    for name in a.datasets:
        dst = Path(a.out).expanduser() / name
        have = sorted(dst.glob("*.jpg")) if dst.exists() else []
        if len(have) >= a.n:
            print(f"{name}: {len(have)} images already in {dst}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        z, f = open_zip(name)
        members = test_members(name, z)
        chosen = pick(members, a.n)
        print(f"{name}: {len(members)} test images in the split, fetching {len(chosen)}")
        ok = 0
        for m, stem in chosen:
            if (dst / stem).with_suffix(".jpg").exists():      # resume after an interrupted run
                ok += 1
                continue
            ok += shrink_save(z.read(m), dst / stem)
        (dst / "_selection.json").write_text(json.dumps({"archive": ARCHIVES[name], "split": "test",
                                                         "members": [m for m, _ in chosen]}, indent=1))
        print(f"{name}: {ok} images in {dst} ({f.fetched / 1e6:.0f} MB transferred)")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

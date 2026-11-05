"""Print the categories of a COCO file, their instance counts, and which ones
the given atlases do not cover (so class_map / atlas can be completed before
training instead of failing mid-run).

    python scripts/inspect_dataset.py ann.json --atlas atlas/taco.yaml atlas/binary.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import json
from collections import Counter

from trashscribe.schema import Atlas

ap = argparse.ArgumentParser()
ap.add_argument("ann")
ap.add_argument("--atlas", nargs="*", default=[])
a = ap.parse_args()
d = json.load(open(a.ann))
names = Atlas.merge([Atlas.load(p) for p in a.atlas]).names if a.atlas else []
cnt = Counter(x["category_id"] for x in d["annotations"])
print(f"{len(d['images'])} images, {len(d['annotations'])} annotations, {len(d['categories'])} categories")
miss = []
for c in d["categories"]:
    ok = c["name"] in names
    miss += [] if ok else [c["name"]]
    print(f"{c['id']:>4}  {c.get('supercategory', ''):<25} {c['name']:<35} {cnt[c['id']]:>7}  {'ok' if ok else 'NOT IN ATLAS'}")
if "scene_categories" in d:
    print("scene categories:", [s["name"] for s in d["scene_categories"]])
seg = Counter(type(x.get("segmentation")).__name__ for x in d["annotations"])
print("segmentation encodings:", dict(seg))
if miss:
    print("\nclass_map skeleton:\n  class_map:\n" + "\n".join(f'    "{m}": null' for m in miss))

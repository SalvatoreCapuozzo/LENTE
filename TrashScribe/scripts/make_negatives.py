"""Waste-free indoor scenes with people, from COCO val2017, as extra NEGATIVES (E3b).

Why: the leave-one-dataset-out model for MJU-Waste (people holding waste indoors)
marks faces, hands, clothes and furniture as waste (IoU 0.055, precision 0.055): no
training environment shows people or furniture among its negatives. This builds a
small negative set from a generic, waste-agnostic source -- nothing from MJU-Waste.

Rule (fixed before any result):
  * a person covering >= 5 % of the image (MJU shows people at arm's length);
  * at least one indoor furniture / appliance / electronics category;
  * none of the categories that could be waste or look like it: containers, tableware,
    food, bags, paper (book) and glass (vase), toothbrush.
Images are copied as JPEG (COCO is already <= 640 px); masks are all zero (everything
is background). Seeded 80/20 train/val split by the registry entry, no test split.

    python scripts/make_negatives.py --zip-img ~/data/waste/_dl/coco/val2017.zip \
        --zip-ann ~/data/waste/_dl/coco/annotations_trainval2017.zip --out ~/data/waste/coco_neg
"""
import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

INDOOR = {"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "clock",
          "teddy bear", "hair drier"}
WASTE_LIKE = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "handbag", "backpack",
              "suitcase", "book", "vase", "toothbrush", "scissors", "umbrella", "frisbee", "kite"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-img", required=True)
    ap.add_argument("--zip-ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-person", type=float, default=0.05)
    a = ap.parse_args()
    ann = json.loads(zipfile.ZipFile(a.zip_ann).read("annotations/instances_val2017.json"))
    cat = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    names, person = {}, {}
    for o in ann["annotations"]:
        n = cat[o["category_id"]]
        names.setdefault(o["image_id"], set()).add(n)
        if n == "person":
            person[o["image_id"]] = max(person.get(o["image_id"], 0.0), o["area"])
    keep = []
    for iid, ns in names.items():
        im = imgs[iid]
        if person.get(iid, 0) / (im["width"] * im["height"]) < a.min_person:
            continue
        if not (ns & INDOOR) or (ns & WASTE_LIKE):
            continue
        keep.append(iid)
    keep.sort()
    out = Path(a.out).expanduser()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip_img)
    for iid in keep:
        fn = imgs[iid]["file_name"]
        buf = np.frombuffer(z.read(f"val2017/{fn}"), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        stem = Path(fn).stem
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), np.zeros(im.shape[:2], np.uint8))
    meta = {"source": "COCO val2017 (instances)", "n": len(keep), "min_person_frac": a.min_person,
            "indoor_any": sorted(INDOOR), "excluded_any": sorted(WASTE_LIKE), "image_ids": keep}
    (out / "selection.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(keep)} images -> {out}")


if __name__ == "__main__":
    main()

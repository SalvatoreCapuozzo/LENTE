"""ZeroWaste-f: estimate the conveyor belt, the only pixels that are surely NOT waste.

ZeroWaste-f annotates cardboard, soft plastic, rigid plastic and metal on a paper
sorting line; paper (and some other objects) is left unannotated. In TrashScribe
paper IS waste, so an unannotated pixel is ambiguous unless it is belt. The belt
is dark and unsaturated: those pixels become the negatives, every other
unannotated pixel is ignored by the loss and by the metrics (see `negatives_dir`
in CocoAdapter). Writes <out>/<image stem>.png, 255 = belt.

    python scripts/make_belt_masks.py ~/data/waste/zerowaste-f/train [--preview out.jpg]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

V_MAX, S_MAX = 135, 65          # HSV value / saturation (0-255) of the dark belt
MIN_AREA = 0.002                # drop belt blobs smaller than this fraction of the image


def belt_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.medianBlur(bgr, 5), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] < V_MAX) & (hsv[..., 1] < S_MAX)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_AREA * m.size
    return keep[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", help="folder with the images and labels.json")
    ap.add_argument("--out", default=None, help="default: <split_dir>/belt")
    ap.add_argument("--preview", default=None, help="write a check image of the first frames")
    a = ap.parse_args()
    d = Path(a.split_dir).expanduser()
    out = Path(a.out) if a.out else d / "belt"
    out.mkdir(parents=True, exist_ok=True)
    from pycocotools.coco import COCO
    coco = COCO(str(d / "labels.json"))
    tiles, fr_belt, fr_ann, fr_ign = [], [], [], []
    for k, iid in enumerate(coco.getImgIds()):
        info = coco.imgs[iid]
        img = cv2.imread(str(d / info["file_name"]))
        ann = np.zeros(img.shape[:2], bool)
        for an in coco.imgToAnns[iid]:
            ann |= coco.annToMask(an).astype(bool)
        belt = belt_mask(img) & ~ann
        cv2.imwrite(str(out / (Path(info["file_name"]).stem + ".png")), belt.astype(np.uint8) * 255)
        fr_belt.append(belt.mean()); fr_ann.append(ann.mean()); fr_ign.append((~belt & ~ann).mean())
        if a.preview and k % 97 == 0 and len(tiles) < 6:
            vis = img.copy()
            vis[ann] = (0.4 * vis[ann] + [0, 150, 0]).astype(np.uint8)
            vis[belt] = (0.4 * vis[belt] + [150, 0, 0]).astype(np.uint8)
            ign = ~ann & ~belt
            vis[ign] = (0.5 * vis[ign] + [0, 0, 110]).astype(np.uint8)
            tiles.append(np.hstack([cv2.resize(img, (480, 270)), cv2.resize(vis, (480, 270))]))
    if tiles:
        cv2.imwrite(a.preview, np.vstack(tiles))
    print(json.dumps({"images": len(fr_belt), "belt": round(float(np.mean(fr_belt)), 3),
                      "annotated": round(float(np.mean(fr_ann)), 3), "ignored": round(float(np.mean(fr_ign)), 3)}))


if __name__ == "__main__":
    main()

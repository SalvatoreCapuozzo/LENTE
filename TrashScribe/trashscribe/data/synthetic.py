"""Procedural compositional waste scenes: a SANITY benchmark, not a result.

Objects are combinations of material x shape rendered on five environments.
Two properties make it useful to validate the whole pipeline in minutes on a
laptop before moving to real data and an NVIDIA GPU:

* compositional hold-out: some (material, shape) pairs never appear in training,
  although both parts do, so "naming from an atlas entry" can be tested;
* environment bias: at training time each material prefers one environment
  (strength `env_bias`), at test time environments are uniform, so the cost of
  reading the background and the effect of transplant/adversary are measurable.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schema import Atlas, Entry
from .canonical import RawSample

ENVS = ["sand, soil or pebbles", "pavement or road", "grass or vegetation",
        "a conveyor belt in a recycling plant", "water"]
MATERIALS = {
    "plastic": dict(optics="opaque", surface=["glossy"], rigidity="semi-rigid"),
    "paper or cardboard": dict(optics="opaque", surface=["matte"], rigidity="semi-rigid"),
    "metal": dict(optics="opaque", surface=["metallic and reflective"], rigidity="rigid"),
    "glass": dict(optics="transparent", surface=["glossy"], rigidity="rigid"),
}
SHAPES = {
    "bottle": "bottle-shaped", "can": "cylindrical like a can or cup", "sheet": "a flat sheet",
    "bag": "a thin film or bag", "cap": "a small lid or cap", "fragment": "a small irregular fragment",
}
CLASSES = [  # (name, material, shape)
    ("plastic bottle", "plastic", "bottle"), ("glass bottle", "glass", "bottle"),
    ("metal can", "metal", "can"), ("plastic cup", "plastic", "can"),
    ("paper sheet", "paper or cardboard", "sheet"), ("metal foil", "metal", "sheet"),
    ("plastic bag", "plastic", "bag"), ("paper bag", "paper or cardboard", "bag"),
    ("plastic cap", "plastic", "cap"), ("metal cap", "metal", "cap"),
    ("glass shard", "glass", "fragment"), ("cardboard piece", "paper or cardboard", "fragment"),
]
HELDOUT = ("glass bottle", "metal foil", "paper bag")
PREFERRED_ENV = {"plastic": 0, "paper or cardboard": 3, "metal": 1, "glass": 2}


def synthetic_atlas() -> Atlas:
    ents = []
    for name, mat, shp in CLASSES:
        d = dict(material=[mat], shape=SHAPES[shp], **{k: v for k, v in MATERIALS[mat].items()})
        if shp in ("bag",) and mat == "plastic":
            d["rigidity"], d["optics"] = "flexible and deformable", "translucent"
        if shp in ("bag", "sheet") and mat == "paper or cardboard":
            d["rigidity"] = "flexible and deformable"
        ents.append(Entry(name=name, descriptors=d, prose=f"a {name}"))
    return Atlas("synthetic", ents)


def _background(env: int, S: int, rng) -> np.ndarray:
    noise = rng.normal(0, 1, (S, S, 1)).astype(np.float32)
    if env == 0:      # sand
        img = np.array([194, 178, 128], np.float32) + 18 * noise
        for _ in range(int(rng.integers(20, 60))):
            c = tuple(int(v) for v in rng.integers(90, 160, 3))
            cv2.circle(img, tuple(int(v) for v in rng.integers(0, S, 2)), int(rng.integers(1, 4)), c, -1)
    elif env == 1:    # pavement
        img = np.array([95, 95, 100], np.float32) + 22 * noise
        for _ in range(int(rng.integers(2, 6))):
            p = rng.integers(0, S, 4)
            cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (55, 55, 58), 1)
    elif env == 2:    # grass
        img = np.array([60, 120, 45], np.float32) + 25 * noise
        for _ in range(200):
            x, y = (int(v) for v in rng.integers(0, S, 2))
            cv2.line(img, (x, y), (x + int(rng.integers(-3, 4)), y - int(rng.integers(3, 9))), (40, 150, 40), 1)
    elif env == 3:    # conveyor
        img = np.array([45, 45, 50], np.float32) + 10 * noise
        for y in range(0, S, int(rng.integers(12, 24))):
            cv2.line(img, (0, y), (S, y), (30, 30, 34), 2)
    else:             # water
        img = np.array([40, 90, 140], np.float32) + 12 * noise
        for y in range(0, S, 7):
            cv2.line(img, (0, y), (S, y + int(rng.integers(-3, 4))), (60, 115, 165), 1)
    return np.clip(img, 0, 255)


def _shape_mask(shape: str, S: int, rng) -> np.ndarray:
    m = np.zeros((S, S), np.uint8)
    c = rng.integers(S // 6, S - S // 6, 2)
    r = int(rng.integers(S // 16, S // 7))
    ang = float(rng.uniform(0, 180))
    if shape == "bottle":
        rect = ((float(c[0]), float(c[1])), (r * 0.9, r * 2.6), ang)
        cv2.fillPoly(m, [cv2.boxPoints(rect).astype(np.int32)], 1)
        t = np.deg2rad(ang)
        neck = ((float(c[0] - np.sin(t) * r * 1.6), float(c[1] + np.cos(t) * r * 1.6)), (r * 0.35, r * 0.9), ang)
        cv2.fillPoly(m, [cv2.boxPoints(neck).astype(np.int32)], 1)
    elif shape == "can":
        rect = ((float(c[0]), float(c[1])), (r * 1.1, r * 1.8), ang)
        cv2.fillPoly(m, [cv2.boxPoints(rect).astype(np.int32)], 1)
    elif shape in ("sheet", "bag"):
        n = int(rng.integers(6, 10))
        th = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = r * rng.uniform(0.9 if shape == "sheet" else 0.6, 1.5, n)
        pts = np.stack([c[0] + rad * np.cos(th), c[1] + rad * np.sin(th)], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        if shape == "bag":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    elif shape == "cap":
        cv2.circle(m, (int(c[0]), int(c[1])), max(4, r // 3), 1, -1)
    else:  # fragment
        n = int(rng.integers(3, 6))
        th = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = (r // 2) * rng.uniform(0.5, 1.4, n)
        pts = np.stack([c[0] + rad * np.cos(th), c[1] + rad * np.sin(th)], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def _paint(img, m, material, rng):
    S = img.shape[0]
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    mm = m.astype(bool)
    if material == "plastic":
        col = cv2.cvtColor(np.uint8([[[int(rng.integers(0, 180)), 220, 230]]]), cv2.COLOR_HSV2RGB)[0, 0]
        tex = np.ones((S, S, 1), np.float32) * col
        hl = np.exp(-(((xx - xx[mm].mean()) ** 2 + (yy - yy[mm].min() - 5) ** 2) / 60.0))[..., None]
        tex = tex + 90 * hl
        alpha = 1.0
    elif material == "paper or cardboard":
        base = rng.choice([np.array([200, 170, 120]), np.array([235, 232, 220]), np.array([160, 120, 80])])
        tex = base + 14 * rng.normal(0, 1, (S, S, 1))
        alpha = 1.0
    elif material == "metal":
        g = (xx - xx[mm].min()) / max(1.0, float(np.ptp(xx[mm])))
        tex = (120 + 110 * np.abs(np.sin(g * np.pi * 2.5)))[..., None] * np.ones(3)
        alpha = 1.0
    else:  # glass
        tint = rng.choice([np.array([90, 160, 90]), np.array([140, 90, 40]), np.array([220, 230, 230])])
        tex = np.ones((S, S, 1)) * tint
        alpha = 0.45
    a = mm[..., None] * alpha
    img[:] = img * (1 - a) + tex * a
    if material == "glass":
        edge = (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)
        img[edge] = 240


class SyntheticAdapter:
    def __init__(self, name="synthetic", n=600, size=256, seed=0, split="train", env_bias=0.8,
                 objects=(1, 4), heldout=HELDOUT):
        self.name, self.n, self.size, self.seed, self.split = name, n, size, seed, split
        self.env_bias = env_bias if split == "train" else 0.0
        self.objects = objects
        # held-out compositions exist only in test: validation drives model
        # selection, so it must not see them either
        self.classes = [c for c in CLASSES if (split == "test" or c[0] not in heldout)]
        self.items = list(range(n))
        self.entries_per_item = [set() for _ in self.items]

    def __len__(self):
        return len(self.items)

    def drop_items_with(self, entries):
        return 0

    def entries_present(self):
        return {c[0] for c in self.classes}

    def load(self, i) -> RawSample:
        split_code = {"train": 0, "val": 1, "test": 2}.get(self.split, 3)   # str hash() is salted per process
        rng = np.random.default_rng((self.seed * 1_000_003 + i * 7919 + split_code * 104_729) % 2 ** 32)
        S = self.size
        k = int(rng.integers(self.objects[0], self.objects[1] + 1))
        picks = [self.classes[int(rng.integers(len(self.classes)))] for _ in range(k)]
        main_mat = picks[0][1]
        if rng.random() < self.env_bias:
            env = PREFERRED_ENV[main_mat]
        else:
            env = int(rng.integers(len(ENVS)))
        img = _background(env, S, rng)
        insts, ents = [], []
        for name, mat, shp in picks:
            m = _shape_mask(shp, S, rng)
            _paint(img, m, mat, rng)
            insts = [(q & (1 - m)).astype(np.uint8) for q in insts]
            insts.append(m)
            ents.append(name)
        keep = [j for j, q in enumerate(insts) if q.sum() >= 16]
        return RawSample(image=np.clip(img, 0, 255).astype(np.uint8), instances=[insts[j] for j in keep],
                         entries=[ents[j] for j in keep], dataset=f"{self.name}-{ENVS[env].split(',')[0]}",
                         image_id=f"{self.split}{i}", surroundings=ENVS[env], has_instances=True)

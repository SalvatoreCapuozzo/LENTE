"""Training-free references on the test split of a config.

  where     OWLv2 text-queried boxes -> SAM2.1-Tiny box prompts -> union = waste mask
  describe  raw CLIP on each ground-truth object crop vs the descriptor phrases
            (the "raw CLIP" column of the MIRF tables): descriptor mAP
  name      raw CLIP crop vs atlas prose, generalised zero-shot top-1

    python scripts/baseline_zeroshot.py configs/exp/e4_heldout_classes.yaml [--max-images 300]
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trashscribe import device as D
from trashscribe.data.canonical import IMAGENET_MEAN, IMAGENET_STD, Canonicalizer, collate
from trashscribe.data.mixing import MultiDataset
from trashscribe.data.registry import build_split
from trashscribe.engine import _p, load_config, load_vocab
from trashscribe.metrics import SegMeter, descriptor_map, gzsl
from trashscribe.schema import Atlas

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--owl", default="google/owlv2-base-patch16")
ap.add_argument("--sam", default="facebook/sam2.1-hiera-tiny")
ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
ap.add_argument("--queries", nargs="*", default=["a piece of litter", "trash", "garbage", "plastic waste",
                                                 "a discarded bottle", "a discarded can"])
ap.add_argument("--box-thr", type=float, default=0.15)
ap.add_argument("--max-images", type=int, default=0)
ap.add_argument("--tasks", nargs="*", default=["where", "describe"])
a = ap.parse_args()
cfg = load_config(a.config)
dev = D.pick_device(cfg.get("device", "auto"))
schema, atlas = load_vocab(cfg)
root = _p(".").__class__(cfg.get("data_root", ".")).expanduser()
S = cfg["model"]["image_size"]
ad = build_split(cfg["datasets"], cfg.get("test_datasets"), "test", atlas, root, set(), a.max_images or None,
                 strict=cfg.get("strict_atlas", True))
dl = DataLoader(MultiDataset(ad, Canonicalizer(schema, atlas, S, False)), batch_size=1, collate_fn=collate)
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
res = {"config": a.config}


def _vec(o):
    """transformers 5 returns an output object; its pooler_output is the projected embedding."""
    return o if torch.is_tensor(o) else o.pooler_output

if "where" in a.tasks:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor
    owl_p, owl = Owlv2Processor.from_pretrained(a.owl), Owlv2ForObjectDetection.from_pretrained(a.owl).to(dev).eval()
    sam_p, sam = Sam2Processor.from_pretrained(a.sam), Sam2Model.from_pretrained(a.sam).to(dev).eval()
    seg = SegMeter()
    for i, b in enumerate(dl):
        img = ((b["image"][0] * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        with torch.no_grad():
            inp = owl_p(text=[a.queries], images=img, return_tensors="pt").to(dev)
            det = owl_p.post_process_grounded_object_detection(owl(**inp), threshold=a.box_thr,
                                                      target_sizes=torch.tensor([[S, S]]))[0]
            pred = torch.zeros(1, 1, S, S)
            if len(det["boxes"]):
                si = sam_p(images=img, input_boxes=[det["boxes"].cpu().tolist()], return_tensors="pt").to(dev)
                o = sam(**si, multimask_output=False)
                m = sam_p.post_process_masks(o.pred_masks.cpu(), si["original_sizes"].cpu())[0]
                pred = (m.float().amax(0, keepdim=True).amax(1, keepdim=True) > 0).float().view(1, 1, S, S)
        seg.update(pred, b["waste"], b["valid"], b["dataset"])
        if a.max_images and i + 1 >= a.max_images:
            break
    res["where_owlv2_sam2"] = seg.result()

if "describe" in a.tasks:
    from transformers import CLIPModel, CLIPProcessor
    cp, cm = CLIPProcessor.from_pretrained(a.clip), CLIPModel.from_pretrained(a.clip).to(dev).eval()
    ext = schema
    with torch.no_grad():
        txt = []
        for ens in ext.phrases():
            t = cp(text=ens, return_tensors="pt", padding=True).to(dev)
            e = F.normalize(_vec(cm.get_text_features(**t)), dim=-1).mean(0)
            txt.append(F.normalize(e, dim=-1))
        T = torch.stack(txt)
        full = Atlas.merge([atlas] + [Atlas.load(_p(p)) for p in cfg.get("novel_atlases", [])])
        tp = cp(text=[e.prose or e.name for e in full.entries], return_tensors="pt", padding=True).to(dev)
        PR = F.normalize(_vec(cm.get_text_features(**tp)), dim=-1)
    P, TG, MK, ENT, NAMED = [], [], [], [], []
    for i, b in enumerate(dl):
        img = ((b["image"][0] * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        for m, t, k, e in zip(b["inst"][0], b["desc_t"][0], b["desc_m"][0], b["entries"][0]):
            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = 4 * ys.min(), 4 * (ys.max() + 1), 4 * xs.min(), 4 * (xs.max() + 1)
            cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, (y1 - y0) * 1.35, (x1 - x0) * 1.35   # MIRF tight crop
            crop = img[max(0, int(cy - h / 2)):int(cy + h / 2) + 1, max(0, int(cx - w / 2)):int(cx + w / 2) + 1]
            with torch.no_grad():
                z = F.normalize(_vec(cm.get_image_features(**cp(images=crop, return_tensors="pt").to(dev))), dim=-1)
            lg = 100 * (z @ T.T)[0].cpu()
            post = torch.empty_like(lg)
            for f in ext.fields:
                sl = ext.slices[f.name]
                post[sl] = lg[sl].softmax(-1) if f.kind == "single" else torch.sigmoid(lg[sl] - lg[sl].mean())
            P.append(post.numpy())
            TG.append(t.numpy())
            MK.append(k.numpy())
            ENT.append(e)
            NAMED.append(full.entries[int((z @ PR.T).argmax())].name)
        if a.max_images and i + 1 >= a.max_images:
            break
    if P:
        res["describe_raw_clip"] = descriptor_map(np.stack(P), np.stack(TG), np.stack(MK), schema)
        hold = set(cfg.get("holdout") or [])
        idx = [j for j, e in enumerate(ENT) if e is not None and e in full and full[e].descriptors]
        res["name_raw_clip_prose"] = gzsl([NAMED[j] for j in idx], [ENT[j] for j in idx],
                                          {ENT[j] for j in idx} - hold, {ENT[j] for j in idx} & hold)
out = _p(cfg.get("output_dir", "runs")) / f"{cfg['run_name']}__zeroshot"
out.mkdir(parents=True, exist_ok=True)
(out / "test_metrics.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: (v.get("_all") if isinstance(v, dict) and "_all" in v else v.get("_macro_intrinsic", v)
                      if isinstance(v, dict) else v) for k, v in res.items()}, indent=1, default=float))

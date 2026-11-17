"""Training-free references on the test split of a config.

  where     OWLv2 (or, --detector gdino, Grounding DINO-T) text-queried boxes -> SAM2.1-Tiny box
            prompts -> union = waste mask. Grounding DINO: box 0.3 / text 0.25 and boxes larger
            than --max-box-area of the image dropped ("trash" also fires on the whole scene).
            --detector sam3: SAM 3 (promptable concept segmentation) returns instance masks for
            each noun phrase of --sam3-prompts directly (score >= 0.5); their union is the mask.
  describe  raw CLIP on each ground-truth object crop vs the descriptor phrases
            (the "raw CLIP" column of the MIRF tables): descriptor mAP
  name      raw CLIP crop vs atlas prose, generalised zero-shot top-1
  pseudo    (--detector sam3) SAM 3 as an OFFLINE TEACHER: masks for the train / val images of the
            config's test dataset (E9: the site the model must be adapted to; labels never used).
            Rule fixed before any result: instances with score >= 0.5 are waste, instances with
            0.3 <= score < 0.5 or covering more than half the image are ignored (255), the rest is
            background. Written as a `masks` dataset under <data_root>/_pseudo/sam3/<dataset>/<split>.
            Quality against the real labels of those images is computed for the record only.

    python scripts/baseline_zeroshot.py configs/exp/e4_heldout_classes.yaml [--max-images 300]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time

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
from trashscribe.progress import Progress
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
ap.add_argument("--detector", default="owlv2", choices=["owlv2", "gdino", "sam3"])
ap.add_argument("--sam3", default="facebook/sam3")
ap.add_argument("--sam3-prompt-set", default="waste4", choices=["waste4", "objects16"])
ap.add_argument("--sam3-prompts", nargs="*", default=None, help="overrides --sam3-prompt-set")
ap.add_argument("--sam3-thr", type=float, default=0.5)
ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-tiny")
ap.add_argument("--max-box-area", type=float, default=0.5)
ap.add_argument("--device", default=None)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--tag", default="", help="suffix of the output folder (re-runs keep earlier results)")
ap.add_argument("--pseudo-splits", nargs="*", default=["train", "val"])
ap.add_argument("--pseudo-lo", type=float, default=0.3)
a = ap.parse_args()
# SAM 3 prompt sets, fixed before any result (15 Nov). waste4: generic waste words. objects16: the
# same plus twelve waste-object nouns after the most frequent TACO supercategories. SAM 3's presence
# head judges "litter"/"trash" absent from scenes that are not litter scenes (an office where a person
# holds a bottle: presence 0.02-0.26), while "bottle" is found there (0.995).
SAM3_SETS = {"waste4": ["litter", "trash", "garbage", "plastic waste"],
             "objects16": ["litter", "trash", "garbage", "plastic waste", "bottle", "can", "cup", "plastic bag",
                           "food wrapper", "carton", "cardboard box", "paper", "bottle cap", "plastic fragment",
                           "styrofoam piece", "cigarette butt"]}
if a.sam3_prompts is None:
    a.sam3_prompts = SAM3_SETS[a.sam3_prompt_set]
cfg = load_config(a.config)
dev = D.pick_device(a.device or cfg.get("device", "auto"))
if a.threads:
    torch.set_num_threads(a.threads)
schema, atlas = load_vocab(cfg)
root = _p(".").__class__(cfg.get("data_root", ".")).expanduser()
S = cfg["model"]["image_size"]
ad = build_split(cfg["datasets"], cfg.get("test_datasets"), "test", atlas, root, set(), a.max_images or None,
                 strict=cfg.get("strict_atlas", True))
dl = DataLoader(MultiDataset(ad, Canonicalizer(schema, atlas, S, False)), batch_size=1, collate_fn=collate)
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
res = {"config": a.config}
_suffix = "" if a.detector == "owlv2" else "_" + a.detector + ("obj" if a.detector == "sam3" and
                                                              a.sam3_prompt_set == "objects16" else "")
out = _p(cfg.get("output_dir", "runs")) / f"{cfg['run_name']}__zeroshot{_suffix}{'_' + a.tag if a.tag else ''}"
out.mkdir(parents=True, exist_ok=True)
prog = Progress(out)
n_img = min(len(dl), a.max_images) if a.max_images else len(dl)


@torch.no_grad()
def _sam3_text(model, proc, prompts, dev):
    """Text embeddings of the prompts, computed once (the text encoder does not depend on the image)."""
    out = []
    for q in prompts:
        t = proc(text=q, return_tensors="pt").to(dev)
        e = model.get_text_features(input_ids=t.input_ids, attention_mask=t.attention_mask, return_dict=True)
        out.append((e.pooler_output, t.attention_mask))
    return out


def _vec(o):
    """transformers 5 returns an output object; its pooler_output is the projected embedding."""
    return o if torch.is_tensor(o) else o.pooler_output

if "pseudo" in a.tasks:
    assert a.detector == "sam3", "pseudo-labels come from SAM 3"
    import cv2
    from trashscribe.data.canonical import IGNORE
    from transformers import Sam3Model, Sam3Processor
    s3p = Sam3Processor.from_pretrained(a.sam3)
    s3 = Sam3Model.from_pretrained(a.sam3, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    s3_txt = _sam3_text(s3, s3p, a.sam3_prompts, dev)
    quality = {}
    for ds in cfg.get("test_datasets"):
        for split in a.pseudo_splits:
            ads = build_split(cfg["datasets"], [ds], split, atlas, root, set(), a.max_images or None,
                              strict=cfg.get("strict_atlas", True))
            if not ads:
                continue
            adp = ads[0]
            od = root / "_pseudo" / "sam3" / ds / split
            (od / "images").mkdir(parents=True, exist_ok=True)
            (od / "masks").mkdir(parents=True, exist_ok=True)
            tp = fp = fn = n_ign = n_pix = 0
            prog(phase=f"pseudo-labels {ds}/{split}", batch=0, batches=len(adp))
            for i in range(len(adp)):
                prog(batch=i + 1)
                smp = adp.load(i)
                img = smp.image
                h, w = img.shape[:2]
                with torch.no_grad():
                    inp = s3p(images=img, return_tensors="pt").to(dev)
                    vis = s3.get_vision_features(pixel_values=inp.pixel_values.to(s3.dtype))
                    pos = np.zeros((h, w), bool)
                    unsure = np.zeros((h, w), bool)
                    for ti in s3_txt:
                        o = s3(vision_embeds=vis, text_embeds=ti[0], attention_mask=ti[1])
                        for k, v in list(o.items()):
                            if torch.is_tensor(v) and v.is_floating_point():
                                o[k] = v.float()
                        d3 = s3p.post_process_instance_segmentation(o, threshold=a.pseudo_lo, mask_threshold=0.5,
                                                                    target_sizes=[(h, w)])[0]
                        for m, sc in zip(d3["masks"].cpu().numpy().astype(bool), d3["scores"].cpu().numpy()):
                            if sc >= a.sam3_thr and m.mean() <= a.max_box_area:
                                pos |= m
                            else:
                                unsure |= m
                lab = np.zeros((h, w), np.uint8)
                lab[unsure & ~pos] = 255
                lab[pos] = 1
                stem = f"{i:05d}_" + "".join(c if c.isalnum() else "_" for c in str(smp.image_id))[-60:]
                cv2.imwrite(str(od / "images" / f"{stem}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(od / "masks" / f"{stem}.png"), lab)
                gt = smp.waste_mask().copy()
                if smp.valid is not None:
                    gt[smp.valid == 0] = IGNORE
                ok = (gt != IGNORE) & (lab != 255)
                pr, g = (lab == 1) & ok, (gt == 1) & ok
                tp += int((pr & g).sum()); fp += int((pr & ~g).sum()); fn += int((~pr & g).sum())
                n_ign += int((lab == 255).sum()); n_pix += h * w
                if a.max_images and i + 1 >= a.max_images:
                    break
            quality[f"{ds}/{split}"] = {"n_images": i + 1, "iou_vs_labels": tp / max(tp + fp + fn, 1),
                                        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                                        "ignored_frac": n_ign / max(n_pix, 1), "dir": str(od)}
            print(f"pseudo {ds}/{split}", json.dumps(quality[f"{ds}/{split}"]), flush=True)
    res["pseudo_sam3"] = {"rule": {"waste_score": a.sam3_thr, "ignore_from": a.pseudo_lo,
                                   "ignore_if_area_above": a.max_box_area, "prompts": a.sam3_prompts},
                          "quality_vs_labels_record_only": quality}
    del s3

if "where" in a.tasks:
    prog(phase=f"where ({a.detector}{'' if a.detector == 'sam3' else ' -> SAM2'})", batch=0, batches=n_img)
    from transformers import Sam2Model, Sam2Processor
    if a.detector == "owlv2":
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        owl_p, owl = Owlv2Processor.from_pretrained(a.owl), Owlv2ForObjectDetection.from_pretrained(a.owl).to(dev).eval()
    elif a.detector == "sam3":
        from transformers import Sam3Model, Sam3Processor
        s3p = Sam3Processor.from_pretrained(a.sam3)
        s3 = Sam3Model.from_pretrained(a.sam3, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
        s3 = s3.to(dev).eval()
        s3_txt = _sam3_text(s3, s3p, a.sam3_prompts, dev)
    else:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        gd_p = AutoProcessor.from_pretrained(a.gdino)
        gd = AutoModelForZeroShotObjectDetection.from_pretrained(a.gdino).to(dev).eval()
        gd_text = ". ".join(a.queries) + "."
    if a.detector != "sam3":
        sam_p, sam = Sam2Processor.from_pretrained(a.sam), Sam2Model.from_pretrained(a.sam).to(dev).eval()
    seg = SegMeter()
    seg_area = SegMeter()     # SAM 3 only: instances larger than --max-box-area of the image dropped
    for i, b in enumerate(dl):
        prog(batch=i + 1)
        img = ((b["image"][0] * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        with torch.no_grad():
            if a.detector == "sam3":
                inp = s3p(images=img, return_tensors="pt").to(dev)
                vis = s3.get_vision_features(pixel_values=inp.pixel_values.to(s3.dtype))
                pred = torch.zeros(1, 1, S, S)
                pred_area = torch.zeros(1, 1, S, S)
                # image extent inside the letterbox (padding is black, bottom/right): the same
                # half-image rule as the Grounding DINO boxes and the E9 teacher labels (16 Nov)
                rows, cols = np.nonzero(img.max(2).max(1))[0], np.nonzero(img.max(2).max(0))[0]
                img_area = float((rows.max() + 1) * (cols.max() + 1)) if len(rows) else float(S * S)
                for ti in s3_txt:
                    o = s3(vision_embeds=vis, text_embeds=ti[0], attention_mask=ti[1])
                    for k, v in list(o.items()):     # bf16 on the GPU: post-processing in fp32
                        if torch.is_tensor(v) and v.is_floating_point():
                            o[k] = v.float()
                    d3 = s3p.post_process_instance_segmentation(o, threshold=a.sam3_thr, mask_threshold=0.5,
                                                                target_sizes=[(S, S)])[0]
                    if len(d3["masks"]):
                        ms = d3["masks"].float().cpu()
                        pred = torch.maximum(pred, ms.amax(0).view(1, 1, S, S))
                        small = ms.flatten(1).sum(1) <= a.max_box_area * img_area
                        if bool(small.any()):
                            pred_area = torch.maximum(pred_area, ms[small].amax(0).view(1, 1, S, S))
                seg.update(pred, b["waste"], b["valid"], b["dataset"])
                seg_area.update(pred_area, b["waste"], b["valid"], b["dataset"], boundary=False)
                if a.max_images and i + 1 >= a.max_images:
                    break
                continue
            if a.detector == "owlv2":
                inp = owl_p(text=[a.queries], images=img, return_tensors="pt").to(dev)
                det = owl_p.post_process_grounded_object_detection(owl(**inp), threshold=a.box_thr,
                                                          target_sizes=torch.tensor([[S, S]]))[0]
            else:
                inp = gd_p(images=img, text=gd_text, return_tensors="pt").to(dev)
                det = gd_p.post_process_grounded_object_detection(gd(**inp), inp.input_ids, threshold=0.3,
                                                                  text_threshold=0.25, target_sizes=[(S, S)])[0]
                bx = det["boxes"]
                # image extent inside the letterbox (padding is black, bottom/right)
                rows, cols = np.nonzero(img.max(2).max(1))[0], np.nonzero(img.max(2).max(0))[0]
                img_area = float((rows.max() + 1) * (cols.max() + 1)) if len(rows) else float(S * S)
                keep = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) <= a.max_box_area * img_area
                det = {"boxes": bx[keep]}
            pred = torch.zeros(1, 1, S, S)
            if len(det["boxes"]):
                si = sam_p(images=img, input_boxes=[det["boxes"].cpu().tolist()], return_tensors="pt").to(dev)
                o = sam(**si, multimask_output=False)
                m = sam_p.post_process_masks(o.pred_masks.cpu(), si["original_sizes"].cpu())[0]
                pred = (m.float().amax(0, keepdim=True).amax(1, keepdim=True) > 0).float().view(1, 1, S, S)
        seg.update(pred, b["waste"], b["valid"], b["dataset"])
        if a.max_images and i + 1 >= a.max_images:
            break
    res[f"where_{a.detector}" if a.detector == "sam3" else f"where_{a.detector}_sam2"] = seg.result()
    if a.detector == "sam3":
        res["where_sam3_area"] = seg_area.result()

if "describe" in a.tasks:
    prog(phase="describe (raw CLIP crops)", batch=0, batches=n_img)
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
        prog(batch=i + 1)
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
        hold = set(cfg.get("holdout") or []) | {e for d in cfg["datasets"] for e in d.get("eval_only_entries") or []}
        idx = [j for j, e in enumerate(ENT) if e is not None and e in full and full[e].descriptors]
        res["name_raw_clip_prose"] = gzsl([NAMED[j] for j in idx], [ENT[j] for j in idx],
                                          {ENT[j] for j in idx} - hold, {ENT[j] for j in idx} & hold)
(out / "test_metrics.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: (v.get("_all") if isinstance(v, dict) and "_all" in v else v.get("_macro_intrinsic", v)
                      if isinstance(v, dict) else v) for k, v in res.items()}, indent=1, default=float))
prog(phase="done", eval_done=time.time())

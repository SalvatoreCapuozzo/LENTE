# Experiments: protocols, configs, results, reading

Numbers are fractions (IoU, AP) on the **test split**, as of 20 Nov 2026 (grid finished). The always-current tables
are at the bottom of [RESULTS.md](../RESULTS.md) (regenerated every 15 min) and in `runs/<run>/test_metrics.json`.
All runs are **single-seed**: treat differences below about 3 mAP points or 0.1 in H as noise.

## Protocols and configs (`configs/exp/`)

All configs inherit `configs/base.yaml` (1024 px, 40 epochs × 4,000 samples, optimizer batch 8 = micro-batch 4 × 2,
AdamW, bf16, early stop patience 8 on val IoU + desc mAP). Seven TACO classes are held out in E4 and its
derivatives (Glass bottle, Pop tab, Plastic straw, Crisp packet, Paper bag, Foam cup, Egg carton; Egg carton has
no test object), plus BePLi's fishing net as a text-only class: seven unseen classes are scored.

| ID | Config(s) | Question | Status |
|---|---|---|---|
| E1 | `e1_indomain_<ds>` | one dataset alone (in-domain reference) | 5/5 done (BePLi 0.650) |
| E2 | `e2_multi_domain` | all datasets, no held-out classes | removed 13 Nov (E4 serves as multi-domain model) |
| E3 | `e3_lodo_<ds>` | leave one dataset out of training and selection | 5/5 done |
| E3b | `e3b_lodo_mju_negatives` | E3-MJU + 243 COCO indoor negatives | done |
| E4 | `e4_heldout_classes` | all five datasets, 7 held-out classes (main model) | done |
| E5 | `e5_ablate_*` | ablations of E4 (probe, no_dense, no_debias, no_transplant, no_adversary, no_mixstyle, lente_rgb; lora8 / adapters_both / geometry removed 13 Nov) | 7 done |
| E6 | `e6_edge_640` | E4 recipe at 640 px (deployment model) | done, exported to ONNX; on the Jetson (JetPack 6.2, 24 Nov): docs/JETSON.md |
| E7 | `e7_heldout_values` | E4 with 3 descriptor values never supervised | done |
| E8 | `e8_clip_distill` | E7 + dense head distilled from CLIP crop embeddings | done |
| E9 | `e9_sam3_<ds>` | LODO model fine-tuned with SAM 3 labels of the site's unlabelled images | 5/5 done |
| E10 | `e10_k10_<ds>`, `e10_k50_uavvaste` | LODO model fine-tuned with 10 (50) labelled site images | 6/6 done |
| baselines | `<cfg>__segformer_mit-b2`, `e4_heldout_classes__zeroshot*`, `e3_lodo_<ds>__zeroshot_sam3obj` | SegFormer-B2 LODO; OWLv2→SAM2, Grounding DINO→SAM2, SAM 3 zero-shot; SAM 3 pseudo-label jobs for E9 | SegFormer MJU done (4 queued); OWLv2, SAM 3 (16 prompts, raw + fair) done; SAM 3 4-word and Grounding DINO queued |

**SAM 3 evaluation rule.** 16 prompts (4 waste words: litter, trash, garbage, plastic waste; 12 object nouns
after TACO supercategories: bottle, can, cup, plastic bag, food wrapper, carton, cardboard box, paper, bottle
cap, plastic fragment, styrofoam piece, cigarette butt), score ≥ 0.5, union of instances. "Fair" = instances
larger than half the image dropped, the same rule as the Grounding DINO boxes and the E9 labels. With the 4
waste words alone SAM 3's presence head rejects waste in non-litter scenes (office: presence 0.02–0.26).

## Main results

### Segmentation (waste IoU)

| run | TACO | ZeroWaste-f | UAVVaste | MJU-Waste | BePLi v2 | pooled |
|---|---|---|---|---|---|---|
| E4 (all five, main) | 0.885 | 0.912 | 0.490 | 0.895 | 0.616 | 0.870 |
| E6 (640 px) | 0.866 | 0.903 | 0.451 | 0.888 | 0.579 | 0.856 |
| E1 (in-domain) | 0.862 | 0.922 | 0.616 | 0.914 | queued | |
| E3 (LODO) | 0.630 | 0.747 | 0.262 | 0.055 | 0.282 | |
| SegFormer-B2 LODO | queued | queued | queued | 0.104 | queued | |
| OWLv2→SAM2 zero-shot | 0.121 | 0.243 | 0.019 | 0.125 | 0.137 | 0.191 |
| SAM 3 zero-shot, 16 prompts, raw | 0.759 | 0.793 | 0.424 | 0.221 | 0.272 | 0.579 |
| SAM 3 zero-shot, 16 prompts, fair | 0.713 | 0.783 | 0.424 | 0.221 | 0.424 | 0.613 |

### Edge model vs SAM 3 by amount of site data (paper Fig. `ladder`, Table `tab:ladder`)

| site | SAM 3 (fair, no site data) | LODO | SegFormer LODO | + negatives (E3b) | + SAM 3 labels (E9) | + 10 labels (E10) | + 50 labels | site seen (E4) |
|---|---|---|---|---|---|---|---|---|
| TACO | 0.713 (raw 0.759) | 0.630 | **0.746** | – | **0.783** | **0.733** | – | **0.885** |
| ZeroWaste-f | 0.783 (raw 0.793) | 0.747 | 0.741 | – | **0.805** | **0.843** | – | **0.912** |
| UAVVaste | 0.424 | 0.262 | **0.480** | – | 0.375 | 0.289 | 0.294 | **0.490** |
| MJU-Waste | 0.221 | 0.055 | 0.104 | **0.363** | **0.231** | **0.751** | – | **0.895** |
| BePLi v2 | 0.424 (raw 0.272) | 0.282 | 0.276 | – | 0.398 | **0.454** | – | **0.616** |

Bold = at or above SAM 3. SAM 3 label quality against the real annotations of the site's training images
(recorded only, never used): TACO 0.757, UAVVaste 0.485, MJU 0.236 (precision 0.24: desk objects labelled
too), BePLi 0.492. Students end close to their labels; longer E9 training was rejected for that reason.
Rule fixed before E10 results: a site below SAM 3 with 10 images gets a 50-image run (UAVVaste).

### Description (E4 test objects, 27,853)

Macro mAP over intrinsic fields 0.691 (prevalence floor 0.226, raw CLIP on crops 0.305, closed per-value
probe 0.712). E6 (640 px) 0.632. Per field and LODO description: RESULTS.md. Without the dense pixel loss
0.649 (−4.2, the one ablation that matters).

### Open vocabulary (unseen classes and unseen values)

| test | raw CLIP crops | E4/E7 text head | E8 distilled field |
|---|---|---|---|
| retrieval of 7 unseen classes, mean AP (class name) | 0.078 (7/7 p<0.01) | fused 0.034 (E4) | **0.082** (7/7) |
| naming seen / unseen / H | 0.182 / 0.291 / 0.217 | E4 0.182 / 0.029 / 0.049; keep-70 % rule H 0.155 | 0.086 / 0.198 / 0.120 |
| unseen value metal (floor 0.006) | 0.033 | E7 0.140 | 0.053 |
| unseen value foam (floor 0.097) | 0.114 | E7 0.096 | **0.152** |
| unseen value crumpled (floor 0.083) | 0.276 | E7 0.062 | 0.240 |

Fixed fusion of E8 head + field: 0.133 / 0.113 / 0.147 (all above floor, below the better source on each).

### Environment

Pooled leakage probe 0.985 vs majority 0.748 (confounded: the class predicts the place). Within-class probe
(15 classes, 9,191 objects): intrinsic embedding +6.9 points over majority, context ring +20.7; without the
adversary +9.7, without transplant +6.8. Ablations: no debiasing component changes seen-environment results.

### Scale (test-time 2×2 tiling, `posthoc.py tiles`)

UAVVaste: E4 0.490 → 0.529, LODO 0.262 → 0.295, but aerial-only E1 0.616 → 0.538 and E9-UAVVaste 0.375 →
0.304 (mean of full + tiles 0.391). Joint training learns ground-level object sizes; scale augmentation is
the proposed (not run) training-side fix.

## Post-hoc artefacts

`runs/<run>/posthoc/` (dumps `test.pt`/`val.pt`, `naming.json`, `clipret*.json`, `tiles2.json`) and
`runs/_posthoc_*.json` (leakage, AUSUC corrected, value APs for E4/E7/E8). All produced by `scripts/posthoc.py`.

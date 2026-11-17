# Architecture and code map

All paths are relative to the TrashScribe folder. `PKG_ROOT` (in `trashscribe/engine.py`) is this folder:
every relative path in configs (`runs/...`, `atlas/...`, `configs/...`) resolves against it, and dataset
paths resolve against `data_root` (default `~/data/waste`, in `configs/base.yaml`).

## 1. Data flow

```
datasets (5 public, native formats) ─► adapters ─► RawSample (image, waste mask, instance masks, atlas entry
                                                    per instance, surroundings value, valid mask)
        ─► letterbox to S (1024 px; 640 for E6) ─► augmentation ─► optional cross-dataset object transplant
        ─► collate ─► batch {image, waste, valid, inst (stride 4), desc_t/desc_m (targets from the atlas),
                             contour, env_id, entries, dataset}

network (one forward pass):
  RGB ─► SAM2.1-Hiera-Tiny (frozen) + zero-init adapters (+ optional LoRA, MixStyle at train time)
        ├─► SAM2 mask decoder, prompt-free ─► waste logits (stride 4)               "where"
        ├─► contour head (stride-4 features + descriptor features) ─► boundaries     "split"
        ├─► descriptor field F: 512-d per pixel at stride 8, CLIP ViT-B/16 TEXT space  "what"
        └─► (E8 only) ClipField: 512-d per pixel at stride 8, CLIP ViT-B/16 IMAGE space

after the network (cheap, also on the device):
  instances = connected components of waste away from contours, then nearest-seed assignment
  z_int = mask-pooled F inside the object; z_ctx = F pooled on a 3-cell ring around it
  descriptor scores s(z,d) = tau_f * cos(z, W_t e_d) + b_f   (e_d = frozen CLIP text embedding of value d)
  posterior: softmax per single-valued field, sigmoid per multi-valued field
  naming: atlas entries ranked by 3 evidence streams (descriptor agreement, text prototype, prose),
          seen-class penalty gamma (calibrated stacking), abstention thresholds
  report: coverage, count, per-object description + name/set-aside + stream, composition, sentence
```

The text path (schema phrases and atlas sentences → CLIP text encoder → `atlas_cache.pt`) runs once on a
workstation; the device never runs the text encoder.

## 2. Vocabulary: schema and atlas (`atlas/`, `trashscribe/schema.py`)

* `atlas/schema.yaml`: 9 fields, 59 values. Fields are single- or multi-valued, intrinsic (property of the
  object) or contextual (`surroundings`, read from the ring, naming weight 0). Each value has a prompt
  ensemble ("an object made of {}", ...).
* `atlas/taco.yaml` (60 TACO classes), `zerowaste.yaml` (4), `bepli.yaml` (BePLi classes as aliases of TACO
  entries or 5 new entries), `binary.yaml` (single-class datasets), `novel_field_classes.yaml` (classes in no
  training set: fishing net, face mask, vape, tyre; used for text-only naming/retrieval).
* An atlas entry = descriptor profile (fields it states) + prose sentence + collection stream. Unstated
  fields are unknown: neither supervised nor used for naming (`entry_target` builds target + mask).

## 3. Model (`trashscribe/models/`)

| Module | File | Notes |
|---|---|---|
| `TrashScribe`, `ModelConfig`, `pool_objects`, `TrashScribeEdge` | `trashscribe.py` | config flags: `adapter_dim`, `adapter_where`, `lora_rank`, `mixstyle_p`, `geometry`, `adversary`, `closed_vocab_probe` (E5 reference), `clip_field` (E8) |
| `Sam2Backbone` (LENTE recipe), `TinyBackbone` (tests) | `backbone.py` | SAM2.1-Hiera-Tiny from Hugging Face `facebook/sam2.1-hiera-tiny`, frozen; patch embedding frozen; adapters after attention in 12 blocks; decoder fed the not-a-point embedding |
| `Adapter`, `LoRALinear`, `MixStyle` | `peft.py` | all start as identity |
| `DescriptorField`, `ClipField`, `ContourHead`, `TextReprojection`, `FieldScorer`, `GeometryGuidance`, `EnvironmentAdversary` | `heads.py` | only `tau_f`, `b_f` are field-specific (18 scalars); nothing indexed by value or class |

Parameters (E4): 33.78 M total, 6.56 M trained (0.61 M adapters, 4.22 M decoder, 1.40 M descriptor field,
0.07 M contour head, 0.13 M text re-projection, 18 scalars); adversary 0.13 M is discarded after training.
E8's `ClipField` adds about 0.72 M.

`TrashScribeEdge` is the export graph: image in; `waste_prob` (B,1,S/4,S/4), `contour_prob` (B,1,S/4,S/4),
`desc` (B,512,S/8,S/8) out. **It does not output the E8 `clip` map yet** (see NEXT_STEPS).

## 4. Training (`trashscribe/engine.py`, `trashscribe/losses.py`, `trashscribe/data/`)

* `load_config`: YAML with `_base_` inheritance and `a.b.c=value` overrides; `datasets_file`
  (`configs/datasets.yaml`) is the dataset registry; `extra_datasets` adds datasets for one config only
  (E3b negatives, E9 pseudo-labelled sets).
* `Trainer`: builds adapters per split (`registry.build_split`; held-out classes drop whole images from
  train and val), `MultiDataset` (transplant, temperature sampling `N^0.5`, optional `dataset_share`),
  AdamW with three LR groups, warm-up + cosine, bf16 autocast, gradient accumulation, early stopping on
  val IoU + desc mAP (patience 8), `last.pt` every epoch (resume), `best.pt` at the best val score.
  Opt-in keys added during the grid: `train.loss.unsupervised_values` (E7), `train.loss.clip` +
  `train.clip_teacher` (E8), `train.init_from` (fine-tune from a checkpoint, E9/E10), `train.subsample`
  (seeded subset of a dataset, E10), `train.dataset_share` (fixed sampling mass, E10), `train.select: last`
  (test the last epoch, E10).
* Losses: mask (BCE pos-weight 2 + Dice), contour (weighted BCE, datasets with real instances), descriptor
  (soft-target CE for single fields, BCE for multi fields, unknown fields masked), dense (same loss on 16
  sampled pixels per object), adversary (CE through gradient reversal, weight 0.1), clip (1 − cos to a
  frozen CLIP ViT-B/16 crop embedding, `trashscribe/clip_teacher.py`, E8 only).
* Evaluation (`engine.evaluate`, `metrics.py`): waste IoU/F1/precision/recall/boundary F1 per dataset,
  descriptor mAP vs prevalence floor, GZSL naming (seen/unseen/H, validation-calibrated gamma, AUSUC with all
  trained classes penalised), retrieval AP/lift/bootstrap/permutation for unseen classes, environment
  leakage probe.

## 5. Datasets (`configs/datasets.yaml`, data under `~/data/waste`)

| Dataset | Environment | Split used | Notes |
|---|---|---|---|
| TACO v1 | street/outdoor mixed | 70/10/20 by image, seed 42 | 60 classes, scene labels → surroundings |
| ZeroWaste-f v1.2.1 | recycling conveyor | official | 4 classes; negatives = conveyor belt only (`scripts/make_belt_masks.py`), other unlabelled pixels ignored |
| UAVVaste | aerial | official 540/116/116 | 1 class |
| MJU-Waste v1.0 | indoor, hand-held | official 1485/248/742 | binary masks decoded by palette colour; instances = connected components |
| BePLi v2 | beach | official 2234/744/744 | 13 classes, fishing net text-only (eval-only) |
| coco_neg (E3b only) | indoor scenes with people | 194/49 | 243 COCO val2017 images, all background (`scripts/make_negatives.py`) |
| `<site>_sam3` (E9 only) | the target site | its train/val images | SAM 3 pseudo-labels, `~/data/waste/_pseudo/sam3/<site>/` |

All images were shrunk to longest side ≤ 1024 px (`scripts/prepare_coco.py`). Monocular depth maps
(`_depth/`) exist for the (removed) geometry ablation.

## 6. Edge runtime (`trashscribe/edge_runtime.py`, `trashscribe/monitoring.py`, `trashscribe/naming.py`)

* Backends: `TorchBackend` (a run folder), `OrtBackend` (ONNX Runtime), `TrtBackend` (TensorRT engine).
* `EdgePipeline`: preprocess (letterbox) → backend → `extract_instances` → `pool_numpy` → scoring against the
  precomputed phrase matrix in `AtlasCache` → `Namer` → `frame_report`.
* `AtlasCache` (`atlas_cache.pt`): phrase matrix, value names, field indices, atlas profiles, prototypes and
  prose embeddings, built by `naming.build_cache` during evaluation and copied by `export_onnx.py`.

## 7. Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `train.py <config> [overrides] [--resume]` | train + evaluate one run into `runs/<run_name>/` |
| `evaluate.py` | re-evaluate a run |
| `run_queue.py` | unattended GPU queue from `configs/gpu_queue.yaml` (see OPERATIONS) |
| `progress.py [--watch 30]` | live view, ETAs, results of finished runs |
| `report_results.py` | regenerates the AUTO block of RESULTS.md (every 15 min by a timer) |
| `make_tables.py` | writes `paper/tables/*.tex` from `runs/` |
| `make_figures.py`, `make_comparison.py` | qualitative figure; literature comparison figure (CPU) |
| `posthoc.py dump|naming|clipret|leakage|values|tiles` | analyses on saved checkpoints |
| `baseline_segformer.py` | SegFormer-B2 under the same protocol |
| `baseline_zeroshot.py --detector owlv2|gdino|sam3 --tasks where|describe|pseudo` | zero-shot baselines; `pseudo` writes SAM 3 labels for E9 |
| `make_negatives.py` | COCO indoor negatives for E3b |
| `export_onnx.py`, `build_trt.sh`, `bench_edge.py`, `infer.py` | edge export, TensorRT build (on the Jetson), benchmark, offline monitoring |
| `prepare_coco.py`, `make_belt_masks.py`, `precompute_depth.py`, `inspect_dataset.py` | data preparation |
| `run_grid.py`, `list_runs.py`, `make_grid.py` | older grid helpers (the queue replaced `run_grid.py`) |

Tests: `python -m pytest tests -q` (14 tests, design properties: identity at init, no parameter grows
with the vocabulary, held-out leakage guard, transplant consistency, E8 head, extra datasets, few-shot
subset and sampling share; the SAM2 test runs only if the weights are cached).

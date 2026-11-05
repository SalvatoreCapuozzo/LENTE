# TrashScribe — describing waste instead of naming it

*Trash* is the domain; a *scribe* is what the model is: it segments waste with the LENTE recipe
(frozen SAM2 + zero-init adapters), **writes down** what each object looks like in a controlled
text vocabulary, and names it only by **reading** a text atlas. The successor of LENTE for
smart waste monitoring.

TrashScribe segments waste in any environment and **describes** each object (material, rigidity,
optics, shape, surface, ...) instead of predicting a fixed class. Classes, municipal sorting
streams and brand-new waste types enter **at inference, as text** (an *atlas* file), with no
retraining and no per-class parameter. Everything runs offline on an NVIDIA Jetson Orin Nano 8 GB.

Target venue: *Neural Computing and Applications*, Topical Collection "Emerging Trends in Smart
Waste Monitoring and Environmental Surveillance" — **deadline 30 November 2026**. See [paper/](paper/).

---

## The four requested ideas, and where each one lives

| Idea | Component | Code |
|---|---|---|
| LENTE architecture in the waste domain | SAM2.1-Hiera-Tiny frozen + zero-init adapters / LoRA + prompt-free mask decoder; LENTE's DSDF terrain gate becomes a **depth-gated geometric guidance** (zero-init, identity at step 0) | [trashscribe/models/backbone.py](trashscribe/models/backbone.py), [trashscribe/models/heads.py](trashscribe/models/heads.py) `GeometryGuidance` |
| Dataset- and class-independent model working on visual descriptors (from *Describing, Not Naming*) | Dense **descriptor field** in a frozen CLIP text space; scorer with **no per-descriptor / per-class parameter** (only τ, b per field); atlas-based naming (descriptor agreement + text prototype + prose) with abstention; unseen phrases and classes scored through the same path | [trashscribe/models/heads.py](trashscribe/models/heads.py), [trashscribe/naming.py](trashscribe/naming.py), [atlas/](atlas/) |
| Dataset-independent, not biased by the environment | (1) every dataset reduced to the common "waste" mask + whatever descriptors its taxonomy implies; (2) **intrinsic vs contextual** pooling (inside the mask vs a ring around it; context never used to name); (3) **cross-dataset object transplant**; (4) **environment adversary** with gradient reversal; (5) MixStyle; (6) dataset-balanced sampling; (7) leave-one-dataset-out protocol and an environment-leakage probe | [trashscribe/data/mixing.py](trashscribe/data/mixing.py), [trashscribe/models/trashscribe.py](trashscribe/models/trashscribe.py), [trashscribe/metrics.py](trashscribe/metrics.py) |
| Runs on Jetson Orin Nano 8 GB, no internet | One ONNX graph (image → waste map, contour map, descriptor field), TensorRT FP16; instances by contour-seeded assignment (no per-object decoder call); text encoder **not** needed on device (phrases precomputed in `atlas_cache.pt`) | [trashscribe/edge_runtime.py](trashscribe/edge_runtime.py), [scripts/export_onnx.py](scripts/export_onnx.py), [scripts/build_trt.sh](scripts/build_trt.sh) |

```
 RGB ─(optional depth gate)─► SAM2-Hiera-Tiny (frozen) + adapters/LoRA + MixStyle
                                   │  S/4 (32ch)   S/8 (64ch)   S/16 (256ch)
          ┌────────────────────────┼────────────────────────────┐
          ▼                        ▼                            ▼
  waste decoder (WHERE)    contour head (SPLIT)       descriptor field (WHAT), stride 8,
  prompt-free, 1 class     touching objects           vectors in the frozen CLIP text space
          │                        │                            │
          └──── instances ─────────┘          mask pooling: inside = intrinsic, ring = context
                                                                 │
                     schema phrases ─► W_t (zero-init residual) ─► s = τ_f·cos + b_f
                                                                 │  descriptor posterior
                       atlas.yaml (text) ─► descriptor agreement + prototype + prose ─► name | set aside
                                                                 │
                       coverage · count · composition by material / sorting stream · written report
```

## What is new with respect to the state of the art (Nov 2026)

* Waste segmentation work fine-tunes per dataset and reports in-domain numbers (e.g. SAM fine-tuning
  on ZeroWaste/TACO/TrashCan, COSNet, the ICML 2026 cluttered-background network); cross-dataset
  generalisation is not evaluated. TrashScribe trains **one** model on heterogeneous datasets and is
  evaluated **leave-one-dataset-out**.
* Open-vocabulary detectors applied to waste (BMVC 2025) need LLM-engineered prompts and still trail
  supervised baselines; SAM 3 / EfficientSAM3 are concept-level, not attribute-level. TrashScribe grounds
  **attributes** (material, optics, rigidity...) densely, so classes are compositions of descriptors
  and new ones are written, not annotated.
* Waste datasets correlate object and environment (conveyor → cardboard, beach → fragments). TrashScribe
  separates intrinsic from contextual evidence by construction and **measures** the residual
  environment information in object embeddings.
* Edge: NanoSAM/EdgeSAM run SAM on Orin, NanoOWL runs OWL-ViT; none describes objects or names
  them from text offline. TrashScribe's whole graph is one engine plus a few matrix products.

Full notes with sources: [docs/SOTA.md](docs/SOTA.md).

## Folder map

```
TrashScribe/
├── atlas/            schema.yaml (descriptor vocabulary), taco.yaml (60 classes), zerowaste.yaml,
│                     binary.yaml, novel_field_classes.yaml (classes in NO dataset, inference only)
├── configs/          base.yaml, datasets.example.yaml, synthetic_*.yaml, exp/ (23 experiment configs)
├── trashscribe/           package: models/, data/, losses, metrics, naming, monitoring, engine, edge_runtime
├── scripts/          train, evaluate, run_grid, list_runs, export_onnx, build_trt.sh, bench_edge, infer,
│                     baseline_segformer, baseline_zeroshot, precompute_depth, inspect_dataset, make_grid
├── tests/            unit tests of the design properties (pytest)
├── runs/             one folder per run (config, history, metrics, checkpoints [git-ignored])
├── attempts/         ATTEMPTS.md — journal of every attempt, hypothesis and outcome
├── docs/             SOTA.md
└── paper/            NCAA manuscript (Springer Nature sn-jnl template), title page, cover letter, checklist
```

## Setup

**Mac / CPU (development, done):** an isolated venv on top of `dl_env`, so `dl_env` keeps its
transformers 4.41 for other projects while TrashScribe gets the 5.2 needed by `Sam2Model`:
```bash
cd TrashScribe
~/anaconda3/envs/dl_env/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install "transformers==5.2.0" "huggingface-hub==1.4.1" pytest
.venv/bin/python -m pytest tests -q
```

**NVIDIA workstation (RTX 3090 / any CUDA GPU):**
```bash
conda create -n trashscribe python=3.10 -y && conda activate trashscribe
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -r requirements.txt
python -m pytest tests -q
```

**Jetson Orin Nano 8 GB (JetPack 6.x):** see [requirements-jetson.txt](requirements-jetson.txt) — PyTorch
from NVIDIA's Jetson wheels, TensorRT from JetPack, then `bash scripts/build_trt.sh`.

## Quickstart

```bash
# 0. sanity: procedural scenes, tiny CNN, 5 min on a laptop (NOT a result)
python scripts/train.py configs/synthetic_sanity.yaml
# 1. data: copy configs/datasets.example.yaml -> configs/datasets.yaml, fix paths, check classes
python scripts/inspect_dataset.py ~/data/waste/TACO/data/annotations.json --atlas atlas/taco.yaml
# 2. the grid (GPU), then the table that feeds the paper
python scripts/run_grid.py configs/exp/e4_heldout_classes.yaml configs/exp/e3_lodo_*.yaml configs/exp/e5_*.yaml
python scripts/list_runs.py --md
# 3. baselines on the same splits
python scripts/baseline_segformer.py configs/exp/e3_lodo_taco.yaml --model nvidia/mit-b2
python scripts/baseline_zeroshot.py  configs/exp/e4_heldout_classes.yaml
# 4. edge
python scripts/export_onnx.py runs/e6_edge_640            # workstation
bash scripts/build_trt.sh runs/e6_edge_640/export/trashscribe_640.onnx fp16      # ON the Jetson
python3 scripts/bench_edge.py --engine .../trashscribe_640_fp16.engine --onnx .../trashscribe_640.onnx \
        --cache .../atlas_cache.pt --check
python3 scripts/infer.py --engine .../trashscribe_640_fp16.engine --cache .../atlas_cache.pt --source 0
```

## Status (5 Nov 2026)

Done and tested on the Mac (M2 Pro, MPS):
* full package; 11 unit tests green (identity at init, no parameter grows with the vocabulary,
  held-out leakage guard, transplant consistency, SAM2 frozen core, LoRA folding);
* end-to-end training + evaluation on the synthetic sanity benchmark with the tiny CNN and with
  SAM2.1-Tiny + adapters (see [attempts/ATTEMPTS.md](attempts/ATTEMPTS.md));
* ONNX export of the complete 640-px graph, max |ONNX − PyTorch| ≈ 8e-6;
* offline pipeline (instances → descriptors → atlas name → report) and benchmark harness;
* zero-shot baselines (OWLv2→SAM2, raw CLIP) and SegFormer baseline wired to the same splits;
* NCAA manuscript (anonymised, 15 pages, compiles with the official Springer Nature class):
  introduction, related work, method and protocol written from the code; results tables are
  generated from `runs/` and show "--" until the GPU runs exist. Title page and cover letter
  compile too. Two guest editors share the authors' university: declare it (see paper/README.md).

Not done (needs NVIDIA hardware and the datasets): every real-data number, TensorRT engines,
Jetson latency/power.

## When to move to NVIDIA hardware

**Now, for training.** Everything that can be validated without real data and without CUDA is
validated. This Mac cannot go further usefully: no CUDA (bf16 autocast, TensorRT and real
throughput are CUDA-only), ~8 GB of free disk (TACO + ZeroWaste-f + UAVVaste + MJU + BePLi
are tens of GB), and MPS runs this model in fp32 only, far below what an RTX 3090 delivers.

1. **Workstation with an NVIDIA GPU (the RTX 3090 used for LENTE) — start immediately:**
   download datasets, `inspect_dataset.py` each one, then run E4 (held-out classes) first, E3 (LODO)
   second, E5 ablations third, E1/E2 references and baselines last. Rough budget, not measured:
   2–4 h per run at 1024 px with `epoch_size: 4000`, 40 epochs, batch 8 — time the first run and
   shrink `epoch_size`/`epochs` if the grid does not fit the calendar.
2. **Jetson Orin Nano 8 GB — once one trained model exists (e.g. `e6_edge_640`):** export ONNX on the
   workstation, copy `.onnx` + `atlas_cache.pt`, build the engine **on the Jetson** (engines are not
   portable), benchmark in MAXN mode with `--check`, record power with `tegrastats`.

Suggested calendar to the 30 Nov deadline: 6–16 Nov runs (E4, E3, E5), 14–18 Nov Jetson,
17–27 Nov writing with numbers from `list_runs.py`, 28–30 Nov anonymisation and submission.

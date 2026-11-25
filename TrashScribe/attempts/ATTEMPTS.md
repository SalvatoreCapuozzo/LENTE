# TrashScribe — attempts journal

One entry per attempt: what was tried, why, the number that decided, what changes next.
Numbers come from `runs/<run>/test_metrics.json` (`python scripts/list_runs.py`), never from memory.
Synthetic runs are pipeline checks, not results.

---

## A0 — 2026-11-05 — environment (Mac M2 Pro, MPS)
* `TrashScribe/.venv` with `--system-site-packages` on `dl_env` (torch 2.12.1) + transformers 5.2.0
  (`Sam2Model` needs ≥ 4.56; same version as the Windows workstation), huggingface-hub 1.4.1,
  scikit-learn 1.7.2 (the one in dl_env was broken against numpy 2; fixed in A8), pytest.
* Weights cached: `facebook/sam2.1-hiera-tiny`, `openai/clip-vit-base-patch16`.
* SAM2 feature levels confirmed: S/4 → 32 ch, S/8 → 64 ch, S/16 → 256 ch; encoder 27.2 M,
  decoder 4.2 M. Patch-embedding output is channels-last (B, H/4, W/4, 96).

## A1 — unit tests
11/11 pass: zero-init identity (adapters, LoRA, text re-projection, geometric gate), LoRA folding
for export, no parameter changes shape when the vocabulary grows, ring pooling, held-out guard,
transplant consistency, SAM2 frozen core.

## A2 — synthetic sanity, tiny CNN (`runs/synthetic_sanity_tiny`)
12 epochs × 600 samples, 256 px, MPS, ~22 s/epoch. Held out: glass bottle, metal foil, paper bag;
environment bias 0.85 in train/val, uniform in test.
* waste IoU 0.846 (val 0.898), BF1 0.886; descriptor mAP 0.790 vs floor 0.283.
* held-out retrieval from the atlas entry alone: lift 4.7–7.1×, permutation p = 0.005 for all three
  (the minimum attainable with 200 permutations).
* naming: seen 0.688, unseen **0.000** at the default point; AUSUC 0.433; along the curve unseen
  reaches 0.66 (oracle, test-chosen γ). Per-stream: dap 0.00, proto 0.00, prose 0.01 unseen.
* environment probe on intrinsic embeddings 0.98 vs majority 0.24 → background still decodable.
* **Bugs found and fixed:** (i) the synthetic *validation* split contained held-out classes; the
  registry's leakage assertion caught it — held-out classes now exist only in test; (ii) MPS
  corrupts `non_blocking` host→device copies of small tensors (index out of bounds) — non-blocking
  only on CUDA now.

## A3 — synthetic sanity, SAM2.1-Tiny + adapters (`runs/synthetic_sanity_sam2`)
3 epochs × 256 samples, 256 px, MPS, ~25 s/epoch; 33.8 M params, 6.56 M trainable (decoder 4.2 M).
* val IoU 0.937 → 0.959 in 3 epochs; test IoU 0.950: the val→test drop under environment shift is
  much smaller than the tiny CNN's (0.959→0.950 vs 0.898→0.846) — consistent with the frozen-VFM
  DG hypothesis, to be verified on real data.
* descriptor mAP 0.746 (floor 0.283). Naming unseen 0.000 at default point, AUSUC 0.237.
* environment probe 0.93 vs 0.24.

## A4 — GZSL bias toward seen classes
Both runs name held-out compositions at 0% at the default operating point, while ranking them
correctly in retrieval (lift > 4) and reaching 0.66 along the calibration curve. Raw CLIP on crops
gets 0.29 unseen (synthetic) — training sharpens seen classes.
* Added calibrated stacking with γ chosen on **validation** (largest γ keeping ≥ 90 % of val
  seen-class accuracy; validation never contains held-out classes): tiny CNN γ = 0.7 → unseen
  0.08 (= chance 0.083), seen 0.65. Too conservative.
* Report three numbers on real data: default point, val-calibrated point, AUSUC (calibration-free).
* Next options, in order: (1) leave-class-out calibration on validation (fit γ with a fold of seen
  classes removed from the *atlas*, as MIRF fitted fusion weights leave-one-class-out);
  (2) `calib_keep` 0.8; (3) heavier weight on the descriptor-agreement stream, which is less
  biased by construction. Decide on E4 real data, not on synthetic.

## A5 — edge path on the Mac
* ONNX export of the full 640-px graph (opset 17, TorchScript exporter): 126.7 MB fp32,
  max |ONNX − PyTorch| ≤ 8e-6 on all three outputs. ORT warns "Error merging shape info" on one
  Concat in Hiera → `build_trt.sh` runs onnxsim before trtexec.
* Offline pipeline on synthetic frames at 256 px, MPS: ~24 ms/frame (net 19 ms, post 4 ms).
  Pipeline check only — Jetson numbers must be measured on the Jetson.
* Report lines are composed only from scored descriptor values; single-valued fields are stated
  only above 0.5 (otherwise unsupervised fields printed an arbitrary argmax).

## A6 — baselines wired (synthetic check only)
* OWLv2 + SAM2 zero-shot "where": IoU 0.26 (precision 0.99, recall 0.26) on 30 synthetic images.
* raw CLIP crops "describe": descriptor mAP 0.435 vs floor 0.283; prose naming unseen 0.29.
* SegFormer baseline script ready (not run here: needs the mit-b2 download and real data).

## A7 — real data not started on the Mac
TACO image host refuses Python's default client (HTTP 403). Use TACO's official `download.py` on
the GPU machine. Free disk on the Mac ≈ 8 GB: datasets go on the workstation.

## A8 — 2026-11-05 — rename and dl_env repair
* Project renamed VAGLIO -> **TrashScribe** (folder `TrashScribe/`, package `trashscribe`, classes
  `TrashScribe` / `TrashScribeEdge`, ONNX `trashscribe_<S>.onnx`). Venv recreated; tests 11/11;
  `runs/synthetic_sanity_tiny` re-evaluated with identical numbers (IoU 0.846, desc mAP 0.790).
* `dl_env` repair (backups: `~/dl_env_backup_pipfreeze_2026-11-05.txt`,
  `~/dl_env_backup_conda_explicit_2026-11-05.txt`). Cause: numpy 2.2.6 and pyarrow 25 had been
  pip-installed over conda packages built for numpy 1.
  - scikit-learn 1.3.0 -> 1.7.2 (failed at import; also broke `transformers.pipelines`)
  - statsmodels 0.14.0 -> 0.14.5 (imported, crashed on use: numpy dtype size mismatch)
  - left as is: `datasets` 2.12.0 (incompatible with pyarrow 25, nothing depends on it; 2.21 would
    downgrade fsspec under s3fs/dask, 3.x/4.x would pull huggingface_hub 2.x and break transformers 4.41)
  - absent, listed in LENTE's requirements-windows.txt: h5py, rasterio, segmentation-models-pytorch
* SAM2 sanity run re-evaluated with the validation-calibrated naming (gamma = 1.0).

## A9 — 2026-11-05 — NVIDIA machine (RTX 3080 Laptop 8 GB), real data, grid started
Results and their running commentary live in [../RESULTS.md](../RESULTS.md); this entry only
records what changed in the code and why.
* Env: `.venv` on conda `dl_env` (torch 2.9.1+cu130), transformers 5.2.0, hub 1.4.1, numpy 1.26.4.
* Data: all five datasets downloaded, shrunk to <= 1024 px with the new `scripts/prepare_coco.py`
  (annotations rescaled), registry `configs/datasets.yaml` (official splits for ZeroWaste-f,
  UAVVaste, MJU-Waste, BePLi v2); `configs/base.yaml` now points at it.
* Bugs fixed: MJU-Waste masks inverted (palette index 0 = waste; `waste_colors` in `VocAdapter`);
  TACO duplicate annotation ids (`CocoAdapter` uses `imgToAnns`); `np.trapezoid` on numpy 1;
  leakage probe disabled by one rare environment; RAM exhaustion from persistent val workers.
* Added: gradient accumulation (`train.accum_steps`), resumable checkpoints (`train.py --resume`),
  `scripts/run_queue.py` + `configs/gpu_queue.yaml` (unattended grid), `scripts/report_results.py`
  (tables of RESULTS.md), depth registry `configs/datasets_depth.yaml` for the geometry ablation,
  SDPA attention + accumulation in `baseline_segformer.py`.
* Pilot `runs/pilot_e4` (2 x 400 samples, before the MJU/TACO fixes): pipeline check only.
* Host RAM (first E4 launch killed by systemd-oomd with VS Code): per-object masks packed into
  uint16 label layers in `Canonicalizer` (exact; BePLi has up to 363 objects/image), COCO
  annotations stored as packed bytes (no copy-on-write growth in workers), 6/3 workers, val batch 4,
  `train.val_prefetch`; queue runs as `systemd-run --user` service with MemoryMax=8G (RESULTS.md).

## A10 — 2026-11-05 — label decisions, preliminary check, grid restarted
* Decisions (user): paper is waste -> ZeroWaste-f negatives = conveyor belt only
  (`scripts/make_belt_masks.py`, `negatives_dir`), other unannotated pixels ignored by loss and
  metrics; boundary F1 scored on edges inside the supervised region only. BePLi described: TACO
  aliases + `atlas/bepli.yaml`. Fishing net (novel atlas) evaluated on BePLi as an eval-only entry
  (`eval_only_entries`; mask-only in training, never seen). 6 held-out classes accepted.
* Depth paths folded into `configs/datasets.yaml` (`geometry_depth_dir`, used only with
  `model.geometry`). Live monitor `scripts/progress.py` (+ `progress.json` per run).
* Preliminary check on the stopped E4 (epoch 7/40, old labels): IoU TACO 0.83 / MJU 0.87, desc mAP
  0.53 vs 0.23, held-out naming 0.000 at default and calibrated points (AUSUC 0.013, oracle H 0.10),
  retrieval lift significant for 4/6 held-out classes, env probe 0.99 vs 0.75. Decision: grid
  restarted from scratch; naming calibration to be revisited post hoc on E4 (A4 options).

## A11 — 2026-11-05 — conda env replaces the venv
* `environment.yml` -> conda env `trashscribe` (Python 3.12.3 conda-forge + pip pins identical to the
  venv: torch 2.9.1+cu130, transformers 5.2.0, numpy 1.26.4, opencv-python-headless 4.10.0.84, ...).
  Data-pipeline hash (decode, masks, letterbox, augmentation, CPU head forward) identical to the
  venv; cuDNN 9.13 in both. Queue and report timer switched at the E4 epoch-7 boundary (resumed);
  `.venv` removed.

## A12 — 2026-11-13 — first results, manuscript and figures
Results and commentary: RESULTS.md, journal of 13 Nov. Changes to code and paper only:
* `scripts/make_figures.py` (new): qualitative figure from E4 + the five E3 checkpoints, CPU only
  (run as a capped `systemd-run` unit next to the GPU queue, ~1.5 min, < 3 GB). Images chosen by a
  seeded rule recorded in `paper/figures/qualitative.json`, not by result. It writes the object
  table `paper/figures/qualitative_objects.tex` too.
* `paper/figures/pipeline.tex` (new): end-to-end data-flow figure (TikZ). In `sn-jnl`
  `\footnotesize` is 7 pt and `\scriptsize` 9 pt (reversed from standard LaTeX).
* `scripts/make_tables.py`: E4 column in `tab_lodo` (seen-environment reference until E1/E2
  exist); p < 0.001 printed as such.
* Finding that changes the plan: E3-MJU IoU 0.055 (people and furniture segmented as waste).
  Next: post hoc naming calibration (A4) and a within-class leakage probe, both on saved
  checkpoints.

## A13 — 2026-11-13 — post-hoc naming, CLIP baseline, within-class leakage; E7 added
Numbers and reading: RESULTS.md journal, 13 Nov 17:00–18:30.
* `scripts/posthoc.py` (new): `dump` (per-object z_int, z_ctx, entry, environment, dataset, optional
  raw-CLIP crop embedding; GPU with `set_per_process_memory_fraction` so it cannot starve the
  training), `naming` (calibration variants, vectorised per-class accuracy), `clipret` (unseen-class
  retrieval vs raw CLIP), `leakage` (within-class environment probe), `values` (per-value AP, for E7).
* A4 options tested: lower keep ratio helps (H 0.049 → 0.155), descriptor agreement alone is the
  least biased stream (H 0.147), entry-wise standardisation does not work. Raw CLIP crops remain
  better at naming (H 0.217) and retrieval (mean AP 0.078 vs 0.034) of unseen classes.
* `metrics.ausuc` penalised only seen classes present in test → underestimated; fixed (`penalized=`).
* `train.loss.unsupervised_values` (opt-in): listed descriptor values leave the loss entirely and
  model selection (`descriptor_map(skip_values=)`). E7 holds out `material: metal`, `material:
  expanded polystyrene foam`, `surface: crumpled or wrinkled` (multi-valued fields only, so nothing
  else is perturbed).
Decision: the purpose of the paper is not rewritten until the closed probe and E7 are in.

## A14 — 2026-11-15 — E7 verdict; SegFormer OOM; E8, E3b, test-time tiling, comparison figure
Numbers and reading: RESULTS.md journal, 15 Nov 17:00.
Hypothesis tested (E7): the text-space descriptor head reads descriptor values never supervised.
Result: 1 of 3 (metal AP 0.140 vs raw CLIP 0.033); foam at the floor, crumpled below it (raw CLIP
0.114 / 0.276). The closed probe matches the open head on supervised values (0.712 vs 0.691).
Decision: open-vocabulary description is NOT claimed. Refocus candidates: the cross-environment
benchmark (five datasets, LODO, atlas profiles) with TrashScribe as reference model, plus E8 if
the distilled field keeps CLIP's open vocabulary inside the dense model.
Changes:
* `scripts/baseline_segformer.py`: `empty_cache` before eval, resume from `last.pt`, patience
  early stop; queue: SegFormer val batch 1, `max_attempts: 6`. Cause: deterministic CUDA OOM at the
  first validation (batch 4 × 1024 px, decode-head concat 1.5 GiB/tensor).
* `models/heads.py` `ClipField`, `ModelConfig.clip_field`, `trashscribe/clip_teacher.py`
  (`ClipTeacher`, `distill_loss`), `Trainer` builds the teacher when `loss.clip > 0`; off by default
  (all earlier configs unchanged). `configs/exp/e8_clip_distill.yaml` (on E7).
* `engine.load_config`: `extra_datasets` (datasets of one config only). `scripts/make_negatives.py`,
  `configs/exp/e3b_lodo_mju_negatives.yaml` (243 COCO val2017 indoor scenes with people, no
  waste-like category; data in `~/data/waste/coco_neg`, archives in `_dl/coco`).
* `scripts/posthoc.py`: `tiles` (zoom-in tiling), `dump` stores `zclip`, `clipret --visual zclip`,
  `values` scores `zclip` like raw CLIP crops.
* `scripts/make_comparison.py`: literature comparison figure + description table.
* `scripts/baseline_zeroshot.py`: `--detector gdino` (box area filter) and `--detector sam3` (noun
  phrases → instance masks), `--device`, `--threads`; both queued as `kind: zeroshot` GPU jobs.
  Reason: on the figure images SAM 3 beats TrashScribe-LODO on 4 of 5.
* `posthoc.py tiles`: zoom helps aerial only for models trained on ground-level views (+4.0 E4,
  +3.3 LODO, −7.8 aerial-only) → scale mismatch; remedy (scale augmentation) proposed, not run.
* `run_queue.py`: gives up after two CUDA OOMs in a row (takes effect at the next queue restart).

## A15 — 2026-11-15 — E9: SAM 3 as an offline teacher at a new site; queue by significance
Hypothesis: the 34 M edge model, fine-tuned with a new site's unlabelled images labelled once by SAM 3
(848 M, never on the device), reaches or passes SAM 3 zero-shot there (literature: SAM 3-labelled
YOLO students within 1.3 mAP of SAM 3, arXiv/alphaXiv 2605.25860; semi-supervised foundation-model
students beating adapted teachers, 2604.03841).
Changes: `baseline_zeroshot.py` (`--tasks pseudo`, `--sam3-prompt-set waste4|objects16`, cached text
embeddings, output dir `__zeroshot_sam3obj` for 16 prompts); `engine.Trainer.train` `train.init_from`
(fine-tune from a checkpoint; ignored on resume); configs `e9_sam3_{mju,uavvaste,bepli}`; progress /
report / paper tables gain the E9 and SAM 3 (16 prompts) columns.
Finding on the way: SAM 3's presence head rejects "litter"/"trash" in non-litter scenes (MJU office),
so the 4-word baseline under-reports SAM 3; the 16-prompt set is the fair reference.
Decision: queue ordered by expected significance (RESULTS.md 15 Nov 19:00).

## A16 — 2026-11-16/17 — fair SAM 3, SegFormer-MJU, E8 verdict, E9 extended to TACO / ZeroWaste-f
* SAM 3 evaluation now writes `where_sam3` (all masks) and `where_sam3_area` (masks <= half image, the rule
  of the Grounding DINO boxes and the E9 labels); `--tag` keeps re-runs apart. Fair SAM 3 BePLi 0.424 (raw
  0.272): E9 reaches 88-105 % of SAM 3 at a new site, it does not beat it. Paper table uses the fair rule.
* SegFormer-B2 LODO MJU 0.104: the MJU failure is the protocol's.
* E8: distilled CLIP field = raw CLIP at retrieval of unseen classes (0.082 vs 0.078) and on unseen values
  (foam better, crumpled close, metal worse), naming still below raw CLIP. Open reading is back in one pass.
* Longer E9 (24 epochs) not run: val plateau from epoch ~6, student already matches its labels; ceiling =
  label quality. Test-time tiling on E9-UAVVaste: +1.6 (mean of full and tiles), tiles alone -7.1.
* `progress.py` / `report_results.py`: NaN shown as n/a with the reason; SAM 3 and pseudo-label jobs shown.
* Queue (17 Nov): E9 for TACO and ZeroWaste-f added after E3b (configs `e9_sam3_{taco,zerowaste}`).

## A17 — 2026-11-17 — E10 (labelled site images), ladder table, Jetson import fix, handover docs
Hypothesis (owner's goal): with a few labelled images of a new site the 34 M edge model matches or beats
SAM 3 (848 M, zero-shot) at every site.
Changes: `train.subsample` (seeded subset, `registry.subsample_adapter`), `train.dataset_share`
(`MultiDataset.sample_weights(share=)`), `train.select: last`; configs `e10_k10_<site>` (10 images, 25 % of
samples, 6 epochs from the LODO checkpoint) and `e10_k50_uavvaste` (rule fixed before results: a site below
SAM 3 with 10 images gets 50); unit test (14/14). Paper: Table `tab:ladder` (`make_tables.ladder`) and its
paragraph; the LODO table keeps the benchmark columns. GSA2Seg cited.
Results so far: E9-TACO 0.783 and E10-BePLi 0.454 above SAM 3 (0.713, 0.424); E10-UAVVaste 0.289 below 0.424
→ 50-image run queued. E3b 0.363 above SAM 3 on MJU.
Fix: `trashscribe/data/canonical.py` imported albumentations at module level and `edge_runtime.py` imports
that module, so the Jetson runtime would have failed (albumentations is not in requirements-jetson.txt); the
import is now inside `build_augment`. Edge export of E6 done (`runs/e6_edge_640/export/`).
Docs: `AGENTS.md`, `CLAUDE.md`, `docs/{CONTEXT,ARCHITECTURE,EXPERIMENTS,OPERATIONS,JETSON,NEXT_STEPS,MIGRATION}.md`.

## A18 — 2026-11-18 — first steps on the Jetson (e6_edge_640)
Hypothesis: the exported E6 graph builds and runs on the owner's Orin Nano as documented in `docs/JETSON.md`.
Finding: the board runs **JetPack 5.1.1** (L4T 35.3.1, TensorRT 8.5.2, CUDA 11.4, Python 3.8), not JetPack 6.
Changes: `.venv` on `dl_env` (torch 2.0.0+nv23.05, CUDA) with the JetPack `tensorrt` / `onnx_graphsurgeon`
symlinked, onnxruntime 1.19.2 CPU; `scripts/onnx_for_trt85.py` (TRT 8.5 has no LayerNormalization importer and
rejects OneHot computing shapes in the HF mask decoder: 34 LN nodes decomposed, graph constant-folded 3,852 →
1,184 nodes, max difference to the original 8.5e-6 under ONNX Runtime), called by `build_trt.sh` on TRT < 8.6;
`build_trt.sh` drops `--builderOptimizationLevel` (TRT ≥ 8.6 only); `bench_edge.py --check` now uses 5 real
frames when `--images` is given and reports the waste-mask IoU against FP32; `scripts/fetch_edge_images.py`
(30 test-split images of TACO, ZeroWaste-f, UAVVaste over HTTP range requests, ≤ 1024 px);
`scripts/jetson_suite.sh` / `jetson_progress.sh` / `jetson_summary.py` / `jetson_env.py` (all measurements of
`docs/JETSON.md`, resumable, `trashscribe-jetson` user unit). Tests on the device: 11 pass, 1 skipped, 2 need
training-only packages (albumentations), as expected.
Result so far (NOT a paper number: clocks unlocked, desktop on the GPU, 0.6 GB free): first FP16 engine
120.0 ms GPU compute (8.3 FPS, trtexec, CUDA graph); 81 tactics of the fused Hiera stage-1/2 attention regions
skipped for memory (they ask 1.4 to 1.7 GB); blocks 0 to 3 take 55 of 122 ms. Kept as `*_fp16_lowmem.*`.
Decision: build again after `sudo jetson_clocks` with ≥ 3 GB free (the suite now waits for both before building).
Not possible on JetPack 5.1: SAM 3 on the device and the PyTorch backend (transformers 5.2 needs Python ≥ 3.10).
MJU-Waste (Google Drive) and BePLi v2 (9.5 GB tar.gz at about 0.6 MB/s) images not fetched.
Second fix: engines built by TRT 8.5 contain the InstanceNormalization plugin; `TrtBackend` now calls
`trt.init_libnvinfer_plugins` (trtexec does it implicitly; without it deserialization fails).
Full suite result (same caveats: clocks unlocked, 0.6 GB free, no power): FP16 119.3 ms trtexec, 5.4 / 5.6 / 4.6
FPS full pipeline on TACO / UAVVaste / ZeroWaste-f, FP16 waste-mask IoU vs ORT FP32 ≥ 0.995 on real frames
(RESULTS.md, 18 Nov). Rebuilding with more free memory did not help at 0.6 GB (88 tactics skipped, 119.3 vs
120.0 ms): the ≥ 3 GB condition is still untested.

## A19 — 2026-11-19 — paper review and revision; ladder figure; ablation table; comparison figure aligned
Review findings fixed: title/abstract on abandoned claims; site adaptation and distilled head missing from
Methods; two qualitative figures on the same images; ablation table with three never-run rows and an AUSUC
column mixing the old and corrected definitions; empty on-device table; empty conclusions; six unverified
bibliography entries printing VERIFY notes. Changes: `make_tables.ablation` (no dead rows, no AUSUC, n/a where a
variant cannot produce a metric); `scripts/make_ladder_figure.py` (palette validated with the dataviz validator:
ordinal blue ramp + orange reference, all checks pass); `make_comparison.py` (SAM 3 with the 16 prompts and
half-image rule of the tables, new `e10` column, dataset badges instead of rotated labels); Jetson numbers in a
hand-written `tab:edge` (the generator expects bench JSON files that are not on this machine); new bib entries
(detect-waste, Mewada et al. 2025, TrashDet, Garberus, X3D-IWDD, SAM 3-assisted YOLO, semi-supervised
distillation), IWDD entry corrected (full author list incl. B. Vento, pages 539–548).
Decision: keep the three remaining queue jobs; add none.

## A20 — 2026-11-20 — grid finished; final numbers; rover use case
SegFormer-B2 LODO TACO / ZeroWaste-f and E1-BePLi completed the tables. Reading: a fully fine-tuned SegFormer-B2
transfers as well as TrashScribe or better (street +11.6, aerial +21.8 points), so the paper no longer implies a
transfer advantage of the frozen encoder. Introduction: the system's deployment on a remotely guided wheeled rover
(beaches, natural habitats); future work: autonomous driving. Footnote shortened (page 2 overflowed by 7 pt).

## A21 — 2026-11-24 — Jetson on JetPack 6.2: locked clocks, power, memory, SAM 3 on the device
Board reflashed to JetPack 6.2 (L4T 36.5.2, TensorRT 10.3, CUDA 12.6). New venv (torch 2.8.0 from jetson-ai-lab;
2.9+ needs libcudss). Fixes: GPU devfreq node renamed on JetPack 6 (`17000000.gpu`; the old test reported unlocked
clocks as locked), user-level tegrastats (rails readable without root), engines in `export/jp6/`, constant folding
also needed on TensorRT 10.3 (OneHot shape computation).
Hypothesis/check: FP16 = FP32 = ONNX Runtime as on JetPack 5. **Failed first**: both TensorRT 10.3 engines gave a
different waste mask (IoU 0.81-0.86 with ONNX Runtime; objects/frame 5.5 vs 4.3 of PyTorch and JetPack 5). Not LayerNorm
(the decomposed graph fails identically), not a token or spatial mix-up; Polygraphy bisection over the 234 waste-path
tensors: marking `output_hypernetworks_mlps.0/proj_in/MatMul_output_0` alone fixes it, i.e. a fusion bug. Workaround in
`onnx_for_trt85.py --keep-layernorm` (hypernetwork proj_in outputs marked); same latency; FP32 vs ORT 2e-5.
Faulty results kept in `runs/e6_edge_640/jetson_jp6/_superseded_2026-11-24/`.
SAM 3 (`bench_sam3_edge.py`, same prompts/path as its accuracy numbers): library post-processing ran out of memory on
a cluttered frame (int64 masks at frame size); same masks computed chunked and boolean (verified identical).
Result: TrashScribe FP16 8.0-10.3 FPS end to end, 13.8 W, 0.5 GB; SAM 3 0.19 FPS, 3.2 GB, 66 J/frame.
Decision: paper Table `tab:edge`, Sect. "On-device performance", implementation details, abstract, contribution 4,
use cases, limitations, future work and conclusions updated; the JetPack 5.1.1 figure is cited once for comparison.

## A22 — 2026-11-24 — SAM 3 in TensorRT on the Jetson (sam3_trt)
Hypothesis: SAM 3 compiled with TensorRT FP16 is much faster than in PyTorch bf16 (5.2 s/frame) but stays orders
of magnitude slower than TrashScribe; the comparison in `tab:edge` should include it (the owner asked for it).
Change: `scripts/export_sam3_onnx.py` exports two graphs from the FP32 checkpoint: the image encoder with the three
FPN levels the detector uses, and the detector/mask decoder for one text prompt; the 16 text embeddings are
computed once in FP32 and stored (no text encoder on the device). `trtexec --fp16`. Same pre-processing, prompts,
score threshold and chunked post-processing as the PyTorch run (`bench_sam3_edge.py --backend trt`).
**Rule fixed before any TensorRT output was seen.** Frames: the 90 edge frames (30 per dataset). Metric: per frame,
IoU of the union mask (all 16 prompts, score >= 0.5, mask > 0.5 at frame size; empty vs empty = 1) and instance count.
Reference: PyTorch bf16 through `Sam3Model` (the configuration of SAM 3's accuracy numbers and of its row in
`tab:edge`). Noise floor: the same metric for PyTorch FP32 (the exported parts, run in PyTorch) against bf16.
The engine counts as SAM 3 in TensorRT if its mean IoU against bf16 is at least the FP32 one minus 0.02; otherwise
it is reported as failing and not put in the table as SAM 3.
Building: one engine of the whole encoder (946 MB graph) was OOM-killed twice (parsing alone 2.1 GB next to the
desktop); exported instead as four chunks of 8 ViT layers (each ends with a global-attention layer), run back to
back with one shared activation buffer (TensorRT USER_MANAGED memory). Parity first: the exported parts in PyTorch
bf16 are bit-identical to `Sam3Model` (only after computing the 72 px position encoding in the model dtype, as
the library does). Every build skipped one tactic for lack of memory (1.76 GB, FP32 attention scores).
**First variant: FAIL** (24 Nov): FP32 noise floor 0.9923 (min 0.901); TensorRT FP16 0.649 (66 of 90 frames
< 0.9), instances/frame 23.6 vs 41.1; 3,077 ms/frame. Diagnosis (25 Nov, one frame, three prompts, PyTorch FP32
reference): encoder chunks fine (FPN within ~1%, FP16 noise); DETR encoder fine (0.1%); decoder layer 0 fine also
for the presence row; the presence logit wrong (sign flipped for 2 of 3 prompts), deterministic. Marked tensors
inside the presence head: LayerNorm output right (0.2%), first Linear output off by >100%; not a transposed or
foreign weight, not another row, not the LayerNorm parameters, not the Identity-shared weights, not a stream race;
still wrong as Gemm. PyTorch FP16 gets the presence right, so it is TensorRT, not precision: the same family as
A21 (a one-row matrix product in a Myelin fusion). Workaround: `sam3_parts.RowMLP` runs the presence head on the
row repeated 8 times (`--trt-fix rep8`) or as multiply-and-sum (`reduce`); both fix it (presence +2.508/-3.043/-0.676
vs +2.506/-3.040/-0.674) and compute the same function (5e-7). An FP32 decoder engine cannot be built on the board
(the fused node needs 1.76 GB). Faulty results kept in `runs/e6_edge_640/jetson_jp6/_superseded_2026-11-25/`.
**Second variant (`rep8`, 25 Nov, board rebooted, clocks locked again, desktop apps closed), same rule, same
reference masks: PASS**: IoU 0.9908 (median 0.9976, min 0.865, 1 of 90 frames < 0.9), instances/frame 41.2 vs
41.1. Result: 3,156 ms/frame (0.32 FPS; encoder 1,118 ms, 124.5 ms per prompt, one prompt 1,265 ms), 1.6 GB,
14.1 W, 45 J/frame (PyTorch bf16: 5,183 ms, 3.2 GB, 12.7 W, 66 J). TrashScribe 25-33x faster (PyTorch vs PyTorch
30-36x); SAM 3's image encoder alone is 9-12x the whole TrashScribe pipeline. The faulty engine ran at 3,077 ms and
14.0 W with Firefox open: the desktop apps did not change the measurement.
Decision: SAM 3 TensorRT FP16 row added to `tab:edge`; abstract, introduction, contribution 4, implementation
details, on-device paragraph, limitations (now: no INT8), future work (EfficientSAM3 students) and conclusions say
3.2 s per frame even with TensorRT instead of 5.2 s / "not attempted". albumentations installed on the board
(tests: 14 passed).

## A23 — 2026-11-25 — severe review of the manuscript after the JetPack 6.2 measurements
Checked every on-device number against A21/A22. Fixed: stale depth gate in the pipeline figure; overloaded
Implementation paragraph (split); "too large" → "too slow" for SAM 3 on the device; unhedged interpretations;
missing SegFormer finding in the conclusions; on-device table 24 pt too wide (TRT/PT abbreviations, 4 pt column
spacing); caption trimmed to column definitions.

---

## Template for the next entries
```
## Ax — date — run name(s)
Hypothesis:
Change (config diff):
Result (from list_runs.py):
Decision:
```

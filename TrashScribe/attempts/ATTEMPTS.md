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

---

## Template for the next entries
```
## Ax — date — run name(s)
Hypothesis:
Change (config diff):
Result (from list_runs.py):
Decision:
```

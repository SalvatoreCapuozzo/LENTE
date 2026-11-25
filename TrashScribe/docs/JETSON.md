# Jetson Orin Nano deployment

Target: Jetson Orin Nano 8 GB, JetPack 6.x (L4T r36, CUDA 12.x, TensorRT 10.x, Python 3.10), MAXN mode.
**Paper numbers (Table `tab:edge`) come from the JetPack 6.2 run of 24 Nov 2026**:
`runs/e6_edge_640/jetson_jp6/summary.md` (git-ignored; copy kept on the board). The JetPack 5.1.1 run of 18 Nov
(`runs/e6_edge_640/jetson/`) is cited once in the paper for comparison.

## JetPack 6.2 (24 Nov 2026): the measurements in the paper

The board was reflashed to **JetPack 6.2** (L4T R36.5.2, CUDA 12.6, TensorRT 10.3, cuDNN 9.3, Python 3.10;
hostname `ubuntu`). Power modes on this board: 0 = 15 W, 1 = 7 W (no MAXN SUPER entry in its nvpmodel.conf).
All numbers with `sudo jetson_clocks` (GPU locked at 624.75 MHz), 15 W, desktop on, Firefox closed.

* **Python env** (`.venv`, git-ignored): `python3 -m venv --system-site-packages .venv` (sees JetPack's
  TensorRT 10.3 bindings), then torch 2.8.0 / torchvision 0.23.0 from `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple`
  and [requirements-jetson.txt](../requirements-jetson.txt) (every pin there has a reason written next to it:
  torch 2.9+ of that index needs libcudss, onnx 1.23 writes IR 14 that onnxruntime 1.23 rejects, the system
  Pillow/SciPy are too old). The JetPack 5 venv (copied from a Mac, unusable) was deleted on 25 Nov.
* **GPU devfreq node renamed** on JetPack 6 (`17000000.gpu`, was `17000000.ga10b`): the old clock-lock test
  compared two empty strings and reported locked clocks. Fixed in `jetson_suite.sh`, `jetson_env.py`,
  `jetson_progress.sh`.
* **Power without root**: on JetPack 6 the INA3221 rails are readable by the user, so the suite runs
  `tegrastats --interval 500` itself into `$OUT/tegrastats.log` (VDD_IN = board input).
* **Engines**: `runs/e6_edge_640/export/jp6/` (`ENG_DIR`), so the JetPack 5 engines stay in `export/`.
  TensorRT 10.3 still rejects the OneHot shape computation of the HF mask decoder, so `build_trt.sh` constant-folds
  the graph (`onnx_for_trt85.py --keep-layernorm`: LayerNormalization kept, 3,852 -> 875 nodes, outputs identical).
* **TensorRT 10.3 fusion bug** (found 24 Nov): FP32 and FP16 engines gave a different waste mask than ONNX
  Runtime (max |difference| 0.94, waste-mask IoU 0.81-0.86; contour and descriptor maps fine). Polygraphy with
  intermediate tensors marked, then a bisection over the 234 tensors only on the waste path: marking
  `output_hypernetworks_mlps.0/proj_in/MatMul_output_0` alone fixes it. `onnx_for_trt85.py --keep-layernorm` now adds
  the four hypernetwork `proj_in` outputs as extra graph outputs (the runtime ignores them; no latency cost:
  47.9 ms before and after). After the fix: FP32 engine vs ONNX Runtime max 2e-5, FP16 waste-mask IoU 0.9975-0.9996.
  The faulty results are kept in `runs/e6_edge_640/jetson_jp6/_superseded_2026-11-24/` (README inside).
* **SAM 3 on the device** (`scripts/bench_sam3_edge.py`, step `sam3_bf16` of the suite): PyTorch, bf16, the 16
  prompts and the inference path of the accuracy numbers. Needs the gated `facebook/sam3` weights in the HF cache
  (`hf auth login` with an account that accepted the licence, then the safetensors snapshot without `sam3.pt`).
  transformers' `post_process_instance_segmentation` builds int64 masks of all kept instances at frame size and was
  refused 1 GB on a cluttered TACO frame (NvMap ENOMEM); the script computes the same masks in chunks of 8,
  boolean (verified identical on 12 frame-prompt pairs, `--check-post`).
* **SAM 3 in TensorRT** (25 Nov, A22; `SAM3_TRT=1` in the suite): `scripts/sam3_parts.py` splits SAM 3 into the image
  encoder (three FPN levels) and the detector/mask decoder for one prompt, loaded part by part from the FP32
  checkpoint (bit-identical to `Sam3Model` in PyTorch); `export_sam3_onnx.py` writes the encoder as four FP16 graphs of
  8 ViT layers (one engine of the whole 454 M encoder was OOM-killed twice: parsing alone takes 2.1 GB), the decoder
  in FP32 and the 16 text embeddings once (no text encoder on the device); `build_sam3_trt.sh` builds the engines;
  `bench_sam3_edge.py --backend trt` runs them back to back with one shared activation buffer, same pre- and
  post-processing as the PyTorch run. **TensorRT 10.3 miscomputes the presence head** on the single presence token
  (deterministic; LayerNorm input right, first Linear off by >100%, presence logit with the wrong sign; still wrong as
  Gemm; PyTorch FP16 is right): the first run failed the agreement rule (IoU 0.649, kept in
  `jetson_jp6/_superseded_2026-11-25/`). `sam3_parts.RowMLP` runs the head on the token repeated 8 times
  (`--trt-fix rep8`, same function to 5e-7) and fixes it. Agreement rule (fixed before any TensorRT output,
  `sam3_agreement.py`): mean union-mask IoU with PyTorch bf16 on the 90 frames >= that of PyTorch FP32 (0.9923) - 0.02.
  Result: **0.9908, pass** (median 0.998, 1 of 90 frames below 0.9; 41.2 vs 41.1 instances/frame). Tools for the
  next such bug: `scripts/onnx_mark.py` (mark tensors as outputs), polygraphy.
* **Memory** is the drop of MemAvailable from before the model is loaded to the lowest point while it runs
  (`trashscribe.device.MemWatch`; includes CUDA context and TensorRT workspace; the desktop adds +-0.3 GB of noise).

| 15 W, clocks locked | network (ms) | full pipeline, TACO / UAVVaste / ZeroWaste-f (ms) | memory | board power |
|---|---|---|---|---|
| TrashScribe TensorRT FP16 | 47.9 (trtexec), 51.8 (Python) | 103.5 / 96.9 / 125.6 (9.7 / 10.3 / 8.0 FPS) | 0.5 GB | 13.8 W (idle 5.8), 0.72 J/frame |
| TrashScribe TensorRT FP32 | 96.7 (trtexec), 101.5 (Python) | 150.7 / 145.9 / 169.6 | 0.6 GB | 14.6 W |
| TrashScribe PyTorch FP16 | 93.4 | 147.6 / 144.8 / 172.6 | 1.1 GB | - |
| SAM 3 PyTorch bf16, 16 prompts | 5,141 (encoder 1,467 + 16 x 230) | 5,181 / 5,176 / 5,191 (0.19 FPS); one prompt 1,716 | 3.2 GB peak (1.9 GB loaded) | 12.7 W, 66 J/frame |
| SAM 3 TensorRT FP16, 16 prompts | 3,110 (encoder 1,118 + 16 x 124.5) | 3,146 / 3,155 / 3,167 (0.32 FPS); one prompt 1,265 | 1.6 GB (engines 0.98 GB) | 14.1 W, 45 J/frame |

Reproduce (everything resumable; about 40 min plus 10 min for SAM 3):

```bash
cd ~/LENTE/TrashScribe
python scripts/fetch_edge_images.py --n 30          # same deterministic 30 test images per dataset as on JetPack 5
sudo jetson_clocks                                  # the suite waits for it
systemd-run --user --unit=trashscribe-jetson --collect --working-directory=$PWD \
  --setenv=OUT=runs/e6_edge_640/jetson_jp6 --setenv=ENG_DIR=runs/e6_edge_640/export/jp6 \
  -p StandardOutput=append:$PWD/runs/e6_edge_640/jetson_jp6/suite.log \
  -p StandardError=append:$PWD/runs/e6_edge_640/jetson_jp6/suite.log bash scripts/jetson_suite.sh
bash scripts/jetson_progress.sh --watch              # defaults to the latest runs/e6_edge_640/jetson*/
# SAM 3 in TensorRT as well: the same systemd-run with --setenv=SAM3_TRT=1 (about 1 h the first time)
```

## The device as it was on 18 Nov 2026: JetPack 5.1.1

The owner's board (`unbutu`, Orin Nano Developer Kit) runs **JetPack 5.1.1** (L4T R35.3.1, CUDA 11.4,
TensorRT 8.5.2, cuDNN 8.6, Python 3.8), power mode 0 = 15 W (MAXN of the Orin Nano on JetPack 5; MAXN SUPER
needs JetPack 6.2). What changed to make the pipeline run there:

* **Python env**: `.venv` (git-ignored) on top of the conda env `dl_env`, which already holds NVIDIA's
  `torch 2.0.0+nv23.05` with CUDA; the JetPack `tensorrt` and `onnx_graphsurgeon` packages are symlinked into it
  from `/usr/lib/python3.8/dist-packages`; `onnxruntime 1.19.2` (CPU, for `--check`) and `pytest` installed in it.
  Recreate with:
  ```bash
  ~/anaconda3/envs/dl_env/bin/python -m venv --system-site-packages .venv
  ln -s /usr/lib/python3.8/dist-packages/{tensorrt,onnx_graphsurgeon} .venv/lib/python3.8/site-packages/
  .venv/bin/pip install --only-binary=:all: onnxruntime pytest
  ```
* **Graph for TensorRT 8.5**: TRT 8.5 has no LayerNormalization importer and cannot use OneHot to compute
  shapes (HF mask decoder). `scripts/onnx_for_trt85.py` decomposes the 34 LayerNormalization nodes and
  constant-folds the graph (3,852 → 1,184 nodes; max difference to the original under ONNX Runtime 8.5e-6).
  `build_trt.sh` calls it automatically when TensorRT < 8.6 (output `trashscribe_640_trt85.onnx`); `onnxsim`
  has no aarch64 wheel for Python 3.8 and is not needed. The engine keeps the name `trashscribe_640_<prec>.engine`.
* **Not possible on JetPack 5.1**: SAM 3 and the PyTorch backend need `transformers 5.2` (Python ≥ 3.10).
  Done on JetPack 6.2 (section above).
* **Root-only steps** (the agent has no sudo): `sudo jetson_clocks` (GPU otherwise idles at 306 of 625 MHz) and
  `sudo tegrastats` (the INA3221 power rails are not readable by the user).

All measurements run in one resumable script, detached from VS Code:

```bash
cd ~/LENTE/TrashScribe
python scripts/fetch_edge_images.py --n 30          # 30 test-split images each of TACO, ZeroWaste-f, UAVVaste (range requests, no full archives)
systemd-run --user --unit=trashscribe-jetson --collect --working-directory=$PWD \
  -p StandardOutput=append:$PWD/runs/e6_edge_640/jetson/suite.log -p StandardError=append:$PWD/runs/e6_edge_640/jetson/suite.log \
  bash scripts/jetson_suite.sh
sudo jetson_clocks                                                            # before the timed steps
sudo tegrastats --interval 500 --logfile runs/e6_edge_640/jetson/tegrastats.log   # leave running during the suite
bash scripts/jetson_progress.sh --watch                                       # progress
```

Steps (each leaves `runs/e6_edge_640/jetson/.done_<step>`; delete a marker to redo it): `env`, wait for locked
clocks (≤ 90 min) and for ≥ 3 GB of free memory (≤ 15 min; close Firefox), `env_bench`, `build_fp16`, `build_fp32`
(trtexec, also measures the network with CUDA graphs; builds wait because TensorRT picks tactics by timing
them and skips the fastest ones of the fused attention regions, which ask 1.4 to 1.7 GB, when memory is short:
a first FP16 build with 0.6 GB free skipped 81 tactics and ran at 120 ms, kept as `trashscribe_640_fp16_lowmem.*`), `bench_<prec>_<dataset>` (network + full pipeline on the real images + FP16/FP32 check against ONNX
Runtime FP32 on 5 real frames), `power` (60 s idle, then 3,000 / 3,000 network iterations per precision, phases
written to `status.log`), `infer_<dataset>` (overlays + written reports with the FP16 engine), then
`summary.md` / `summary.json` (`scripts/jetson_summary.py`, also cuts `tegrastats.log` into the phases).
MJU-Waste (Google Drive) and BePLi v2 (one 9.5 GB tar.gz, about 0.6 MB/s) are not fetched: copy 30 test
images of each from the workstation into `~/data/waste/_edge_test/{mju,bepli}/` to add them.

## Files not in git

> **Agent: warn the owner before any Jetson step.** The files below are excluded by `.gitignore`
> (`runs/*/`, `*.onnx`, `*.engine`, `*.pt`, `export/`) or live outside the repository. A fresh clone on the
> Jetson does not have them. Ask the owner to copy them, and check they are present before running anything.

| Needed for | File (on the training workstation) | Size | How it is produced |
|---|---|---|---|
| **TensorRT engine build (required)** | `runs/e6_edge_640/export/trashscribe_640.onnx` | 127.5 MB | `python scripts/export_onnx.py runs/e6_edge_640` (done 17 Nov; max \|ONNX − PyTorch\| 2.4e-5) |
| **naming and descriptions on device (required)** | `runs/e6_edge_640/export/atlas_cache.pt` | 0.46 MB | copied by `export_onnx.py` from `runs/e6_edge_640/atlas_cache.pt` |
| export metadata (recommended) | `runs/e6_edge_640/export/export_meta.json` | 1 KB | written by `export_onnx.py` (input size 640, output shapes, naming weights) |
| real-image benchmark / demo (recommended) | a few test images from `~/data/waste/<dataset>/...` (test splits) | any | copy 20–50 images per environment; `bench_edge.py --images <folder>` |
| PyTorch backend on device (optional) | `runs/e6_edge_640/{best.pt, config.yaml, atlas_cache.pt}` + HF cache `models--facebook--sam2.1-hiera-tiny` (149 MB) + the repo's `atlas/` and `configs/` | ~180 MB | already on the workstation |
| embedding NEW atlas phrases on device (optional, usually done on the workstation) | HF cache `models--openai--clip-vit-base-patch16` (1.2 GB) + `transformers==5.2.0` | 1.2 GB | already on the workstation |
| SAM 3 latency for the comparison claim (optional, recommended) | HF cache `models--facebook--sam3` (3.3 GB, gated licence on Hugging Face) + `transformers==5.2.0` + PyTorch | 3.3 GB | already on the workstation; may not fit in fp32 on 8 GB, try fp16 |
| site-adapted models (E9/E10), if a site-specific engine is wanted | `runs/e9_sam3_<site>/` or `runs/e10_k10_<site>/` | ~100 MB each | trained at **1024 px**: export with `export_onnx.py` (engine at 1024 is slower than the 640 one) |

Engines (`*.engine`) are tied to the GPU, TensorRT and JetPack version: **never copy an engine**, copy the ONNX
and build on the device.

## Software on the device

From [requirements-jetson.txt](../requirements-jetson.txt): NVIDIA's PyTorch wheel for the JetPack (not PyPI;
PyTorch is needed even for the TensorRT path, the runtime uses torch CUDA tensors as buffers), TensorRT Python
bindings from JetPack, `numpy<2`, `scipy`, `opencv-python` (or JetPack OpenCV), `PyYAML`, `pillow`. Optional:
`onnxsim` (simplifies the graph before the build), `onnxruntime-gpu` from the jetson-ai-lab index (needed for
`bench_edge.py --check`), `transformers==5.2.0` (only for new phrases or SAM 3). `albumentations` is not
needed by the runtime (its import was made lazy on 17 Nov); it is installed on the board since 25 Nov so that all of
`tests/` runs (14 passed), with opencv-python-headless pinned to 4.10 (the latest needs numpy 2).

## Procedure

```bash
# on the Jetson, in the repository root (TrashScribe/)
sudo nvpmodel -m 0 && sudo jetson_clocks                       # MAXN (MAXN SUPER on JetPack 6.2)
mkdir -p runs/e6_edge_640/export                                # then copy the files listed above here
bash scripts/build_trt.sh runs/e6_edge_640/export/trashscribe_640.onnx fp16
python3 scripts/bench_edge.py --engine runs/e6_edge_640/export/trashscribe_640_fp16.engine \
        --onnx runs/e6_edge_640/export/trashscribe_640.onnx --cache runs/e6_edge_640/export/atlas_cache.pt \
        --check --images <folder of test images>
python3 scripts/infer.py --engine <engine> --cache runs/e6_edge_640/export/atlas_cache.pt --source <folder|video|0>
tegrastats --interval 500 --logfile power.log                    # power while bench_edge runs
```

`build_trt.sh` always names the engine after the original graph (`trashscribe_640_fp16.engine`), whether it
built from the onnxsim graph (`*_sim.onnx`), the TRT 8.5 graph (`*_trt85.onnx`) or the raw one. `--check` compares the FP16 engine with ONNX Runtime FP32 (SAM-family
encoders sometimes drift in FP16).

## Measurements the paper needs (Table `tab:edge`, Sect. "Edge deployment"): all done on JetPack 6.2, 24 Nov 2026

1. Network-only latency and FPS of the FP16 engine at 640 px (and FP32 for reference), peak memory.
2. Full-pipeline latency (pre-processing + network + instances + naming + report) on real images, split by stage.
3. FP16 vs FP32 agreement (`--check`).
4. Power from `tegrastats` during the benchmark.
5. **SAM 3 on the same device** (or a statement that it does not fit), with the 16 prompts: this is what turns
   "34 M vs 848 M parameters" into "real time vs not real time" in the comparison with SAM 3.
6. Optional: a NanoSAM-style detector→segmenter pipeline whose cost grows with the number of objects.

## Known gaps

* The export graph (`TrashScribeEdge`) outputs waste, contour and descriptor maps only. The E8 distilled CLIP
  head (open reading of new classes and properties) is **not** exported; deploying it needs a fourth output and
  phrase scoring for it in `edge_runtime.py` (see NEXT_STEPS). The deployed model (E6) does not have that head.
* The deployed model is E6 (E4 recipe, 640 px, all five datasets). The site-adaptation results (E3b, E9, E10)
  were obtained at 1024 px; a 640-px site-adapted model has not been trained.
* The paper's implementation details state JetPack 6.2 (L4T 36.5.2), TensorRT 10.3, CUDA 12.6, PyTorch 2.8 and the
  TensorRT 10.3 workarounds. SAM 3 is measured in PyTorch bf16 and in TensorRT FP16; not tried: INT8 for either model,
  the distilled EfficientSAM3 students.

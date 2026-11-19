# Jetson Orin Nano deployment

Target: Jetson Orin Nano 8 GB, JetPack 6.x (L4T r36, CUDA 12.x, TensorRT 10.x, Python 3.10), MAXN mode.
**Nothing has been run on the device yet.** Every on-device number in the paper is a red `\tbd`.

## The device actually in use (18 Nov 2026): JetPack 5.1.1

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
  Measuring SAM 3 on the device needs JetPack 6 (reflash) or a container with a newer Python and torch.
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
`bench_edge.py --check`), `transformers==5.2.0` (only for new phrases or SAM 3). `albumentations` is **not**
needed on the device (its import was made lazy on 17 Nov).

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

## Measurements the paper needs (Table `tab:edge`, Sect. "Edge deployment")

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
* JetPack/TensorRT versions are `\tbd` in the paper's implementation details.

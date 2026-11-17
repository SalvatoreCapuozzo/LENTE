# Jetson Orin Nano deployment

Target: Jetson Orin Nano 8 GB, JetPack 6.x (L4T r36, CUDA 12.x, TensorRT 10.x, Python 3.10), MAXN mode.
**Nothing has been run on the device yet.** Every on-device number in the paper is a red `\tbd`.

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
python3 scripts/bench_edge.py --engine runs/e6_edge_640/export/trashscribe_640_sim_fp16.engine \
        --onnx runs/e6_edge_640/export/trashscribe_640.onnx --cache runs/e6_edge_640/export/atlas_cache.pt \
        --check --images <folder of test images>
python3 scripts/infer.py --engine <engine> --cache runs/e6_edge_640/export/atlas_cache.pt --source <folder|video|0>
tegrastats --interval 500 --logfile power.log                    # power while bench_edge runs
```

`build_trt.sh` names the engine after the simplified graph when `onnxsim` is installed (`*_sim_fp16.engine`),
otherwise `trashscribe_640_fp16.engine`. `--check` compares the FP16 engine with ONNX Runtime FP32 (SAM-family
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

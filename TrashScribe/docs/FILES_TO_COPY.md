Required: runs/e6_edge_640/export/trashscribe_640.onnx (127.5 MB, exported tonight; matches PyTorch to 2.4e-5) and runs/e6_edge_640/export/atlas_cache.pt.
Recommended: export_meta.json from the same folder, and 20–50 test images per environment for a real-image benchmark.
For the SAM 3 comparison: the SAM 3 weights, 3.3 GB from ~/.cache/huggingface (a gated licence on Hugging Face). Measuring SAM 3 on the device is what supports "it can't run in real time there".

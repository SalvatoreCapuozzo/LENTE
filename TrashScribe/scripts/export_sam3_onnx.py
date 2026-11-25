"""Export SAM 3 for TensorRT on the Jetson (A22): two ONNX graphs and the stored text embeddings.

    python scripts/export_sam3_onnx.py --out runs/e6_edge_640/export/jp6/sam3
    trtexec --onnx=<out>/sam3_vision.onnx --fp16 --saveEngine=<out>/sam3_vision_fp16.engine
    trtexec --onnx=<out>/sam3_decoder_b1.onnx --fp16 --saveEngine=<out>/sam3_decoder_b1_fp16.engine
(scripts/jetson_suite.sh does all three with SAM3_TRT=1.)

* sam3_vision_<k>.onnx, k = 0..3: the image encoder in four chunks of 8 ViT layers (sam3_parts.CHUNKS):
  pixel_values (1, 3, 1008, 1008) float32 -> hidden (1, 72, 72, 1024) -> ... -> fpn0/fpn1/fpn2 float32. One
  graph of the whole encoder (946 MB) could not be built: trtexec was OOM-killed twice (parsing alone 2.1 GB).
  Exported in float16 (--vision-dtype): the weights are those an FP16 engine uses anyway, and an FP32 export
  (1.8 GB of weights, held three times by the exporter) does not fit next to the desktop on the 8 GB board.
  LayerNorm is opset-17 LayerNormalization (TensorRT computes it in FP32); RoPE in FP32 as in the library.
  The chunks are checked against the unsplit module in PyTorch before export (identical).
* sam3_decoder_b<B>.onnx: fpn0/1/2 + text_embeds (B, 32, 256) + text_mask (B, 32) int32 -> pred_logits (B, 200),
  presence_logits (B, 1), pred_masks (B, 200, 288, 288); FP32 weights (25 M).
* text_<prompt set>.pt: the 16 prompts' text embeddings in FP32 (Sam3Model.get_text_features); with them the
  353 M text encoder never goes on the device.
Agreement with PyTorch is checked on whole frames by bench_sam3_edge.py --save-masks + sam3_agreement.py.
"""
import _bootstrap  # noqa: F401
import argparse
import gc
import json
import time
from pathlib import Path

import torch

import sam3_parts as P

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="facebook/sam3")
ap.add_argument("--out", required=True)
ap.add_argument("--prompts", default="objects16", choices=list(P.PROMPTS))
ap.add_argument("--parts", default="text,vision,decoder")
ap.add_argument("--vision-dtype", default="fp16", choices=["fp16", "fp32"])
ap.add_argument("--batch", type=int, nargs="+", default=[1], help="decoder graphs for these prompt batch sizes")
ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--device", default="cuda")
ap.add_argument("--no-trt-fix", action="store_true", help="export the decoder without sam3_parts.trt_workarounds")
ap.add_argument("--trt-fix", default="rep8", choices=["rep8", "reduce"], help="sam3_parts.RowMLP mode")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
snap = P.snapshot(a.model)
cfg = P.config(snap)
parts = a.parts.split(",")
dev = torch.device(a.device)
info = {"model": a.model, "torch": torch.__version__, "opset": a.opset}
info_f = out / "export_info.json"
if info_f.exists():
    info = {**json.loads(info_f.read_text()), **info}


def export(module, args, path, inputs, outputs):
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                          opset_version=a.opset, do_constant_folding=True, dynamo=False)
    print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB in {time.perf_counter() - t0:.0f} s", flush=True)
    return {"file": path.name, "inputs": inputs, "outputs": outputs, "mb": path.stat().st_size / 1e6}


if "text" in parts:
    t = P.text_embeddings(snap, cfg, P.PROMPTS[a.prompts])
    torch.save(t, out / f"text_{a.prompts}.pt")
    info["text"] = {"file": f"text_{a.prompts}.pt", "prompts": t["prompts"], "shape": list(t["embeds"].shape),
                    "tokens": t["mask"].sum(1).tolist()}
    print("text embeddings", t["embeds"].shape, flush=True)
    del t
    gc.collect()

if "vision" in parts:
    dt = torch.float16 if a.vision_dtype == "fp16" else torch.float32
    v = P.load_vision(snap, cfg, dev, dt)
    x = torch.randn(1, 3, P.SIZE, P.SIZE, device=dev)
    chunks = [P.VisionChunk(v, s, e) for s, e in P.CHUNKS]
    with torch.no_grad():                   # the chunks compute exactly the unsplit encoder
        ref, h = v(x), x
        for c in chunks:
            h = c(h)
        assert all(torch.equal(r, o) for r, o in zip(ref, h)), "chunked encoder differs from Sam3Vision"
    del ref, h
    info["vision"] = {"dtype": a.vision_dtype, "chunks": []}
    h = x
    for k, (c, (s, e)) in enumerate(zip(chunks, P.CHUNKS)):
        ins = ["pixel_values"] if k == 0 else ["hidden_in"]
        outs = ["fpn0", "fpn1", "fpn2"] if k == len(chunks) - 1 else ["hidden_out"]
        r = export(c, (h,), out / f"sam3_vision_{k}.onnx", ins, outs)
        info["vision"]["chunks"].append({**r, "layers": [s, e]})
        with torch.no_grad():
            h = c(h)
    del v, x, h, chunks
    gc.collect()
    torch.cuda.empty_cache()

if "decoder" in parts:
    d = P.load_decoder(snap, cfg, dev, torch.float32)
    if not a.no_trt_fix:
        P.trt_workarounds(d, a.trt_fix)
    info["trt_workarounds"] = None if a.no_trt_fix else f"presence head RowMLP({a.trt_fix})"
    g = P.SIZE // cfg.vision_config.backbone_config.patch_size
    f = [torch.randn(1, cfg.vision_config.fpn_hidden_size, g * s, g * s, device=dev) for s in (4, 2, 1)]
    for b in a.batch:
        te = torch.randn(b, 32, cfg.detr_encoder_config.hidden_size, device=dev)
        tm = torch.zeros(b, 32, dtype=torch.int32, device=dev)
        tm[:, :4] = 1                       # padded prompts, as all 16 are (3 to 6 tokens of 32)
        info[f"decoder_b{b}"] = export(d, (*f, te, tm), out / f"sam3_decoder_b{b}.onnx",
                                       ["fpn0", "fpn1", "fpn2", "text_embeds", "text_mask"],
                                       ["pred_logits", "presence_logits", "pred_masks"])

info_f.write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))

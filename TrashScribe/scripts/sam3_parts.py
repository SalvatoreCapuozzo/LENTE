"""SAM 3 split into the parts that run on the device, for TensorRT (A22, 24 Nov 2026).

* Sam3Vision: the image encoder (ViT, 1008 px) and the three FPN levels the detector uses (288, 144, 72 px);
  the fourth level (36 px) only feeds the video tracker. For TensorRT it is exported as VisionChunk graphs of
  8 layers each (the 454 M encoder in one piece got the engine build OOM-killed twice on the 8 GB board:
  parsing alone takes 2.1 GB); the chunks run back to back and share one activation buffer (TrtEngine).
* Sam3Decoder: Sam3Model.forward for text prompts on precomputed image features (transformers 5.2,
  no box prompts): DETR encoder, DETR decoder, dot-product scoring, mask decoder. Returns the score logits,
  the presence logit and the 200 mask logits at 288 px, i.e. what the post-processing reads.
* text embeddings: computed once (FP32) and stored; the text encoder is not needed on the device.

Each part is built on the meta device and filled straight from the FP32 checkpoint, so the 8 GB board
never holds the whole 848 M model (the text encoder alone is 353 M). Used by export_sam3_onnx.py (ONNX graphs
for trtexec) and bench_sam3_edge.py (--backend parts: these modules in PyTorch; --backend trt: the engines).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

SIZE = 1008                       # Sam3Processor resizes every frame to 1008 x 1008
# fixed on 15 Nov in scripts/baseline_zeroshot.py (objects16); copied, not imported (that script runs on import)
PROMPTS = {"waste4": ["litter", "trash", "garbage", "plastic waste"],
           "objects16": ["litter", "trash", "garbage", "plastic waste", "bottle", "can", "cup", "plastic bag",
                         "food wrapper", "carton", "cardboard box", "paper", "bottle cap", "plastic fragment",
                         "styrofoam piece", "cigarette butt"]}
PREFIX = "detector_model."        # checkpoint prefix (Sam3Model._checkpoint_conversion_mapping)


def snapshot(model_id="facebook/sam3") -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


def config(snap: Path, attn="sdpa"):
    from transformers import Sam3Config
    cfg = Sam3Config.from_pretrained(snap)
    for c in (cfg, cfg.geometry_encoder_config, cfg.detr_encoder_config, cfg.detr_decoder_config,
              cfg.mask_decoder_config, cfg.vision_config, cfg.vision_config.backbone_config):
        c._attn_implementation = attn          # as Sam3Model.__init__ passes it to the sub-configs
    return cfg


def load_part(module: nn.Module, snap: Path, prefix: str, device="cpu", dtype=torch.float32):
    """Fill a module built on the meta device with the checkpoint tensors under `prefix` (strict)."""
    from safetensors import safe_open
    want = set(module.state_dict().keys())
    sd = {}
    with safe_open(str(snap / "model.safetensors"), "pt", device=str(device)) as f:
        for k in f.keys():
            if k.startswith(prefix) and k[len(prefix):] in want:
                t = f.get_tensor(k)
                sd[k[len(prefix):]] = t.to(dtype) if t.is_floating_point() else t
    missing = want - set(sd)
    if missing:
        raise RuntimeError(f"{prefix}: {len(missing)} tensors not in the checkpoint, e.g. {sorted(missing)[:5]}")
    module.load_state_dict(sd, strict=True, assign=True)
    return module


def _fix_buffers(module: nn.Module, cfg, device, dtype):
    """The non-persistent buffers (not in the checkpoint) are recomputed as in the modules' __init__."""
    from transformers.models.sam3 import modeling_sam3 as S
    for m in module.modules():
        if isinstance(m, S.Sam3ViTRotaryEmbedding):
            ref = S.Sam3ViTRotaryEmbedding(cfg.vision_config.backbone_config, end_x=m.end_x, end_y=m.end_y,
                                           scale=m.scale)
            m.rope_embeddings_cos = ref.rope_embeddings_cos.to(device)   # kept in FP32, as in the library
            m.rope_embeddings_sin = ref.rope_embeddings_sin.to(device)
        if hasattr(m, "position_ids") and getattr(m, "position_ids").is_meta:
            m.position_ids = torch.arange(m.position_ids.shape[-1], device=device).expand((1, -1))
    left = [n for n, t in list(module.named_parameters()) + list(module.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"tensors left on the meta device: {left[:5]}")
    return module


class Sam3Vision(nn.Module):
    """pixel_values (1, 3, 1008, 1008), float32 -> fpn0 (1, 256, 288, 288), fpn1 (.., 144), fpn2 (.., 72), float32.
    `compute` is the dtype the encoder runs in (the ONNX graph for TensorRT FP16 is exported in float16)."""

    def __init__(self, cfg):
        super().__init__()
        from transformers.models.sam3 import modeling_sam3 as S
        self.v = S.Sam3VisionModel(cfg.vision_config)
        self.patch = cfg.vision_config.backbone_config.patch_size
        self.compute = torch.float32

    def forward(self, pixel_values):
        g = pixel_values.shape[-1] // self.patch
        h = self.v.backbone(pixel_values.to(self.compute)).last_hidden_state
        h = h.view(h.shape[0], g, g, -1).permute(0, 3, 1, 2)
        return tuple(layer(h).float() for layer in self.v.neck.fpn_layers[:3])


# ViT layers per chunk: each chunk ends with one of the four global-attention layers (7, 15, 23, 31)
CHUNKS = ((0, 8), (8, 16), (16, 24), (24, 32))


class VisionChunk(nn.Module):
    """Layers [start, end) of Sam3Vision. The first chunk takes pixel_values (float32) and adds the patch and
    position embeddings and the pre-norm; the others take hidden (1, 72, 72, 1024) in the compute dtype; the
    last one ends with the FPN neck (fpn0/1/2, float32). Same operations as Sam3Vision.forward, cut in pieces."""

    def __init__(self, vis: Sam3Vision, start: int, end: int):
        super().__init__()
        self.vis, self.start, self.end = vis, start, end

    def forward(self, x):
        bb = self.vis.v.backbone
        if self.start == 0:
            g = x.shape[-1] // self.vis.patch
            h = bb.embeddings(x.to(self.vis.compute))
            h = bb.layer_norm(h.view(h.shape[0], g, g, -1))
        else:
            h = x
        for layer in bb.layers[self.start:self.end]:
            h = layer(h)
        if self.end < len(bb.layers):
            return h
        h = h.permute(0, 3, 1, 2)
        return tuple(layer(h).float() for layer in self.vis.v.neck.fpn_layers[:3])


class Sam3Decoder(nn.Module):
    """(fpn0, fpn1, fpn2 of one image; text_embeds (B, 32, 256); text_mask (B, 32) int32) ->
    pred_logits (B, 200), presence_logits (B, 1), pred_masks (B, 200, 288, 288). B prompts, one image."""

    def __init__(self, cfg):
        super().__init__()
        from transformers.models.sam3 import modeling_sam3 as S
        self.detr_encoder = S.Sam3DetrEncoder(cfg.detr_encoder_config)
        self.detr_decoder = S.Sam3DetrDecoder(cfg.detr_decoder_config)
        self.mask_decoder = S.Sam3MaskDecoder(cfg.mask_decoder_config)
        self.dot_product_scoring = S.Sam3DotProductScoring(cfg)
        self.fpn_dim = cfg.vision_config.fpn_hidden_size
        self.grid = SIZE // cfg.vision_config.backbone_config.patch_size
        self.pos2 = None

    def init_pos(self, device, dtype=torch.float32):
        """Position encoding of the 72 px level: depends on the shape only. Computed in the model's dtype, as
        Sam3VisionNeck does (in bf16 that rounding changes the decoder output: checked 24 Nov)."""
        from transformers.models.sam3 import modeling_sam3 as S
        pe = S.Sam3SinePositionEmbedding(num_pos_feats=self.fpn_dim // 2, normalize=True)
        self.pos2 = pe((1, self.fpn_dim, self.grid, self.grid), device, dtype)
        return self

    def forward(self, fpn0, fpn1, fpn2, text_embeds, text_mask):
        b = text_embeds.shape[0]
        feats = [f.expand(b, -1, -1, -1) for f in (fpn0, fpn1, fpn2)]
        pos = self.pos2.to(fpn2.dtype).expand(b, -1, -1, -1)
        mask = text_mask.bool()
        enc = self.detr_encoder(vision_features=[feats[-1]], text_features=text_embeds, vision_pos_embeds=[pos],
                                text_mask=mask)
        dec = self.detr_decoder(vision_features=enc.last_hidden_state, text_features=enc.text_features,
                                vision_pos_encoding=enc.pos_embeds_flattened, text_mask=mask,
                                spatial_shapes=enc.spatial_shapes)
        last = dec.intermediate_hidden_states[-1:]           # scores of the last layer only, as Sam3Model uses
        logits = self.dot_product_scoring(decoder_hidden_states=last, text_features=enc.text_features,
                                          text_mask=mask).squeeze(-1)[0]
        masks = self.mask_decoder(decoder_queries=last[0], backbone_features=feats,
                                  encoder_hidden_states=enc.last_hidden_state, prompt_features=text_embeds,
                                  prompt_mask=mask).pred_masks
        return logits, dec.presence_logits[-1], masks


def _import_sam3():
    """Import transformers' SAM 3 code before any `with torch.device("meta")`: imported inside it,
    torch._dynamo fails to initialise (circular import, seen 24 Nov)."""
    from transformers import CLIPTextModelWithProjection  # noqa: F401
    from transformers.models.sam3 import modeling_sam3  # noqa: F401


class RowMLP(nn.Module):
    """Sam3DecoderMLP (2 or 3 Linear + ReLU) on a single token, written so that TensorRT 10.3 does not compute
    a one-row matrix product. TensorRT 10.3 miscomputes the presence head on the one presence token, as a
    MatMul and as a Gemm alike (its LayerNorm input agrees with PyTorch to 0.2%, its first Linear is off by
    over 100%, deterministically: A22, 25 Nov); TrashScribe's one-token hypernetwork MatMul failed the same
    way (A21). mode "rep8": the row repeated 8 times, row 0 kept; mode "reduce": each Linear as a broadcast
    multiply and a sum (no matrix product). Both compute the same function as the MLP."""

    def __init__(self, mlp: nn.Module, mode: str = "rep8"):
        super().__init__()
        self.mlp, self.mode = mlp, mode

    def _linear(self, x, lin):
        if self.mode == "reduce":
            return (x.unsqueeze(-1) * lin.weight.t()).sum(-2) + lin.bias
        return lin(x)

    def forward(self, x):
        lead = [int(s) for s in x.shape[:-1]]           # constant shapes: see export (fixed batch)
        h = x.reshape(-1, 1, int(x.shape[-1]))           # (N, 1, C)
        if self.mode == "rep8":
            h = h.expand(-1, 8, -1).contiguous()
        m = self.mlp
        h = F.relu(self._linear(h, m.layer1))
        if m.layer3 is not None:
            h = self._linear(F.relu(self._linear(h, m.layer2)), m.layer3)
        else:
            h = self._linear(h, m.layer2)
        return h[:, :1].reshape(lead + [int(h.shape[-1])])


def trt_workarounds(decoder: "Sam3Decoder", mode: str = "rep8") -> "Sam3Decoder":
    """Graph changes for TensorRT 10.3 that compute the same function (applied for export only, so the
    `parts` backend stays bit-identical to Sam3Model)."""
    dd = decoder.detr_decoder
    if not isinstance(dd.presence_head, RowMLP):
        dd.presence_head = RowMLP(dd.presence_head, mode)
    return decoder


def load_vision(snap, cfg, device="cuda", dtype=torch.float32):
    _import_sam3()
    with torch.device("meta"):
        m = Sam3Vision(cfg)
    load_part(m.v, snap, PREFIX + "vision_encoder.", device, dtype)
    _fix_buffers(m, cfg, device, dtype)
    m.compute = dtype
    return m.eval()


def load_decoder(snap, cfg, device="cuda", dtype=torch.float32):
    _import_sam3()
    with torch.device("meta"):
        m = Sam3Decoder(cfg)
    for name in ("detr_encoder", "detr_decoder", "mask_decoder", "dot_product_scoring"):
        load_part(getattr(m, name), snap, PREFIX + name + ".", device, dtype)
    _fix_buffers(m, cfg, device, dtype)
    return m.init_pos(device, dtype).eval()


@torch.no_grad()
def text_embeddings(snap, cfg, prompts, device="cpu"):
    """{'prompts', 'embeds' (P, 32, 256) float32, 'mask' (P, 32) int32}: Sam3Model.get_text_features in FP32."""
    from transformers import CLIPTextModelWithProjection, Sam3Processor
    with torch.device("meta"):
        te = CLIPTextModelWithProjection(cfg.text_config)
        proj = nn.Linear(cfg.text_config.hidden_size, cfg.detr_encoder_config.hidden_size)
    load_part(te, snap, PREFIX + "text_encoder.", device)
    load_part(proj, snap, PREFIX + "text_projection.", device)
    _fix_buffers(te, cfg, device, torch.float32)
    proc = Sam3Processor.from_pretrained(snap)
    emb, msk = [], []
    for q in prompts:
        t = proc(text=q, return_tensors="pt").to(device)
        h = te.eval()(input_ids=t.input_ids, attention_mask=t.attention_mask, return_dict=True).last_hidden_state
        emb.append(proj(h)[0].float().cpu())
        msk.append(t.attention_mask[0].to(torch.int32).cpu())
    return {"prompts": list(prompts), "embeds": torch.stack(emb), "mask": torch.stack(msk)}


class TrtEngine:
    """A TensorRT engine with torch CUDA tensors as I/O buffers (as trashscribe.edge_runtime.TrtBackend);
    `bind(name, tensor)` makes an input read another engine's output buffer in place (no copy).
    Engines given the same `shared` list run with one activation buffer (TensorRT USER_MANAGED device memory,
    sized for the largest; safe because they run one after the other on one stream): share_memory(list)."""

    def __init__(self, path, stream=None, shared: list | None = None):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if shared is None:
            self.ctx = self.engine.create_execution_context()
        else:
            self.ctx = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            shared.append(self)
        self.stream = stream or torch.cuda.Stream()
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
              trt.int64: torch.int64, trt.bool: torch.bool}
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.dtypes = {n: dt[self.engine.get_tensor_dtype(n)] for n in self.names}
        self.bufs = {}
        for n in self.names:
            self.bind(n, torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=self.dtypes[n], device="cuda"))

    def bind(self, name, t):
        assert t.is_contiguous() and tuple(t.shape) == tuple(self.engine.get_tensor_shape(name)), name
        assert t.dtype == self.dtypes[name], (name, t.dtype, self.dtypes[name])
        self.bufs[name] = t
        self.ctx.set_tensor_address(name, t.data_ptr())

    def __call__(self, sync=True):
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        if sync:
            self.stream.synchronize()
        return self.bufs


def share_memory(engines: list) -> torch.Tensor:
    """One activation buffer for engines created with the same `shared` list; returns it (keep a reference)."""
    size = max(e.engine.device_memory_size_v2 for e in engines)
    buf = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
    for e in engines:
        e.ctx.set_device_memory(buf.data_ptr(), e.engine.device_memory_size_v2)
    return buf

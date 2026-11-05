"""Unit tests of the properties TrashScribe's design relies on. No downloads needed
(tiny backbone + hash text encoder); the SAM2 test runs only if weights are cached.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trashscribe.data.canonical import Canonicalizer, collate, letterbox  # noqa: E402
from trashscribe.data.mixing import MultiDataset, transplant  # noqa: E402
from trashscribe.data.synthetic import SyntheticAdapter, synthetic_atlas  # noqa: E402
from trashscribe.losses import descriptor_loss, field_layout  # noqa: E402
from trashscribe.models.heads import GeometryGuidance, TextReprojection  # noqa: E402
from trashscribe.models.peft import Adapter, LoRALinear  # noqa: E402
from trashscribe.models.trashscribe import ModelConfig, TrashScribe, pool_objects  # noqa: E402
from trashscribe.naming import Namer, build_cache  # noqa: E402
from trashscribe.schema import Atlas, Schema, entry_target  # noqa: E402
from trashscribe.text_bank import TextBank  # noqa: E402

SCHEMA = Schema.load(ROOT / "atlas/schema.yaml")


def test_atlases_use_schema_vocabulary():
    """Every value in the shipped training atlases exists in the schema (typos would
    silently become 'new' values otherwise)."""
    for name in ("taco", "zerowaste", "binary"):
        a = Atlas.load(ROOT / f"atlas/{name}.yaml")
        ext = SCHEMA.extended([a])
        assert ext.flat == SCHEMA.flat, f"{name}: values outside schema: {set(ext.flat) - set(SCHEMA.flat)}"


def test_taco_atlas_covers_60_classes():
    a = Atlas.load(ROOT / "atlas/taco.yaml")
    assert len(a.entries) == 60


def test_novel_atlas_extends_without_parameters():
    a = Atlas.load(ROOT / "atlas/novel_field_classes.yaml")
    ext = SCHEMA.extended([a])
    assert ext.n_values > SCHEMA.n_values
    assert len(ext.fields) == len(SCHEMA.fields)        # same per-field tau/b reused


def test_entry_target_masks_unknown_fields():
    a = Atlas.load(ROOT / "atlas/zerowaste.yaml")
    t, m = entry_target(a["rigid_plastic"], SCHEMA)
    assert m[SCHEMA.slices["material"]].all() and not m[SCHEMA.slices["shape"]].any()
    assert abs(t[SCHEMA.slices["rigidity"]].sum() - 1) < 1e-6


def test_zero_init_modules_are_identity():
    x = torch.randn(4, 10, 96)
    assert torch.allclose(Adapter(96, 16)(x), x)
    lin = torch.nn.Linear(96, 288)
    lo = LoRALinear(lin, 4, 16.0)
    assert torch.allclose(lo(x), lin(x))
    assert torch.allclose(lo.merged()(x), lin(x), atol=1e-6)
    e = torch.randn(5, 512)
    assert torch.allclose(TextReprojection(512, 32)(e), e)
    rgb, d = torch.randn(2, 3, 32, 32), torch.rand(2, 1, 32, 32)
    assert torch.allclose(GeometryGuidance()(rgb, d), rgb)


def _tiny_model(n_env=8):
    tb = TextBank("hash:64")
    m = TrashScribe(ModelConfig(backbone="tiny", image_size=128), text_dim=64, n_fields=len(SCHEMA.fields), n_env=n_env)
    m.set_vocabulary(tb.encode_ensembles(SCHEMA.phrases()), SCHEMA.field_index(),
                     torch.tensor([f.scope == "contextual" for f in SCHEMA.fields]))
    return m, tb


def test_no_parameter_scales_with_vocabulary():
    """The descriptor head has no per-descriptor parameter: growing the vocabulary
    changes no parameter shape."""
    m, tb = _tiny_model()
    shapes = {n: p.shape for n, p in m.named_parameters()}
    a = Atlas.load(ROOT / "atlas/novel_field_classes.yaml")
    ext = SCHEMA.extended([a])
    m.set_vocabulary(tb.encode_ensembles(ext.phrases()), ext.field_index(),
                     torch.tensor([f.scope == "contextual" for f in ext.fields]))
    assert {n: p.shape for n, p in m.named_parameters()} == shapes
    z = torch.randn(3, 64)
    assert m.score(z).shape == (3, ext.n_values)


def test_pooling_and_ring():
    desc = torch.zeros(1, 4, 8, 8)
    desc[0, 0, :4] = 1.0     # top half carries channel 0
    desc[0, 1, 4:] = 1.0     # bottom half channel 1
    m = torch.zeros(1, 16, 16)
    m[0, :8, :] = 1          # object = top half (stride-4 grid)
    zi, zc, b = pool_objects(desc, [m], ring_cells=1)
    assert zi[0, 0] > 0.99 and zi[0, 1] < 1e-6
    assert zc[0, 1] > 0.99 and zc[0, 0] < 1e-6   # ring lies only below the object


def test_heldout_never_in_training_split():
    a = SyntheticAdapter(n=50, size=64, split="train")
    names = {e for i in range(len(a)) for e in a.load(i).entries}
    assert not names & {"glass bottle", "metal foil", "paper bag"}


def test_transplant_keeps_labels_consistent():
    rng = np.random.default_rng(0)
    src = letterbox(SyntheticAdapter(n=4, size=96, split="val", seed=1).load(0), 128)
    bg = letterbox(SyntheticAdapter(n=4, size=96, split="val", seed=2).load(1), 128)
    out = transplant(src, bg, rng)
    assert len(out.instances) == len(out.entries)
    union = np.zeros_like(out.instances[0])
    for q in out.instances:
        assert (q & union).sum() == 0              # instances never overlap
        union |= q
    assert (out.waste_mask()[union == 1] == 1).all()
    assert out.surroundings == bg.surroundings


def test_end_to_end_tiny_step_and_naming():
    atlas = synthetic_atlas()
    m, tb = _tiny_model()
    canon = Canonicalizer(SCHEMA, atlas, 128, True, {"strength": 0.5})
    ds = MultiDataset([SyntheticAdapter(n=8, size=128)], canon, transplant_p=0.5, cross_dataset_only=False)
    batch = collate([ds[i] for i in range(4)])
    out = m(batch["image"])
    zi, zc, _ = pool_objects(out["desc"], [t.float() for t in batch["inst"]])
    loss = descriptor_loss(m.score(zi, zc), torch.cat(batch["desc_t"]), torch.cat(batch["desc_m"]),
                           field_layout(SCHEMA))
    loss.backward()
    assert torch.isfinite(loss)
    namer = Namer(build_cache(m, SCHEMA, atlas, tb))
    r = namer.name(zi.detach(), zc.detach())
    assert len(r["name"]) == zi.shape[0]


@pytest.mark.skipif(not (Path.home() / ".cache/huggingface/hub/models--facebook--sam2.1-hiera-tiny").exists(),
                    reason="SAM2 weights not cached")
def test_sam2_backbone_shapes_and_frozen_core():
    m = TrashScribe(ModelConfig(backbone="sam2", image_size=256, adapter_dim=16, lora_rank=2), 64, len(SCHEMA.fields), 8)
    o = m(torch.randn(1, 3, 256, 256))
    assert o["waste"].shape == (1, 1, 64, 64) and o["desc"].shape == (1, 64, 32, 32)
    enc = m.backbone.model.vision_encoder
    frozen = [p for n, p in enc.named_parameters() if ".A." not in n and ".B." not in n
              and "adapter" not in n]
    assert all(not p.requires_grad for p in frozen)

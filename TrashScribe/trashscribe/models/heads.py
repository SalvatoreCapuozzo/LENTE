"""Heads of TrashScribe: dense descriptor field, contour map, text re-projection,
per-field scorer, geometric early guidance and the environment adversary."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import LEVEL_CHANNELS


def _gn(c):
    return nn.GroupNorm(min(32, c), c)


class DescriptorField(nn.Module):
    """Dense visual re-projection W_v: encoder features -> text space, at stride 8.

    Every pixel gets a vector that is compared by cosine with descriptor phrases,
    so the same map serves object description (mask pooling) and free-text
    heatmaps (any phrase, at inference).
    """

    def __init__(self, text_dim=512, hidden=256):
        super().__init__()
        _, c1, c2 = LEVEL_CHANNELS
        self.l1 = nn.Conv2d(c1, hidden, 1)
        self.l2 = nn.Conv2d(c2, hidden, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), _gn(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), _gn(hidden), nn.GELU())
        self.proj = nn.Conv2d(hidden, text_dim, 1)

    def forward(self, feats):
        _, f1, f2 = feats
        x = self.l1(f1) + F.interpolate(self.l2(f2), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.fuse(x)
        return self.proj(h), h


class ContourHead(nn.Module):
    """Object-boundary logits at stride 4, used to split touching waste objects
    into instances without per-object decoder calls (cheap on the edge)."""

    def __init__(self, hidden=256, mid=64):
        super().__init__()
        c0 = LEVEL_CHANNELS[0]
        self.red = nn.Conv2d(hidden, mid, 1)
        self.net = nn.Sequential(nn.Conv2d(c0 + mid, mid, 3, padding=1), _gn(mid), nn.GELU(),
                                 nn.Conv2d(mid, 1, 1))

    def forward(self, f0, h8):
        up = F.interpolate(self.red(h8), size=f0.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat([f0, up], 1))


class TextReprojection(nn.Module):
    """W_t e = e + A sigma(B e), A zero-initialised: starts at the raw text space
    and departs from it only where the data supports it (as in the MIRF head)."""

    def __init__(self, dim=512, rank=128):
        super().__init__()
        self.B = nn.Linear(dim, rank)
        self.A = nn.Linear(rank, dim)
        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.A.bias)

    def forward(self, e):
        return e + self.A(F.gelu(self.B(e)))


class FieldScorer(nn.Module):
    """s(i, d) = tau_f * cos(z_i, W_t e_d) + b_f.

    Only tau and b are per FIELD; nothing is per descriptor or per class, so a
    phrase never seen in training is scored through the identical path.
    """

    def __init__(self, n_fields: int, init_tau=10.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.full((n_fields,), float(torch.log(torch.tensor(init_tau)))))
        self.bias = nn.Parameter(torch.zeros(n_fields))

    def forward(self, z: torch.Tensor, P: torch.Tensor, field_idx: torch.Tensor) -> torch.Tensor:
        """z (..., D) visual, P (V, D) re-projected phrases, field_idx (V,) -> (..., V)."""
        cos = F.normalize(z, dim=-1) @ F.normalize(P, dim=-1).T
        return cos * self.log_tau.exp()[field_idx] + self.bias[field_idx]


class GeometryGuidance(nn.Module):
    """LENTE's DSDF early guidance, transposed to waste: the terrain (DEM + slope)
    becomes scene geometry (relative depth + its gradient magnitude), from an
    RGB-D sensor or a monocular depth model. Output is a multiplicative gain
    1 + tanh(.) whose last conv is zero-initialised: identity at step 0, so the
    frozen encoder sees its pre-training distribution until the gate learns."""

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))

    def forward(self, rgb, depth):
        d = depth
        lo = d.amin((2, 3), keepdim=True)
        hi = d.amax((2, 3), keepdim=True)
        d = (d - lo) / (hi - lo + 1e-6)
        g = torch.sqrt(F.conv2d(d, self.kx, padding=1) ** 2 + F.conv2d(d, self.ky, padding=1) ** 2 + 1e-12)
        return rgb * (1.0 + torch.tanh(self.net(torch.cat([d, g * 10], 1))))


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float):
    return _GradReverse.apply(x, lam)


class EnvironmentAdversary(nn.Module):
    """Predicts the surroundings of an object from its INTRINSIC embedding through
    a gradient-reversal layer (Ganin & Lempitsky, 2015). Training it pushes the
    object embedding to carry no information about where the object lies."""

    def __init__(self, dim, n_env, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_env))

    def forward(self, z, lam=1.0):
        return self.net(grad_reverse(F.normalize(z, dim=-1), lam))

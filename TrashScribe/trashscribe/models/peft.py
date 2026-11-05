"""Parameter-efficient modules inherited from LENTE, plus MixStyle.

All of them start as the identity (zero-initialised up/B projections, MixStyle
inactive at eval), so the frozen foundation model is exactly reproduced at step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Bottleneck adapter with residual bypass; up-projection zero-initialised (LENTE)."""

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x with W frozen and B zero-initialised (LENTE)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / r)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(x))

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Plain Linear with the low-rank update folded in (for export)."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.copy_(self.base.weight + self.scaling * self.B.weight @ self.A.weight)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin.to(self.base.weight.device, self.base.weight.dtype)


class MixStyle(nn.Module):
    """Feature-statistics mixing between samples of a batch (Zhou et al., ICLR 2021).

    Applied to early features only during training: it mixes instance-level mean
    and std across images, which are dominated by camera and illumination, i.e.
    by the dataset rather than by the object. Identity in eval mode.
    `channel_last=True` for (B, H, W, C) tensors such as the Hiera patch embedding.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, channel_last: bool = True):
        super().__init__()
        self.p, self.eps, self.channel_last = p, eps, channel_last
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or self.p <= 0 or x.shape[0] < 2 or torch.rand(()) > self.p:
            return x
        dims = (1, 2) if self.channel_last else (2, 3)
        mu = x.mean(dim=dims, keepdim=True)
        sig = (x.var(dim=dims, keepdim=True) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        xn = (x - mu) / sig
        B = x.shape[0]
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        return xn * (sig * lam + sig2 * (1 - lam)) + (mu * lam + mu2 * (1 - lam))


def output_hook(module: nn.Module):
    """forward_hook that routes a module output (tensor or tuple) through `module`."""
    def hook(_m, _args, out):
        if isinstance(out, tuple):
            return (module(out[0]),) + tuple(out[1:])
        return module(out)
    return hook

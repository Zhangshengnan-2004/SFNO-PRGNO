"""V2 multi-scale dynamic gating, designed for this project (not a paper replica)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    """Normalize channels at each spatial position, independently of image size."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class MultiScaleDynamicGating(nn.Module):
    """Shape-preserving MDG with per-pixel softmax weights over spatial scales.

    ``scales`` are downsampling divisors, independent of the reconstruction sf.
    All branches are aligned explicitly, including for odd and rectangular sizes.
    """

    def __init__(self, dim=64, scales=(1, 2, 4), mlp_ratio=4):
        super().__init__()
        scales = tuple(scales)
        if not scales or any(type(s) is not int or s < 1 for s in scales):
            raise ValueError("scales must contain positive integers")
        if scales[0] != 1 or len(set(scales)) != len(scales):
            raise ValueError("scales must start with 1 and contain no duplicates")
        if dim < 1 or int(dim * mlp_ratio) < 1:
            raise ValueError("dim and hidden dimension must be positive")
        self.dim = dim
        self.scales = scales
        self.norm1 = ChannelLayerNorm(dim)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1),
            ) for _ in scales
        ])
        self.gate = nn.Sequential(
            nn.Conv2d(dim * len(scales), dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, len(scales), 1),
        )
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm2 = ChannelLayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1),
        )

    def forward(self, height, width, x, *, return_gates=False):
        if x.ndim != 4 or x.shape[1:] != (self.dim, height, width):
            raise ValueError("expected x with shape [B, dim, height, width]")
        normalized = self.norm1(x)
        features = []
        for scale, branch in zip(self.scales, self.branches):
            size = (max(1, height // scale), max(1, width // scale))
            local = normalized if scale == 1 else F.adaptive_avg_pool2d(normalized, size)
            local = branch(local)
            if local.shape[-2:] != (height, width):
                local = F.interpolate(local, size=(height, width), mode="bilinear", align_corners=False)
            features.append(local)
        gates = self.gate(torch.cat(features, dim=1)).softmax(dim=1)
        fused = sum(feature * gates[:, i:i + 1] for i, feature in enumerate(features))
        output = x + self.proj(fused)
        output = output + self.ffn(self.norm2(output))
        return (output, gates) if return_gates else output

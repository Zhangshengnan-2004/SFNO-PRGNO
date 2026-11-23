"""SFNO-PRGNO V1: physical-residual-guided Galerkin kernel generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from VectorAttention import VectorAttention
from PhysicsGalerkin import PhysicsGuidedGalerkin, PhysicsResidualGuidance


def _to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * x.new_empty(shape).bernoulli_(keep)


def _partition(x, ssl):
    batch, height, width, channels = x.shape
    x = x.view(batch, height // ssl, ssl, width // ssl, ssl, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ssl, ssl, channels)


def _reverse(areas, ssl, height, width):
    batch = int(areas.shape[0] / (height * width / ssl / ssl))
    x = areas.view(batch, height // ssl, width // ssl, ssl, ssl, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class _Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=4, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attention = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        x = (self.attn_drop(attention) @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Progressive_Resampling_integral(nn.Module):
    """Behavior- and state-dict-compatible copy of the original SFNO PR block."""

    def __init__(self, dim=32, input_resolution=16, num_heads=8, ssl=4,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=4, drop=0.0,
                 attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = _to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.ssl = ssl
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, height, width, x):
        x = rearrange(x, "B C H W -> B (H W) C")
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("input feature has wrong size")
        shortcut = x
        x = self.norm1(x)
        permutation = torch.randperm(length, device=x.device)
        inverse = torch.argsort(permutation)
        x = x[:, permutation, :].contiguous().view(batch, height, width, channels)
        areas = _partition(x, self.ssl).view(-1, self.ssl * self.ssl, channels)
        areas = self.attn(areas).view(-1, self.ssl, self.ssl, channels)
        x = _reverse(areas, self.ssl, height, width).view(batch, height * width, channels)
        x = x[:, inverse, :].contiguous()
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return rearrange(x, "B (H W) C -> B C H W", H=height)


class PhysicsIntegral(nn.Module):
    def __init__(self, img_size=64, in_chans=64, head=8, ssl=4):
        super().__init__()
        self.PR = Progressive_Resampling_integral(
            dim=in_chans, input_resolution=img_size, num_heads=head, ssl=ssl
        )
        self.GI = PhysicsGuidedGalerkin(in_chans, head)

    def forward(self, height, width, x, guidance):
        shortcut = x
        x = self.PR(height, width, x)
        x = self.GI(x, guidance)
        return x + shortcut


class SFNO_PRGNO(nn.Module):
    """Drop-in SFNO replacement with the same ``forward(ms, rgb, sf)`` API."""

    def __init__(
        self,
        size=64,
        response_path: str | Path | None = None,
        response: torch.Tensor | np.ndarray | None = None,
        scale_factor: int = 4,
        use_spatial_residual: bool = True,
        use_spectral_residual: bool = True,
        guidance_mode: str = "physics",
    ):
        super().__init__()
        if response is None:
            if response_path is None:
                raise ValueError("SFNO_PRGNO requires response_path or response")
            response = np.load(Path(response_path), allow_pickle=False)["response"]
        response = torch.as_tensor(response, dtype=torch.float32)

        self.hschannels = 31
        self.mschannels = 3
        self.embed = 64
        self.img_size = size
        self.scale_factor = int(scale_factor)
        if guidance_mode not in ("physics", "learned"):
            raise ValueError("guidance_mode must be physics or learned")
        self.guidance_mode = guidance_mode

        # Names match SFNO wherever possible for checkpoint compatibility.
        self.conv_block = nn.Conv2d(34, self.embed, 1)
        self.block1 = PhysicsIntegral(size, self.embed, 8, 4)
        self.block2 = PhysicsIntegral(size, self.embed, 8, 4)
        self.block3 = PhysicsIntegral(size, self.embed, 8, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(self.embed * 4, self.hschannels, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.VI1_1 = VectorAttention(34, 34)
        self.VI1_2 = VectorAttention(34, 34)
        self.VI1_3 = VectorAttention(34, 34)

        # One shared head supplies X_k at all three stages.
        self.intermediate_reconstruction = nn.Conv2d(self.embed, self.hschannels, 1)
        nn.init.zeros_(self.intermediate_reconstruction.weight)
        nn.init.zeros_(self.intermediate_reconstruction.bias)
        self.physics = PhysicsResidualGuidance(
            response=response,
            scale_factor=self.scale_factor,
            feature_channels=self.embed,
            use_spatial=use_spatial_residual,
            use_spectral=use_spectral_residual,
        )
    def _guidance(self, feature, ms_up, ms, rgb):
        estimate = ms_up + self.intermediate_reconstruction(feature)
        if self.guidance_mode == "learned":
            # Identical trainable capacity, without observation residuals or operators.
            return self.physics.lift(estimate)
        guidance, _ = self.physics(estimate, ms, rgb)
        return guidance

    def forward(self, ms, rgb, sf):
        if int(sf) != self.scale_factor:
            raise ValueError(
                f"Model was built for sf={self.scale_factor}, but forward received sf={sf}"
            )
        ms_up = F.interpolate(ms, scale_factor=sf, mode="bicubic", align_corners=False)
        if ms_up.shape[-2:] != rgb.shape[-2:]:
            raise ValueError(f"Upsampled HSI {ms_up.shape[-2:]} != MSI {rgb.shape[-2:]}")
        xt = torch.cat((rgb, ms_up), 1)
        xt = self.VI1_3(self.VI1_2(self.VI1_1(xt)))
        _, _, height, width = xt.shape
        x1 = self.conv_block(xt)

        g1 = self._guidance(x1, ms_up, ms, rgb)
        x2 = self.block1(height, width, x1, g1)
        g2 = self._guidance(x2, ms_up, ms, rgb)
        x3 = self.block2(height, width, x2, g2)
        g3 = self._guidance(x3, ms_up, ms, rgb)
        x4 = self.block3(height, width, x3, g3)

        return self.conv(torch.cat((x1, x2, x3, x4), 1)) + ms_up

    def load_sfno_state_dict(self, state_dict: dict) -> tuple[list[str], list[str]]:
        """Load all shape-compatible baseline weights; return missing/unexpected keys."""
        result = self.load_state_dict(state_dict, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)

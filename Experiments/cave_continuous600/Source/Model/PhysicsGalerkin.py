"""Physics-guided Galerkin operator and matched CAVE degradation operators.

The spectral response used here is an *effective response fitted on training
data*.  It must not be described as a measured sensor response.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn


class LastDimLayerNorm(nn.Module):
    """Keep the normalization used by the original SFNO Galerkin operator."""

    def __init__(self, shape, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class PhysicsGuidedGalerkin(nn.Module):
    """Original SFNO Galerkin integral with physics modulation of Q and K only."""

    def __init__(self, midc: int, heads: int):
        super().__init__()
        if midc % heads:
            raise ValueError(f"midc={midc} must be divisible by heads={heads}")
        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        # Names and shapes intentionally match SFNO.Galerkin_integral.
        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)
        self.kln = LastDimLayerNorm((heads, 1, self.headc))
        self.vln = LastDimLayerNorm((heads, 1, self.headc))
        self.act = nn.GELU()

        # Produces [Delta Q, Delta K]. Zero initialization makes the initial
        # operator exactly equal to the original Galerkin operator.
        self.delta_qk = nn.Conv2d(midc, 2 * midc, 1)
        nn.init.zeros_(self.delta_qk.weight)
        nn.init.zeros_(self.delta_qk.bias)

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        if x.shape != guidance.shape:
            raise ValueError(f"feature {tuple(x.shape)} != guidance {tuple(guidance.shape)}")
        batch, channels, height, width = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1)
        qkv = qkv.reshape(batch, height * width, self.heads, 3 * self.headc)
        q, k, v = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)

        dqk = self.delta_qk(guidance).permute(0, 2, 3, 1)
        dqk = dqk.reshape(batch, height * width, self.heads, 2 * self.headc)
        delta_q, delta_k = dqk.permute(0, 2, 1, 3).chunk(2, dim=-1)
        q = q + delta_q
        k = k + delta_k

        # V and the original K/V normalization are deliberately unchanged.
        k = self.kln(k)
        v = self.vln(v)
        kernel = torch.matmul(k.transpose(-2, -1), v) / (height * width)
        value = torch.matmul(q, kernel)
        value = value.permute(0, 2, 1, 3).reshape(batch, height, width, channels)

        ret = value.permute(0, 3, 1, 2) + bias
        return self.o_proj2(self.act(self.o_proj1(ret))) + bias


def _gaussian_psf(size: int, sigma: float, dtype=torch.float64) -> torch.Tensor:
    # Use OpenCV exactly because CAVE_Dataset/Utils.para_setting does so;
    # OpenCV's discrete kernel is not bit-equivalent to the textbook formula.
    vector = cv2.getGaussianKernel(size, sigma)
    return torch.as_tensor(vector @ vector.T, dtype=dtype)


class CAVESpatialDegradation(nn.Module):
    """Exact linear DB used by CAVE_Dataset: circular blur then phase sampling."""

    def __init__(self, scale_factor: int, sigma: float = 2.0):
        super().__init__()
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        self.scale_factor = int(scale_factor)
        self.sigma = float(sigma)
        self.sample_offset = self.scale_factor // 2 - 1
        self.register_buffer("psf", _gaussian_psf(self.scale_factor, self.sigma).float())
        self._otf_cache = {}

    def _otf(self, height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        key = (height, width, reference.device, reference.dtype)
        if key not in self._otf_cache:
            padded = torch.zeros((height, width), device=reference.device, dtype=reference.dtype)
            kernel = self.psf.to(device=reference.device, dtype=reference.dtype)
            kh, kw = kernel.shape
            padded[:kh, :kw] = kernel
            padded = torch.roll(padded, shifts=(-kh // 2, -kw // 2), dims=(0, 1))
            self._otf_cache[key] = torch.fft.fft2(padded)
        return self._otf_cache[key]

    def blur(self, x: torch.Tensor) -> torch.Tensor:
        otf = self._otf(x.shape[-2], x.shape[-1], x)
        return torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1)) * otf, dim=(-2, -1)).real

    def blur_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        otf = self._otf(x.shape[-2], x.shape[-1], x)
        return torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1)) * otf.conj(), dim=(-2, -1)).real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blurred = self.blur(x)
        o, s = self.sample_offset, self.scale_factor
        return blurred[..., o::s, o::s]

    def adjoint(self, y: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        height, width = output_size
        injected = y.new_zeros((*y.shape[:-2], height, width))
        o, s = self.sample_offset, self.scale_factor
        target = injected[..., o::s, o::s]
        if target.shape != y.shape:
            raise ValueError(f"LR shape {tuple(y.shape)} incompatible with HR size {output_size}")
        injected[..., o::s, o::s] = y
        return self.blur_adjoint(injected)


class EffectiveSpectralResponse(nn.Module):
    """Fixed 3x31 effective response fitted exclusively from training scenes."""

    def __init__(self, response: torch.Tensor):
        super().__init__()
        response = torch.as_tensor(response, dtype=torch.float32)
        if response.ndim != 2 or min(response.shape) < 1 or not torch.isfinite(response).all():
            raise ValueError(f"Expected finite response [MSI channels, HSI channels], got {tuple(response.shape)}")
        self.register_buffer("response", response)

    @classmethod
    def from_file(cls, filename: str | Path) -> "EffectiveSpectralResponse":
        data = np.load(Path(filename), allow_pickle=False)
        if "response" not in data:
            raise KeyError(f"No 'response' array in {filename}")
        return cls(torch.from_numpy(data["response"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oc,bchw->bohw", self.response.to(x), x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oc,bohw->bchw", self.response.to(y), y)


class PhysicsResidualGuidance(nn.Module):
    """Back-project observation residuals and lift them to latent features."""

    def __init__(
        self,
        response: torch.Tensor,
        scale_factor: int,
        feature_channels: int = 64,
        use_spatial: bool = True,
        use_spectral: bool = True,
        sigma: float = 2.0,
    ):
        super().__init__()
        if not (use_spatial or use_spectral):
            raise ValueError("At least one residual source must be enabled")
        self.use_spatial = bool(use_spatial)
        self.use_spectral = bool(use_spectral)
        self.spatial = CAVESpatialDegradation(scale_factor, sigma=sigma)
        self.spectral = EffectiveSpectralResponse(response)
        self.lift = nn.Conv2d(self.spectral.response.shape[1], feature_channels, 1)

    def forward(
        self, estimate: torch.Tensor, lr_hsi: torch.Tensor, hr_msi: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gradient = torch.zeros_like(estimate)
        diagnostics = {}
        if self.use_spatial:
            spatial_residual = lr_hsi - self.spatial(estimate)
            gradient = gradient + self.spatial.adjoint(spatial_residual, estimate.shape[-2:])
            diagnostics["spatial_residual"] = spatial_residual
        if self.use_spectral:
            spectral_residual = hr_msi - self.spectral(estimate)
            gradient = gradient + self.spectral.adjoint(spectral_residual)
            diagnostics["spectral_residual"] = spectral_residual
        diagnostics["backprojected_residual"] = gradient
        return self.lift(gradient), diagnostics

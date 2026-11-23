"""Shared fusion backbone, physics integration and reconstruction heads."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .VectorAttention import VectorAttention
    from .PhysicsGalerkin import PhysicsGuidedGalerkin, PhysicsResidualGuidance
    from .MultiScaleDynamicGating import MultiScaleDynamicGating
else:
    from VectorAttention import VectorAttention
    from PhysicsGalerkin import PhysicsGuidedGalerkin, PhysicsResidualGuidance
    from MultiScaleDynamicGating import MultiScaleDynamicGating


class FusionIntegral(nn.Module):
    def __init__(self, in_chans=64, head=8, mdg_scales=(1, 2, 4)):
        super().__init__()
        self.MDG = MultiScaleDynamicGating(in_chans, scales=mdg_scales)
        self.GI = PhysicsGuidedGalerkin(in_chans, head)

    def forward(self, height, width, x, guidance):
        return self.GI(self.MDG(height, width, x), guidance) + x


class FusionBackbone(nn.Module):
    """Fusion features with physics-guided Galerkin integration."""

    def __init__(
        self,
        size=64,
        response_path: str | Path | None = None,
        response: torch.Tensor | np.ndarray | None = None,
        scale_factor: int = 4,
        use_spatial_residual: bool = True,
        use_spectral_residual: bool = True,
        guidance_mode: str = "physics",
        mdg_scales: tuple[int, ...] = (1, 2, 4),
        hschannels: int = 31,
        mschannels: int = 3,
    ):
        super().__init__()
        if response is None:
            if response_path is None:
                raise ValueError("FusionBackbone requires response_path or response")
            response = np.load(Path(response_path), allow_pickle=False)["response"]
        response = torch.as_tensor(response, dtype=torch.float32)

        self.hschannels = hschannels
        self.mschannels = mschannels
        channels = hschannels + mschannels
        if response.shape != (mschannels, hschannels):
            raise ValueError("Response shape does not match model channels")
        self.embed = 64
        self.img_size = size
        self.scale_factor = int(scale_factor)
        if guidance_mode not in ("physics", "learned"):
            raise ValueError("guidance_mode must be physics or learned")
        self.guidance_mode = guidance_mode

        # Preserve V1 feature extraction, physics guidance and reconstruction.
        self.conv_block = nn.Conv2d(channels, self.embed, 1)
        self.block1 = FusionIntegral(self.embed, 8, mdg_scales)
        self.block2 = FusionIntegral(self.embed, 8, mdg_scales)
        self.block3 = FusionIntegral(self.embed, 8, mdg_scales)
        self.conv = nn.Sequential(
            nn.Conv2d(self.embed * 4, self.hschannels, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.VI1_1 = VectorAttention(channels, channels)
        self.VI1_2 = VectorAttention(channels, channels)
        self.VI1_3 = VectorAttention(channels, channels)

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

    def load_v1_state_dict(self, state_dict: dict) -> tuple[list[str], list[str]]:
        """Warm-start compatible V1 parameters; report new MDG and removed PR keys.

        This is not a V1-equivalent model. MDG parameters require training.
        Shape-mismatched shared tensors raise rather than being silently skipped.
        """
        result = self.load_state_dict(state_dict, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)

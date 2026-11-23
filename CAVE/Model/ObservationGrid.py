"""Shared observation-grid backbone and regional mixing for V3 models."""
import math

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .FusionBackbone import FusionBackbone
    from .MultiScaleDynamicGating import GlobalContextDynamicGating
    from .PhysicsGalerkin import PhysicsResidualGuidance
    from .DynamicSpatial import DynamicSpatial
else:
    from FusionBackbone import FusionBackbone
    from MultiScaleDynamicGating import GlobalContextDynamicGating
    from PhysicsGalerkin import PhysicsResidualGuidance
    from DynamicSpatial import DynamicSpatial


class SpacingMDG(GlobalContextDynamicGating):
    def forward(self, height, width, x, *, lr_size, return_gates=False):
        if x.shape[1:] != (self.dim, height, width):
            raise ValueError('feature shape does not match dimensions')
        low_h, low_w = lr_size
        normalized = self.norm1(x)
        features = []
        # Target grids directly encode d=(1, s/2, s), no integer sf switch.
        sizes = [(height, width), (min(height, 2*low_h), min(width, 2*low_w)), (low_h, low_w)]
        for size, branch in zip(sizes, self.branches):
            local = normalized if size == (height, width) else F.interpolate(
                normalized, size=size, mode='bilinear', align_corners=False, antialias=True)
            local = branch(local)
            if size != (height, width):
                local = F.interpolate(local, size=(height, width), mode='bilinear', align_corners=False)
            features.append(local)
        gh, gw = min(self.grid_size, height), min(self.grid_size, width)
        pooled = F.adaptive_avg_pool2d(normalized, (gh, gw))
        tokens = pooled.flatten(2).transpose(1, 2)
        context, _ = self.global_attention(tokens, tokens, tokens, need_weights=False)
        context = context.transpose(1, 2).reshape(x.shape[0], self.dim, gh, gw)
        context = self.global_proj(context + pooled)
        features.append(F.interpolate(context, size=(height, width), mode='bilinear', align_corners=False))
        spatial = self.gate(torch.cat(features, 1)).unsqueeze(2)
        descriptors = torch.cat([f.mean((-2,-1), keepdim=True) for f in features], 1)
        channel = self.channel_gate(descriptors).reshape(x.shape[0], 4, self.dim, 1, 1)
        gates = (spatial + channel).softmax(1)
        output = x + self.proj(sum(f * gates[:, i] for i, f in enumerate(features)))
        output = output + self.ffn(self.norm2(output))
        return (output, gates) if return_gates else output


class DynamicGuidance(PhysicsResidualGuidance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial = DynamicSpatial()

    def forward(self, estimate, lr_hsi, hr_msi):
        gradient = torch.zeros_like(estimate)
        diagnostics = {}
        if self.use_spatial:
            residual = lr_hsi - self.spatial(estimate, lr_hsi.shape[-2:])
            gradient = gradient + self.spatial.adjoint(residual, estimate.shape[-2:])
            diagnostics['spatial_residual'] = residual
        if self.use_spectral:
            residual = hr_msi - self.spectral(estimate)
            gradient = gradient + self.spectral.adjoint(residual)
            diagnostics['spectral_residual'] = residual
        diagnostics['backprojected_residual'] = gradient
        return self.lift(gradient), diagnostics


class ObservationGridBackbone(FusionBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the original shared-backbone initialization sequence. These
        # temporary modules consume RNG before the spacing-aware replacements;
        # skipping them would change fresh V3 weights for the same seed.
        for block in (self.block1, self.block2, self.block3):
            block.MDG = GlobalContextDynamicGating(self.embed, scales=block.MDG.scales)
        for block in (self.block1, self.block2, self.block3):
            block.MDG = SpacingMDG(self.embed)
        old = self.physics
        self.physics = DynamicGuidance(response=old.spectral.response, scale_factor=4,
                                       feature_channels=self.embed, use_spatial=old.use_spatial,
                                       use_spectral=old.use_spectral)

    def forward(self, ms, rgb, sf=None):
        if ms.ndim != 4 or rgb.ndim != 4 or ms.shape[:2] != (rgb.shape[0], self.hschannels) or rgb.shape[1] != self.mschannels:
            raise ValueError(f'expected ms with {self.hschannels} channels and MSI with {self.mschannels} channels')
        height, width = rgb.shape[-2:]
        low_h, low_w = ms.shape[-2:]
        if min(low_h,low_w) < 1 or low_h > height or low_w > width:
            raise ValueError('LR size must be positive and no larger than HR')
        if sf is not None:
            requested = (float(sf),float(sf)) if isinstance(sf,(int,float)) else tuple(map(float,sf))
            if len(requested)!=2 or any(not math.isfinite(v) or v < 1 for v in requested):
                raise ValueError('sf must be finite and >=1')
            # Allow quantization by rounding LR dimension, but reject silent misuse.
            if max(1, round(height/requested[0])) != low_h or max(1, round(width/requested[1])) != low_w:
                raise ValueError('sf inconsistent with actual LR/HR dimensions')
        ms_up = F.interpolate(ms, size=(height,width), mode='bicubic', align_corners=False)
        xt = torch.cat((rgb,ms_up),1)
        x1 = self.conv_block(self.VI1_3(self.VI1_2(self.VI1_1(xt))))
        features = [x1]
        for block in (self.block1,self.block2,self.block3):
            feature = features[-1]
            guidance = self._guidance(feature,ms_up,ms,rgb)
            mixed = block.MDG(height,width,feature,lr_size=(low_h,low_w))
            features.append(block.GI(mixed,guidance)+feature)
        return self.conv(torch.cat(features,1))+ms_up

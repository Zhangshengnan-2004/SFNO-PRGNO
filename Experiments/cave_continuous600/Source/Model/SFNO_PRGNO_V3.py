"""V3 restores regional multiscale aggregation with gated deformable correction."""
import torch
from torch import nn

if __package__:
    from .ObservationGrid import ObservationGridBackbone, SpacingMDG
    from .SpacingDeformableSampling import SpacingDeformableSampling
else:
    from ObservationGrid import ObservationGridBackbone, SpacingMDG
    from SpacingDeformableSampling import SpacingDeformableSampling


ABLATIONS = ("full", "no_deformable", "no_regional", "fixed_gate", "fixed_grid",
             "deform_stage1", "deform_stage2", "deform_stage3")


class RegionalDeformableMixing(nn.Module):
    def __init__(self, dim, ablation="full"):
        super().__init__()
        self.ablation = ablation
        self.regional = SpacingMDG(dim)
        self.deformable = SpacingDeformableSampling(dim)
        # Small but nonzero correction gives the sampling branch gradients at start.
        self.correction_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, height, width, x, *, lr_size):
        if self.ablation == "fixed_grid":
            lr_size = (max(1, round(height / 4)), max(1, round(width / 4)))
        regional = x if self.ablation == "no_regional" else self.regional(height, width, x, lr_size=lr_size)
        if self.ablation == "no_deformable":
            return regional
        correction = self.deformable(height, width, x, lr_size=lr_size) - x
        weight = x.new_tensor(-3.0).sigmoid() if self.ablation == "fixed_gate" else self.correction_logit.sigmoid()
        return regional + weight * correction


class SFNO_PRGNO_V3(ObservationGridBackbone):
    def __init__(self, *args, ablation="full", **kwargs):
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown V3 ablation: {ablation}")
        super().__init__(*args, **kwargs)
        self.ablation = ablation
        for index, block in enumerate((self.block1, self.block2, self.block3), start=1):
            mode = ablation
            if ablation.startswith("deform_stage"):
                mode = "full" if index == int(ablation[-1]) else "no_deformable"
            block.MDG = RegionalDeformableMixing(self.embed, mode)

    def load_v21_state_dict(self, state):
        raise ValueError('V3 requires native V3 weights')

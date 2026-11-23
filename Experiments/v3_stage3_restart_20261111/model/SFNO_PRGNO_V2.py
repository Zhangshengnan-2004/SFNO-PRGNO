"""V2: multiscale regional mixing with global context and channel gates."""
if __package__:
    from .FusionBackbone import FusionBackbone
    from .MultiScaleDynamicGatingV2 import MultiScaleDynamicGatingV2
else:
    from FusionBackbone import FusionBackbone
    from MultiScaleDynamicGatingV2 import MultiScaleDynamicGatingV2


class SFNO_PRGNO_V2(FusionBackbone):
    """Regional mixing with shared physics and reconstruction heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in (self.block1, self.block2, self.block3):
            block.MDG = MultiScaleDynamicGatingV2(self.embed, scales=block.MDG.scales)

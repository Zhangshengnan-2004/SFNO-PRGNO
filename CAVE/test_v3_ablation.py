"""CPU checks of ablation semantics, cross-scale gradients and default compatibility."""
import torch
from Model.SFNO_PRGNO_V3 import SFNO_PRGNO_V3, RegionalDeformableMixing, ABLATIONS


def main():
    torch.set_num_threads(2)
    response = torch.ones(3,31)/31
    torch.manual_seed(42)
    default = SFNO_PRGNO_V3(response=response)
    torch.manual_seed(42)
    full = SFNO_PRGNO_V3(response=response, ablation='full')
    for k,v in default.state_dict().items():
        torch.testing.assert_close(v,full.state_dict()[k],rtol=0,atol=0)
    x = torch.rand(1,64,16,16)
    for variant in ABLATIONS:
        mix = RegionalDeformableMixing(64,variant)
        actual = mix(16,16,x,lr_size=(2,2))
        grid = (4,4) if variant == 'fixed_grid' else (2,2)
        reg = x if variant == 'no_regional' else mix.regional(16,16,x,lr_size=grid)
        expected = reg if variant == 'no_deformable' else reg + torch.sigmoid(torch.tensor(-3.))*(mix.deformable(16,16,x,lr_size=grid)-x)
        torch.testing.assert_close(actual,expected)
        for sf in (4,8):
            model = SFNO_PRGNO_V3(response=response,ablation=variant)
            result = model(torch.rand(1,31,16//sf,16//sf),torch.rand(1,3,16,16),sf)
            result.square().mean().backward()
            assert result.shape == (1,31,16,16) and torch.isfinite(result).all()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            for index, stage in enumerate((model.block1, model.block2, model.block3), start=1):
                block = stage.MDG
                mode = variant
                if variant.startswith('deform_stage'):
                    mode = 'full' if index == int(variant[-1]) else 'no_deformable'
                assert block.ablation == mode
                assert (block.correction_logit.grad is None) == (mode in ('fixed_gate','no_deformable'))
                assert (block.deformable.offset.weight.grad is None) == (mode == 'no_deformable')
                assert (block.regional.branches[0][0].weight.grad is None) == (mode == 'no_regional')
    for mode,spatial,spectral in [('physics',True,False),('physics',False,True),('learned',True,True)]:
        model = SFNO_PRGNO_V3(response=response,guidance_mode=mode,use_spatial_residual=spatial,use_spectral_residual=spectral)
        model(torch.rand(1,31,4,4),torch.rand(1,3,16,16),4).mean().backward()
    print('PASS: default weights, all variant formulas, x4/x8 forward/backward, disabled-branch gradients, guidance modes')


if __name__ == '__main__': main()

"""V3 sampling geometry, cross-grid gradients, optimizer and serialization checks."""
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.SpacingDeformableSampling import SpacingDeformableSampling
from Model.SFNO_PRGNO_V3 import SFNO_PRGNO_V3


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    module = SpacingDeformableSampling(8)
    x = torch.randn(1,8,9,15)
    grids = []
    original = F.grid_sample
    def capture(value, grid, **kwargs):
        grids.append(grid.detach())
        return original(value,grid,**kwargs)
    with patch.object(F,'grid_sample',side_effect=capture):
        module(9,15,x,lr_size=(3,3))
    # Zero-initialized learned offsets: adjacent lattice points are 5 HR x
    # pixels / 3 HR y pixels apart; center is exactly the pixel center.
    torch.testing.assert_close(grids[5]-grids[4],torch.tensor([2*5/15,0.]).expand_as(grids[4]))
    torch.testing.assert_close(grids[7]-grids[4],torch.tensor([0.,2*3/9]).expand_as(grids[4]))
    torch.testing.assert_close(original(x,grids[4],align_corners=False,padding_mode='border'),x,atol=3e-6,rtol=1e-5)
    response=torch.rand(3,31); response/=response.sum(1,keepdim=True)
    model=SFNO_PRGNO_V3(response=response)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    parameter_ids = tuple(map(id,model.parameters()))
    for hr,lr in [((24,32),(6,8)),((24,32),(3,4)),((21,29),(8,11)),((21,29),(5,8)),((7,9),(7,9)),((8,10),(1,1))]:
        optimizer.zero_grad(set_to_none=True)
        ms=torch.rand(1,31,*lr); rgb=torch.rand(1,3,*hr)
        output=model(ms,rgb)
        assert output.shape==(1,31,*hr) and torch.isfinite(output).all()
        (output-torch.rand_like(output)).square().mean().backward()
        for block in (model.block1,model.block2,model.block3):
            for layer in (block.MDG.deformable.offset,block.MDG.deformable.weight,block.MDG.deformable.local,block.GI.delta_qk):
                g=layer.weight.grad
                assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
        for block in (model.block1,model.block2,model.block3):
            assert block.MDG.correction_logit.grad is not None
            assert block.MDG.correction_logit.grad.abs()>0
            for branch in block.MDG.regional.branches:
                assert branch[0].weight.grad.abs().sum()>0
        optimizer.step()
        assert tuple(map(id,model.parameters()))==parameter_ids
    model.eval()
    restored=SFNO_PRGNO_V3(response=response).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(ms,rgb),restored(ms,rgb),atol=0,rtol=0)
    try: model(ms,rgb,4)
    except ValueError: pass
    else: raise AssertionError('mismatched sf accepted')
    print('PASS: sampling geometry, integer/noninteger/anisotropic/unit/tiny grids, gradients, optimizer step, state roundtrip, sf validation')
    print('Parameters:',sum(p.numel() for p in model.parameters()))
    if torch.cuda.is_available():
        model=model.cuda().train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss=model(ms.cuda(),rgb.cuda()).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        print('PASS: CUDA autocast forward/backward')


if __name__=='__main__': main()

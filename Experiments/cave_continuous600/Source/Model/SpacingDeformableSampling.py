"""Observation-spacing-conditioned sparse spatial mixing (no custom CUDA ops)."""
import torch
from torch import nn
from torch.nn import functional as F


class SpacingDeformableSampling(nn.Module):
    """Nine samples at HR offsets (base + 0.5*tanh(delta))*HR/LR.

    Axis order for grid_sample is (x,y). Border replication and align_corners=False
    are explicit. Samples are accumulated one at a time to bound activation size.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.norm = nn.GroupNorm(1, dim)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.context = nn.Sequential(nn.Conv2d(dim + 2, dim, 3, padding=1), nn.GELU())
        self.offset = nn.Conv2d(dim, 18, 1)
        self.weight = nn.Conv2d(dim, 9, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)
        self.proj = nn.Conv2d(2 * dim, dim, 1)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = nn.Sequential(nn.Conv2d(dim, 2*dim, 1), nn.GELU(), nn.Conv2d(2*dim, dim, 1))
        self.register_buffer('base', torch.tensor([(x,y) for y in (-1.,0.,1.) for x in (-1.,0.,1.)]))

    def forward(self, height, width, x, *, lr_size):
        if x.ndim != 4 or x.shape[1:] != (self.dim, height, width):
            raise ValueError('feature shape does not match dimensions')
        lh, lw = lr_size
        if min(lh, lw) < 1 or lh > height or lw > width:
            raise ValueError('invalid LR size')
        z = self.norm(x)
        # log-spacing also conditions the predictor; no nominal integer sf switch.
        spacing = x.new_tensor([width/lw, height/lh], dtype=torch.float32)
        cond = spacing.log().to(z.dtype).view(1,2,1,1).expand(x.shape[0],2,height,width)
        context = self.context(torch.cat((z,cond),1))
        offsets = self.offset(context).float().reshape(x.shape[0],9,2,height,width)
        weights = self.weight(context).float().softmax(1)
        # FP32 sampling supports CPU and CUDA autocast without half-grid rounding.
        with torch.autocast(device_type=x.device.type, enabled=False):
            yy, xx = torch.meshgrid(torch.arange(height,device=x.device,dtype=torch.float32),
                                    torch.arange(width,device=x.device,dtype=torch.float32), indexing='ij')
            origin = torch.stack((xx,yy),-1)[None]
            normalizer = spacing.new_tensor([width,height])
            mixed = torch.zeros_like(z, dtype=torch.float32)
            for i in range(9):
                delta = self.base[i].float().view(1,1,1,2) + 0.5*offsets[:,i].tanh().permute(0,2,3,1)
                grid = 2*(origin + delta*spacing + 0.5)/normalizer - 1
                sampled = F.grid_sample(z.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)
                mixed = mixed + sampled*weights[:,i:i+1]
        output = x + self.proj(torch.cat((self.local(z),mixed.to(z.dtype)),1))
        return output + self.ffn(self.norm2(output))

"""Exact row-chunked local vector attention for full-image evaluation on 8GB GPUs."""
import types
import torch
import torch.nn.functional as F
from VectorAttention import VectorAttention


def chunked_forward(self, x):
    if self.training or x.shape[-2] <= 128:
        return self._original_forward(x)
    b,c,h,w=x.shape
    q,k,v=self.qkv_conv(x).chunk(3,dim=1)
    k,v=F.pad(k,(1,1,1,1)),F.pad(v,(1,1,1,1))
    output=torch.empty_like(x)
    for start in range(0,h,32):
        end=min(start+32,h)
        rows=end-start
        ku=F.unfold(k[:,:,start:end+2],3).view(b,c,9,rows,w)
        vu=F.unfold(v[:,:,start:end+2],3).view(b,c,9,rows,w)
        delta=(q[:,:,start:end].unsqueeze(2)-ku).permute(0,3,4,2,1).contiguous()
        scores=F.softmax(self.mlp(delta),dim=-2)
        weighted=(vu.permute(0,3,4,2,1)*scores).sum(dim=3).permute(0,3,1,2)
        output[:,:,start:end]=weighted
    return output+self.gate(x)+x


def configure(model):
    for layer in model.modules():
        if isinstance(layer,VectorAttention) and not hasattr(layer,'_original_forward'):
            layer._original_forward=layer.forward
            layer.forward=types.MethodType(chunked_forward,layer)
    return model

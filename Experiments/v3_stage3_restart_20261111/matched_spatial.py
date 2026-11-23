"""FFT forward/adjoint matching the frozen periodic sigma2 sf-sized degradation."""
import torch


class MatchedSpatial(torch.nn.Module):
    def __init__(self):
        super().__init__();self.cache={}

    def geometry(self,size,low_size,reference):
        h,w=size;sh,sw=h//low_size[0],w//low_size[1]
        if h%low_size[0] or w%low_size[1] or sh!=sw or sh not in (4,8):
            raise ValueError('Expected integer x4/x8 observation geometry')
        key=(h,w,sh,reference.device,reference.dtype)
        if key not in self.cache:
            # Health probes may warm the cache in inference mode. Autograd must
            # still be able to save this fixed multiplier on later training calls.
            with torch.inference_mode(False),torch.no_grad():
                coord=torch.arange(sh,device=reference.device,dtype=reference.dtype)-(sh-1)/2
                g=torch.exp(-coord.square()/8);kernel=g[:,None]*g[None,:];kernel/=kernel.sum()
                padded=reference.new_zeros(h,w);padded[:sh,:sh]=kernel
                otf=torch.fft.fft2(torch.roll(padded,(-sh//2,-sh//2),(0,1)))
            if len(self.cache)>=8:self.cache.clear()
            self.cache[key]=otf
        return self.cache[key],sh,sh//2-1

    def forward(self,x,target_size):
        otf,sf,phase=self.geometry(x.shape[-2:],target_size,x)
        return torch.fft.ifft2(torch.fft.fft2(x)*otf).real[...,phase::sf,phase::sf]

    def adjoint(self,y,output_size):
        otf,sf,phase=self.geometry(output_size,y.shape[-2:],y)
        injected=y.new_zeros(*y.shape[:-2],*output_size);injected[...,phase::sf,phase::sf]=y
        return torch.fft.ifft2(torch.fft.fft2(injected)*otf.conj()).real

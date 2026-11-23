"""Matched blur/sampling and its explicit adjoint for arbitrary LR shapes."""
import cv2
import torch
from torch import nn


class DynamicSpatial(nn.Module):
    """Circular Gaussian blur followed by periodic bilinear observation sampling.

    Sampling coordinate is (j+1/2)*H/h-1, extending the legacy even-integer phase.
    For a unit ratio, use identity coordinates. PSF support is rounded from each
    actual axis ratio, sigma=2; this is a specified synthetic degradation model,
    not an inferred or measured sensor PSF.
    """

    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma
        self._cache = {}

    def _geometry(self, hr, lr, reference):
        height, width = map(int, hr)
        low_h, low_w = map(int, lr)
        if min(height, width, low_h, low_w) < 1 or low_h > height or low_w > width:
            raise ValueError('require positive LR dimensions no larger than HR')
        key = (height, width, low_h, low_w, reference.device, reference.dtype)
        if key in self._cache:
            return self._cache[key]
        sh, sw = height / low_h, width / low_w
        kh, kw = max(1, round(sh)), max(1, round(sw))
        kernel = cv2.getGaussianKernel(kh, self.sigma) @ cv2.getGaussianKernel(kw, self.sigma).T
        kernel = torch.as_tensor(kernel, dtype=torch.float32, device=reference.device).to(reference.dtype)
        padded = reference.new_zeros((height, width))
        padded[:kh, :kw] = kernel
        padded = torch.roll(padded, (-(kh // 2), -(kw // 2)), (0, 1))
        otf = torch.fft.fft2(padded)
        y = torch.arange(low_h, dtype=reference.dtype, device=reference.device)
        x = torch.arange(low_w, dtype=reference.dtype, device=reference.device)
        y = y if sh == 1 else (y + .5) * sh - 1
        x = x if sw == 1 else (x + .5) * sw - 1
        y0, x0 = y.floor().long(), x.floor().long()
        fy, fx = y - y0, x - x0
        entries = []
        for dy, wy in ((0, 1-fy), (1, fy)):
            for dx, wx in ((0, 1-fx), (1, fx)):
                indices = (((y0+dy) % height)[:, None] * width + ((x0+dx) % width)[None]).flatten()
                weight = (wy[:, None] * wx[None]).flatten()
                entries.append((indices, weight))
        if len(self._cache) >= 16:
            self._cache.clear()
        self._cache[key] = (otf, entries)
        return otf, entries

    def forward(self, x, target_size):
        otf, entries = self._geometry(x.shape[-2:], target_size, x)
        blurred = torch.fft.ifft2(torch.fft.fft2(x) * otf).real.flatten(-2)
        sampled = sum(blurred[..., index] * weight for index, weight in entries)
        return sampled.reshape(*x.shape[:-2], *target_size)

    def adjoint(self, y, output_size):
        otf, entries = self._geometry(output_size, y.shape[-2:], y)
        flat = y.flatten(-2)
        injected = y.new_zeros((*y.shape[:-2], output_size[0] * output_size[1]))
        for index, weight in entries:
            index = index.expand(*flat.shape[:-1], -1)
            injected.scatter_add_(-1, index, flat * weight)
        injected = injected.reshape(*y.shape[:-2], *output_size)
        return torch.fft.ifft2(torch.fft.fft2(injected) * otf.conj()).real

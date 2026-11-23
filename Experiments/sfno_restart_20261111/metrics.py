"""Common, explicitly versioned metrics for normalized B,C,H,W hyperspectral images."""
import torch

PROTOCOL = 'common_hsi_eval_v1'
RULES = dict(peak=1.0, psnr='mean of per-band PSNR; MSE floor 1e-12',
             mask='sensor-valid AND GT L2 norm > 1e-8; independent of prediction',
             sam='degrees; zero prediction at valid GT => 90 degrees',
             ergas='100/s * sqrt(mean(MSE_band / max(mean_GT_band^2,1e-12)))',
             prediction='clamp [0,1]; unclipped results also reported',
             aggregation='arithmetic mean of patch metrics; not training-seed std')


def evaluate(pred, gt, sf, sensor_valid=None, clip=True):
    if pred.shape != gt.shape or pred.ndim != 4 or gt.shape[0] != 1:
        raise ValueError('Expected matching single-image BCHW tensors')
    if sf <= 0 or not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError('Invalid scale or non-finite input')
    if gt.min() < 0 or gt.max() > 1:
        raise ValueError('GT must use the recorded [0,1] normalization')
    p, g = pred.double().flatten(2), gt.double().flatten(2)
    keep = g.norm(dim=1)[0] > 1e-8
    if sensor_valid is not None:
        mask = torch.as_tensor(sensor_valid, device=gt.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(gt.shape[-2:]):
            raise ValueError('Sensor mask shape mismatch')
        keep &= mask.flatten()
    if not keep.any():
        raise ValueError('No valid GT pixels')
    p, g = p[:,:,keep], g[:,:,keep]
    if clip:
        p = p.clamp(0,1)
    mse = (p-g).square().mean(dim=(0,2))
    psnr = (-10*mse.clamp_min(1e-12).log10()).mean()
    pn, gn = p.norm(dim=1), g.norm(dim=1)
    cosine = ((p*g).sum(dim=1)/(pn*gn).clamp_min(1e-30)).clamp(-1,1)
    cosine = torch.where(pn>1e-12, cosine, torch.zeros_like(cosine))
    sam = torch.rad2deg(cosine.acos()).mean()
    mean_squared = g.mean(dim=(0,2)).square()
    ergas = 100/sf*(mse/mean_squared.clamp_min(1e-12)).mean().sqrt()
    return dict(psnr=float(psnr), sam_degrees=float(sam), ergas=float(ergas))

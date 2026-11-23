"""Pavia synthetic-MSI baseline; protocol is specified in README.md."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / 'CAVE'), str(ROOT.parent / 'CAVE/Model')]
from SFNO import SFNO
from test_result_log import write_result_log


def response():
    # Artificial Gaussian filters on band indices, not a calibrated camera SRF.
    bands = np.arange(102)
    r = np.exp(-0.5 * ((bands[None, :] - np.array([20, 50, 80])[:, None]) / 12.) ** 2)
    return (r / r.sum(axis=1, keepdims=True)).astype('float32')


def load_cube(path):
    cube = sio.loadmat(path)['pavia']
    if cube.shape != (1096, 715, 102):
        raise ValueError(f'Expected (1096,715,102), got {cube.shape}')
    if not np.isfinite(cube).all() or cube.min() < 0 or cube.max() > 8000:
        raise ValueError('Expected finite data in [0,8000]')
    return cube.astype('float32') / 8000.


def degrade(x, sf):
    # Periodic convolution, half-pixel-centered even kernel, explicit sampling phase.
    coord = torch.arange(sf, device=x.device, dtype=x.dtype) - (sf - 1) / 2
    g = torch.exp(-coord.square() / 8)
    kernel = g[:, None] * g[None, :]
    kernel /= kernel.sum()
    left, right = sf // 2 - 1, sf // 2
    padded = F.pad(x, (left, right, left, right), mode='circular')
    blurred = F.conv2d(padded, kernel[None, None].expand(x.shape[1], 1, sf, sf), groups=x.shape[1])
    phase = sf // 2 - 1
    return blurred[:, :, phase::sf, phase::sf]


class Patches(Dataset):
    def __init__(self, cube, samples):
        self.cube, self.samples = cube[:704], samples

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        y = random.randrange(704 - 64 + 1)
        x = random.randrange(715 - 64 + 1)
        patch = np.rot90(self.cube[y:y+64, x:x+64], random.randrange(4))
        if random.random() < .5:
            patch = patch[::-1]
        if random.random() < .5:
            patch = patch[:, ::-1]
        return torch.from_numpy(patch.copy()).permute(2, 0, 1)


class PaviaSpatial(torch.nn.Module):
    """Exact FFT form and adjoint of this protocol's even integer degradation."""
    def __init__(self):
        super().__init__()
        self.cache = {}

    def geometry(self, size, low_size, reference):
        h, w = size
        sh, sw = h // low_size[0], w // low_size[1]
        if h % low_size[0] or w % low_size[1] or sh != sw or sh not in (4, 8):
            raise ValueError('Pavia protocol supports integer x4/x8 observations')
        key = (h, w, sh, reference.device, reference.dtype)
        if key not in self.cache:
            coord = torch.arange(sh, device=reference.device, dtype=reference.dtype) - (sh-1)/2
            g = torch.exp(-coord.square()/8)
            kernel = g[:,None]*g[None,:]
            kernel /= kernel.sum()
            padded = reference.new_zeros(h, w)
            padded[:sh,:sh] = kernel
            otf = torch.fft.fft2(torch.roll(padded, (-sh//2, -sh//2), (0,1)))
            if len(self.cache) >= 8:
                self.cache.clear()
            self.cache[key] = otf
        return self.cache[key], sh, sh//2-1

    def forward(self, x, target_size):
        otf, sf, phase = self.geometry(x.shape[-2:], target_size, x)
        blurred = torch.fft.ifft2(torch.fft.fft2(x)*otf).real
        return blurred[...,phase::sf,phase::sf]

    def adjoint(self, y, output_size):
        otf, sf, phase = self.geometry(output_size, y.shape[-2:], y)
        injected = y.new_zeros(*y.shape[:-2], *output_size)
        injected[...,phase::sf,phase::sf] = y
        return torch.fft.ifft2(torch.fft.fft2(injected)*otf.conj()).real


def inputs(gt, sf, r):
    return degrade(gt, sf), torch.einsum('mc,bchw->bmhw', r, gt)


def metrics(pred, gt, sf):
    p, g = pred.double().flatten(2), gt.double().flatten(2)
    mse = (p-g).square().mean(dim=(0, 2))
    psnr = (-10 * mse.clamp_min(1e-12).log10()).mean()
    denom = p.norm(dim=1) * g.norm(dim=1)
    valid = denom > 1e-12
    cos = ((p*g).sum(dim=1) / denom.clamp_min(1e-12)).clamp(-1, 1)
    sam = torch.rad2deg(cos[valid].acos()).mean()
    ergas = 100 / sf * (mse / g.mean(dim=(0, 2)).square().clamp_min(1e-12)).mean().sqrt()
    return dict(psnr=float(psnr), sam_degrees=float(sam), ergas=float(ergas))


def format_test_report(report):
    lines = [f"Loading checkpoint: {report['checkpoint']}", '模型权重加载成功',
             f"训练轮数: {report['epoch']}"]
    patches = report['patches']
    for i, patch in enumerate(patches, 1):
        lines.append(f"[{i}/{len(patches)}] PSNR={patch['psnr']:.4f} "
                     f"SAM={patch['sam_degrees']:.4f} ERGAS={patch['ergas']:.4f}")
    mean = report['mean']
    lines.extend(['', '========== Test Results ==========',
                  f"Scale: {report['sf']}", f'Test Images: {len(patches)}',
                  f"Avg PSNR = {mean['psnr']:.4f}",
                  f"Avg SAM = {mean['sam_degrees']:.4f}",
                  f"Avg ERGAS = {mean['ergas']:.4f}",
                  '=================================='])
    return '\n'.join(lines)


def training_lr(initial_lr, completed_epochs):
    return initial_lr * .5 ** sum(completed_epochs >= e for e in (200, 300, 350, 500, 550))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['smoke', 'train', 'test'])
    parser.add_argument('--model', choices=['sfno', 'v3'], default='sfno')
    parser.add_argument('--ablation', default='full', choices=['full', 'deform_stage3'])
    parser.add_argument('--data', type=Path, default=ROOT/'Data/Pavia.mat')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--sf', type=int, choices=[4, 8], default=4)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output is None:
        name = 'sfno_continuous600_x4_seed42' if args.model == 'sfno' else f'v3_{args.ablation}_continuous600_x4_seed42'
        args.output = ROOT/'Checkpoint'/name
    if min(args.epochs, args.samples, args.batch_size, args.lr, args.threads, args.save_every) <= 0:
        parser.error('Training sizes, learning rate and threads must be positive')
    if args.mode == 'test' and args.checkpoint is None:
        parser.error('test requires --checkpoint')
    if args.resume and args.mode != 'train':
        parser.error('--resume requires train')
    if args.checkpoint and args.mode != 'test':
        parser.error('--checkpoint is only for test; training resumes with --resume')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cube = load_cube(args.data)
    r = torch.from_numpy(response()).to(device)
    if args.model == 'v3':
        from SFNO_PRGNO_V3 import SFNO_PRGNO_V3
        model = SFNO_PRGNO_V3(size=64, hschannels=102, mschannels=3,
                              response=response(), ablation=args.ablation).to(device)
        model.physics.spatial = PaviaSpatial()
    else:
        model = SFNO(size=64, hschannels=102, mschannels=3).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['protocol'] != 'pavia_synthetic_v1':
            raise ValueError('Incompatible checkpoint protocol')
        if checkpoint['config'].get('model', 'sfno') != args.model or (args.model == 'v3' and checkpoint['config'].get('ablation') != args.ablation):
            raise ValueError('Checkpoint model/ablation mismatch')
        model.load_state_dict(checkpoint['model'])
    if args.mode == 'test':
        model.eval()
        results = []
        with torch.no_grad():
            for y, x in [(712, 0), (712, 192), (904, 0), (904, 192)]:
                gt = torch.from_numpy(cube[y:y+192, x:x+192].copy()).permute(2, 0, 1)[None].to(device)
                low, msi = inputs(gt, args.sf, r)
                pred = model(low, msi, args.sf)
                if not torch.isfinite(pred).all():
                    raise RuntimeError('Non-finite prediction')
                results.append(dict(y=y, x=x, **metrics(pred, gt, args.sf)))
        report = dict(protocol='pavia_synthetic_v1', checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint['epoch'],
                      source_sf=checkpoint['config']['sf'], sf=args.sf, model=args.model, ablation=args.ablation,
                      patches=results, mean={k: float(np.mean([v[k] for v in results])) for k in ['psnr', 'sam_degrees', 'ergas']})
        # Exclusive creation keeps earlier reports safe.
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output/f'{args.checkpoint.stem}_test_x{args.sf}.json').open('x') as f:
            json.dump(report, f, indent=2)
        write_result_log(report, directory=args.output)
        print(format_test_report(report), flush=True)
        return
    if args.mode == 'train' and not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.output}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(Patches(cube, args.samples if args.mode == 'train' else 1),
                        batch_size=args.batch_size if args.mode == 'train' else 1)
    start_epoch = 0
    if args.mode == 'train':
        args.output.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(protocol='pavia_synthetic_v1', data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                      torch_version=torch.__version__, numpy_version=np.__version__)
        config['schedule'] = 'cave_continuous600_v1'
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if checkpoint.get('config', {}).get('schedule') != config['schedule']:
                raise ValueError('Resume requires the aligned schedule; old 0.95 checkpoints cannot be continued as a fair baseline')
            for key in ('data_sha256', 'sf', 'samples', 'batch_size', 'lr', 'seed', 'model', 'ablation'):
                if checkpoint['config'].get(key, 'sfno' if key == 'model' else 'full' if key == 'ablation' else None) != config[key]:
                    raise ValueError(f'Resume config mismatch: {key}')
            if args.resume.resolve().parent != args.output.resolve():
                raise ValueError('--output must be the checkpoint directory when resuming')
            start_epoch = checkpoint['epoch']
            if start_epoch >= args.epochs:
                raise ValueError('Checkpoint already reached requested epochs')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            random.setstate(checkpoint['rng_python'])
            np.random.set_state(checkpoint['rng_numpy'])
            torch.set_rng_state(checkpoint['rng_torch'])
            if checkpoint['rng_cuda'] is not None:
                if device.type != 'cuda':
                    raise ValueError('Resume this CUDA run on CUDA to restore RNG state')
                torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
            # Drop CSV rows beyond the last atomic checkpoint after an interrupted save.
            log_path = args.output/'train.csv'
            if log_path.exists():
                rows = list(csv.reader(log_path.open()))
                with log_path.open('w', newline='') as f:
                    csv.writer(f).writerows(row for row in rows if int(row[0]) <= start_epoch)
            (args.output/f'resume_epoch{start_epoch}.json').write_text(json.dumps(config, indent=2))
        else:
            (args.output/'config.json').write_text(json.dumps(config, indent=2))
            np.save(args.output/'response.npy', response())
    for epoch in range(start_epoch, args.epochs if args.mode == 'train' else 1):
        lr = training_lr(args.lr, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        total, count = 0., 0
        for gt in loader:
            gt = gt.to(device)
            low, msi = inputs(gt, args.sf, r)
            optimizer.zero_grad(set_to_none=True)
            pred = model(low, msi, args.sf)
            if pred.shape != gt.shape:
                raise RuntimeError(f'Unexpected output shape {pred.shape}')
            loss = F.l1_loss(pred, gt)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Non-finite gradients')
            optimizer.step()
            total += float(loss.detach()) * len(gt)
            count += len(gt)
        print(f'epoch={epoch+1} lr={optimizer.param_groups[0]["lr"]:.8g} L1={total/count:.6f} shape={tuple(pred.shape)}', flush=True)
        if args.mode == 'train':
            with (args.output/'train.csv').open('a', newline='') as f:
                csv.writer(f).writerow([epoch+1, optimizer.param_groups[0]['lr'], total/count])
            target = args.output/'last.pth'
            torch.save(dict(protocol='pavia_synthetic_v1', epoch=epoch+1, model=model.state_dict(),
                            optimizer=optimizer.state_dict(), config=config,
                            rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                            rng_torch=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None),
                       target.with_suffix('.tmp'))
            target.with_suffix('.tmp').replace(target)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                import shutil
                shutil.copyfile(target, args.output/f'epoch_{epoch+1}.pth')


if __name__ == '__main__':
    main()

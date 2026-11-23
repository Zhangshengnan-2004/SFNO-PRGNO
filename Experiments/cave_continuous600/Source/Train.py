"""SFNO CAVE training (community reconstruction, not the authors' released training script).
Run from any directory: python -u Train.py --data_path ./Data/Train --epochs 100 ...
"""
import argparse
from continuation_schedule import add_arguments, validate, apply
import csv
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy_compat import install as install_numpy_compat
install_numpy_compat()
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Model'))
from SFNO import SFNO
from SFNO_PRGNO import SFNO_PRGNO
from CAVE_Dataset import cave_dataset
from Utils import loadpath, prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train SFNO on CAVE from scratch or resume.')
    parser.add_argument('--data_path', type=Path, required=True, help='Directory with Train.txt, HSI/, RGB/')
    parser.add_argument('--epochs', type=int, required=True, help='Total training epochs, including resumed epochs')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--patch_size', type=int, required=True, help='HR training patch size')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--sf', type=int, default=4, help='Training scale factor')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA logical GPU index')
    parser.add_argument('--trainset_num', type=int, default=1000, help='Random patches per epoch')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--save_dir', type=Path, default=None,
                        help='Defaults to model-specific Checkpoint/train_x{sf}[_prgno]')
    parser.add_argument('--resume', type=Path, default=None, help='Existing training checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true', help='Enable FP16 autocast and GradScaler')
    parser.add_argument('--model', choices=('sfno', 'sfno_prgno'), default='sfno')
    parser.add_argument('--response', type=Path, default=ROOT / 'physics' / 'cave_effective_response.npz',
                        help='Training-set fitted effective spectral response for SFNO-PRGNO')
    parser.add_argument("--guidance-mode", choices=("physics", "learned"), default="physics")
    parser.add_argument('--disable-spatial-residual', action='store_true')
    parser.add_argument('--disable-spectral-residual', action='store_true')
    parser.add_argument('--init-sfno', type=Path, default=None,
                        help='Initialize compatible SFNO-PRGNO weights from an SFNO checkpoint')
    parser.add_argument("--ablation-lr-schedule", action="store_true",
                        help="Continuous schedule matching V3 balanced-scale experiments")
    add_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    validate(args)
    if args.resume and args.init_sfno:
        raise ValueError('Use either --resume or --init-sfno, not both')
    if min(args.epochs, args.batch_size, args.patch_size, args.sf,
           args.trainset_num, args.save_every) <= 0:
        raise ValueError('epochs/batch_size/patch_size/sf/trainset_num/save_every must be positive')
    if args.patch_size % args.sf != 0 or args.patch_size % 4 != 0:
        raise ValueError('patch_size must be divisible by sf and 4')
    if args.lr <= 0:
        raise ValueError('lr must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; install GPU-enabled PyTorch')
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f'--gpu {args.gpu} invalid; {torch.cuda.device_count()} CUDA devices visible')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_path.expanduser().resolve()
    train_txt = data_dir / 'Train.txt'
    if not train_txt.is_file():
        raise FileNotFoundError(f'Missing {train_txt}')
    names = loadpath(str(train_txt), shuffle=False)
    if len(names) != 20:
        raise ValueError(f'Original CAVE dataset assumes 20 training scenes, found {len(names)}')
    for name in names:
        for sub in ('HSI', 'RGB'):
            file = data_dir / sub / f'{name}.mat'
            if not file.is_file():
                raise FileNotFoundError(f'Missing {file}')

    print(f'GPU: {device}, {torch.cuda.get_device_name(device)}', flush=True)
    print('Loading CAVE training data...', flush=True)
    hr_hsi, hr_msi = prepare_data(str(data_dir), names, len(names))
    opt = SimpleNamespace(data_path=str(data_dir), sf=args.sf,
                          trainset_num=args.trainset_num, sizeI=args.patch_size)
    dataset = cave_dataset(opt, hr_hsi, hr_msi, istrain=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    if args.model == 'sfno':
        if args.init_sfno:
            raise ValueError('--init-sfno is only valid with --model sfno_prgno')
        model = SFNO(size=args.patch_size).to(device)
    else:
        if args.disable_spatial_residual and args.disable_spectral_residual:
            raise ValueError('Cannot disable both spatial and spectral residuals')
        model = SFNO_PRGNO(
            size=args.patch_size,
            response_path=args.response,
            scale_factor=args.sf,
            use_spatial_residual=not args.disable_spatial_residual,
            use_spectral_residual=not args.disable_spectral_residual,
            guidance_mode=args.guidance_mode,
        ).to(device)
        if args.init_sfno:
            source = torch.load(args.init_sfno, map_location='cpu', weights_only=False)
            missing, unexpected = model.load_sfno_state_dict(source.get('model', source))
            if unexpected:
                raise RuntimeError(f'Unexpected SFNO checkpoint keys: {unexpected}')
            print(f'Initialized shared SFNO weights from {args.init_sfno}; new keys: {missing}', flush=True)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    if args.save_dir is None:
        folder = f'train_x{args.sf}' if args.model == 'sfno' else f'train_x{args.sf}_prgno'
        save_dir = (ROOT / 'Checkpoint' / folder).resolve()
    else:
        save_dir = args.save_dir.expanduser().resolve()
    if not args.resume and save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty experiment: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint['epoch'])
        best_loss = float(checkpoint.get('best_train_loss', float('inf')))
        if args.amp and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f'Resumed from epoch {start_epoch}', flush=True)
    if start_epoch >= args.epochs:
        raise ValueError(f'Checkpoint epoch {start_epoch} >= requested total epochs {args.epochs}')

    def update_lr(completed_epochs):
        if not args.ablation_lr_schedule:
            return apply(optimizer, args, completed_epochs)
        lr = args.lr * 0.5 ** sum(completed_epochs >= e for e in (200, 300, 350, 500, 550))
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr

    update_lr(start_epoch)

    csv_path = save_dir / 'train_log.csv'
    if start_epoch == 0 or not csv_path.exists():
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_l1'])

    def save(filename, epoch):
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_train_loss': best_loss,
            'config': vars(args) | {'data_path': str(args.data_path),
                                   'save_dir': str(save_dir),
                                   'resume': str(args.resume) if args.resume else None},
        }, save_dir / filename)

    print(f'Training x{args.sf}, epochs={args.epochs}, patch={args.patch_size}, '
          f'batch={args.batch_size}, patches/epoch={args.trainset_num}, AMP={args.amp}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        current_lr = update_lr(epoch)
        print(f'Epoch {epoch + 1} learning rate={current_lr:.8g}', flush=True)
        model.train()
        total_loss = 0.0
        total_samples = 0
        progress = tqdm(loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for lr_hsi, rgb, gt in progress:
            lr_hsi = lr_hsi.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=args.amp):
                output = model(lr_hsi, rgb, args.sf)
                if output.shape != gt.shape:
                    raise RuntimeError(f'Output {tuple(output.shape)} != GT {tuple(gt.shape)}')
                loss = criterion(output, gt)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite loss at epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = lr_hsi.size(0)
            total_loss += loss.item() * n
            total_samples += n
            progress.set_postfix(loss=f'{loss.item():.6f}')
        avg_loss = total_loss / total_samples
        print(f'Epoch {epoch + 1}/{args.epochs} | train L1={avg_loss:.6f}', flush=True)
        with csv_path.open('a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{avg_loss:.8f}'])
        if avg_loss < best_loss:
            best_loss = avg_loss
            save('best_train_loss.pth', epoch + 1)
        save('last.pth', epoch + 1)
        if (epoch + 1) % args.save_every == 0:
            save(f'epoch_{epoch + 1:04d}.pth', epoch + 1)
    print('Finished. Lowest training loss is not necessarily best validation PSNR.', flush=True)


if __name__ == '__main__':
    main()

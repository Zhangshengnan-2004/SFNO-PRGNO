"""Generate or execute isolated ablations; defaults remain V1, dry-run."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'baseline': ['--model', 'sfno'],
    'spatial': ['--model', 'sfno_prgno', '--disable-spectral-residual'],
    'spectral': ['--model', 'sfno_prgno', '--disable-spatial-residual'],
    'dual': ['--model', 'sfno_prgno'],
    'capacity': ['--model', 'sfno_prgno', '--guidance-mode', 'learned'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', nargs='+', choices=VARIANTS,
                   default=['baseline', 'spatial', 'spectral', 'dual', 'capacity'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--sf', type=int, choices=[4, 8], default=4)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    plan = []
    for seed in args.seeds:
        for name in args.variants:
            folder = ROOT / 'Checkpoint' / 'ablation' / f'x{args.sf}' / f'{name}_seed{seed}'
            cmd = [sys.executable, str(ROOT / 'Train.py'), '--data_path', str(ROOT / 'Data/Train'),
                   '--epochs', str(args.epochs), '--batch_size', '1', '--patch_size', '64',
                   '--lr', '0.0001', '--sf', str(args.sf), '--trainset_num', '1000',
                   '--seed', str(seed), '--save_dir', str(folder), *VARIANTS[name]]
            evaluation = [sys.executable, str(ROOT / 'Test.py'), '--data_path', str(ROOT / 'Data/Test'),
                          '--checkpoint', str(folder / 'last.pth'), '--sf', str(args.sf),
                          '--seed', '1', '--report', str(folder / 'metrics.json'), *VARIANTS[name]]
            plan.append({'variant': name, 'seed': seed, 'train': cmd, 'test': evaluation})
    output = ROOT / 'physics' / f'ablation_plan_x{args.sf}.json'
    output.write_text(json.dumps(plan, indent=2) + '\n')
    for job in plan:
        print(shlex.join(job['train']))
    print(f'Plan: {output}', flush=True)
    if not args.execute:
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Plan saved; no training started.')
    # Fresh runs only: prevent accidental replacement of existing experiments.
    for job in plan:
        folder = Path(job['train'][job['train'].index('--save_dir') + 1])
        if folder.exists() and any(folder.iterdir()):
            raise SystemExit(f'Experiment directory is not empty: {folder}')
    for job in plan:
        folder = Path(job['train'][job['train'].index('--save_dir') + 1])
        folder.mkdir(parents=True, exist_ok=True)
        for phase in ('train', 'test'):
            with (folder / f'{phase}.log').open('w') as log:
                subprocess.run(job[phase], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()

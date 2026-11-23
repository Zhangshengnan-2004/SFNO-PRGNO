"""Sequential V3 ablation training and paired x4/x8 evaluation. Default: dry run."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SPECS = {
    'full': [],
    'deform_stage1': ['--ablation', 'deform_stage1'],
    'deform_stage2': ['--ablation', 'deform_stage2'],
    'deform_stage3': ['--ablation', 'deform_stage3'],
    'no_deformable': ['--ablation', 'no_deformable'],
    'no_regional': ['--ablation', 'no_regional'],
    'fixed_gate': ['--ablation', 'fixed_gate'],
    'fixed_grid': ['--ablation', 'fixed_grid'],
    'spatial_only': ['--disable-spectral-residual'],
    'spectral_only': ['--disable-spatial-residual'],
    'learned_guidance': ['--guidance-mode', 'learned'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-existing', action='store_true', help='Resume last.pth in existing experiment directories')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--variants', choices=SPECS, nargs='+', default=list(SPECS))
    parser.add_argument('--output', type=Path, default=ROOT/'Checkpoint/ablation_v3')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize:
        rows = []
        for path in sorted(output.glob('*/metrics*.json')):
            report = json.loads(path.read_text())
            if report['epoch'] != 600 or len(report['per_scene']) != 12:
                raise ValueError(f'Incomplete protocol: {path}')
            rows.append(dict(experiment=path.parent.name, epoch=report['epoch'],
                             sf=report['sf'], **report['mean']))
        output.mkdir(parents=True, exist_ok=True)
        with (output/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['experiment','epoch','sf','psnr','sam','ergas'])
            writer.writeheader(); writer.writerows(rows)
        print(f'{len(rows)} reports → {output / "comparison.csv"}')
        return
    if args.execute:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no training started')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    for seed in args.seeds:
        for variant in args.variants:
            experiment = output/f'{variant}_seed{seed}'
            resume = experiment/'last.pth'
            if args.execute and experiment.exists() and not (args.resume_existing and resume.is_file()):
                raise FileExistsError(f'Refusing to overwrite {experiment}; use --resume-existing with last.pth')
            train = [sys.executable, '-u', 'Train_V3.py', '--data_path', 'Data/Train',
                     '--epochs', '600', '--batch_size', '1', '--patch_size', '64',
                     '--lr', '0.0001', '--sf', '4', '--trainset_num', '1000',
                     '--num_workers', '2', '--seed', str(seed), '--save_every', '600',
                     '--ablation-lr-schedule', '--save_dir', str(experiment), *SPECS[variant]]
            if args.resume_existing and resume.is_file():
                train += ['--resume', str(resume)]
            jobs = [(train, experiment/'train_continuous.log')]
            for sf in (4,8):
                suffix = '' if sf == 4 else '_x8'
                test = [sys.executable, '-u', 'Test_V3.py', '--data_path', 'Data/Test',
                        '--checkpoint', str(resume), '--sf', str(sf),
                        '--seed', '1', '--report', str(experiment/f'metrics{suffix}.json')]
                if sf == 8: test += ['--cross-scale']
                jobs.append((test, experiment/f'test{suffix}.log'))
            for command, log in jobs:
                print(' '.join(command), flush=True)
                if args.execute:
                    # Training creates its own directory after the overwrite guard.
                    log = experiment.parent / (experiment.name + '_' + log.name)
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('x') as stream:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=True)



if __name__ == '__main__':
    main()

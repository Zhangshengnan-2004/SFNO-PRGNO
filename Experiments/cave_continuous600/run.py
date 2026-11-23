"""Inspect or reproduce the relocated CAVE SFNO/Stage3 experiment."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = ROOT / 'Source'
SEEDS = (42, 43, 44)
MODELS = ('sfno', 'v3_stage3')
KEYS = ('psnr', 'sam', 'ergas')


def folder(model, seed):
    return ROOT / 'Checkpoint' / f'cave_{model}_x4_seed{seed}'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def reports():
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for sf in (4, 8):
                path = folder(model, seed) / ('metrics.json' if sf == 4 else 'metrics_x8.json')
                report = json.loads(path.read_text())
                expected = 'sfno' if model == 'sfno' else 'sfno_prgno_v3'
                assert report['epoch'] == 600 and report['source_sf'] == 4
                assert report['sf'] == sf and report['seed'] == 1
                assert report['model'] == expected and len(report['per_scene']) == 12
                assert bool(report['cross_scale']) == (sf == 8)
                if model == 'v3_stage3':
                    assert report['ablation'] == 'deform_stage3'
                assert all(math.isfinite(report['mean'][key]) for key in KEYS)
                rows.append(dict(model=model, seed=seed, sf=sf, **report['mean']))
    return rows


def verify(load_weights=False):
    manifest = json.loads((ROOT / 'migration_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for entry in manifest['files']:
        path = PROJECT / entry['new']
        assert path.is_file() and digest(path) == entry['current_sha256'], path
    snapshot = json.loads((ROOT / 'source_snapshot.json').read_text())
    for name, expected in snapshot['sha256'].items():
        assert digest(ROOT / name) == expected, name
    rows = reports()
    with (ROOT / 'Summary/per_seed.csv').open() as stream:
        old_rows = list(csv.DictReader(stream))
    assert len(old_rows) == len(rows) == 12
    aliases = {'sfno': 'baseline', 'v3_stage3': 'deform_stage3'}
    for row in rows:
        old = next(value for value in old_rows if value['model'] == aliases[row['model']]
                   and int(value['seed']) == row['seed'] and int(value['sf']) == row['sf'])
        assert all(abs(float(old[key]) - row[key]) < 1e-10 for key in KEYS)
    for model in MODELS:
        for seed in SEEDS:
            with (folder(model, seed) / 'train_log.csv').open() as stream:
                train = list(csv.DictReader(stream))
            assert [int(row['epoch']) for row in train] == list(range(1, 601))
            logs = ROOT / 'Runs' / f'cave_{model}_x4_seed{seed}'
            assert all((logs / name).is_file() for name in ('train.log', 'test_x4.log', 'test_x8.log'))
    loaded = []
    if load_weights:
        import torch
        torch.set_num_threads(4)
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'Model')]
        from SFNO import SFNO
        from SFNO_PRGNO_V3 import SFNO_PRGNO_V3
        for model in MODELS:
            for seed in SEEDS:
                path = folder(model, seed)
                checkpoint = torch.load(path / 'last.pth', map_location='cpu', weights_only=False)
                config = checkpoint['config']
                assert checkpoint['epoch'] == 600 and config['seed'] == seed
                # The native checkpoint stores Path objects; JSON records their strings.
                serialized_config = json.loads(json.dumps(config, default=str))
                assert serialized_config == json.loads((path / 'config.json').read_text())
                assert config['sf'] == 4 and config['epochs'] == 600
                assert config['ablation_lr_schedule'] and not config['amp']
                if model == 'sfno':
                    network = SFNO(size=64)
                else:
                    assert config['ablation'] == 'deform_stage3'
                    network = SFNO_PRGNO_V3(size=64, response=checkpoint['model']['physics.spectral.response'],
                                            ablation='deform_stage3', scale_factor=4)
                network.load_state_dict(checkpoint['model'], strict=True)
                network.eval()
                with torch.inference_mode():
                    for sf in (4, 8):
                        prediction = network(torch.rand(1, 31, 16 // sf, 16 // sf), torch.rand(1, 3, 16, 16), sf)
                        assert prediction.shape == (1, 31, 16, 16) and torch.isfinite(prediction).all()
                loaded.append(dict(model=model, seed=seed, epoch=600, strict_load=True, cpu_forward_scales=[4, 8]))
                del network, checkpoint
    return dict(status='passed', migrated_files=len(manifest['files']), reports=12,
                training_seeds=list(SEEDS), test_seed=1, models=list(MODELS), loaded_weights=loaded)


def main():
    parser = argparse.ArgumentParser(__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    check = actions.add_parser('verify')
    check.add_argument('--load-weights', action='store_true', help='Also strict-load six CPU models and probe x4/x8 forward')
    check.add_argument('--output', type=Path)
    actions.add_parser('summary')
    for action in ('train', 'test'):
        item = actions.add_parser(action)
        item.add_argument('--model', choices=MODELS, required=True)
        item.add_argument('--seed', type=int, choices=SEEDS, required=True, help='Training seed identifying the experiment')
        item.add_argument('--output', type=Path, required=True, help='New output directory (train) or JSON path (test)')
        item.add_argument('--gpu', type=int, default=0)
        item.add_argument('--data-dir', type=Path)
        item.add_argument('--dry-run', action='store_true')
        if action == 'test':
            item.add_argument('--sf', type=int, choices=(4, 8), required=True)
            item.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.action == 'verify':
        result = verify(args.load_weights)
        if args.output:
            with args.output.open('x') as stream:
                json.dump(result, stream, indent=2)
                stream.write('\n')
        print(json.dumps(result, indent=2))
        return
    if args.action == 'summary':
        result = []
        rows = reports()
        for model in MODELS:
            for sf in (4, 8):
                selected = [row for row in rows if row['model'] == model and row['sf'] == sf]
                entry = dict(model=model, sf=sf, n=3)
                for key in KEYS:
                    values = [row[key] for row in selected]
                    entry[key + '_mean'] = statistics.mean(values)
                    entry[key + '_std'] = statistics.stdev(values)
                result.append(entry)
        print(json.dumps(result, indent=2))
        return
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new location to preserve the archived experiment')
    v3 = args.model == 'v3_stage3'
    script = ('Train_V3.py' if v3 else 'Train.py') if args.action == 'train' else ('Test_V3.py' if v3 else 'Test.py')
    data = (args.data_dir or PROJECT / 'CAVE/Data' / ('Train' if args.action == 'train' else 'Test')).resolve()
    command = [sys.executable, '-u', str(SOURCE / script), '--data_path', str(data), '--gpu', str(args.gpu),
               '--model', 'sfno_prgno_v3' if v3 else 'sfno', '--response', str(SOURCE / 'physics/cave_effective_response.npz')]
    if v3:
        command += ['--ablation', 'deform_stage3']
    if args.action == 'train':
        command += ['--seed', str(args.seed), '--epochs', '600', '--batch_size', '1', '--patch_size', '64',
                    '--trainset_num', '1000', '--num_workers', '2', '--lr', '0.0001', '--sf', '4',
                    '--save_every', '600', '--ablation-lr-schedule', '--save_dir', str(output)]
    else:
        command += ['--seed', '1', '--checkpoint', str(folder(args.model, args.seed) / 'last.pth'),
                    '--sf', str(args.sf), '--report', str(output)]
        if args.sf == 8:
            command.append('--cross-scale')
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    if args.action == 'test' and args.device == 'cpu':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    if args.action == 'test':
        output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


if __name__ == '__main__':
    main()

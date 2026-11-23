"""Run the 600-epoch SFNO baseline, then test x4 and x8."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Checkpoint/sfno_continuous600_x4_seed42'
MODEL = 'sfno'
ABLATION = 'full'
RUN = ROOT / 'Runs/sfno_continuous600_x4_seed42'


def state(phase, **details):
    data = dict(phase=phase, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details)
    temp = RUN / 'status.tmp'
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(RUN / 'status.json')
    print(json.dumps(data), flush=True)


def execute(arguments, log):
    command = [sys.executable, '-u', str(ROOT / 'run.py'), '--model', MODEL, '--ablation', ABLATION, *arguments]
    print('RUN:', ' '.join(map(str, command)), flush=True)
    with (RUN / log).open('x') as f:
        subprocess.run(list(map(str, command)), cwd=ROOT.parent.parent, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--model', choices=['sfno', 'v3'], default='sfno')
    parser.add_argument('--ablation', choices=['full', 'deform_stage3'], default='full')
    parser.add_argument('--wait-for-cave', action='store_true', help='Wait for existing CAVE three-seed queue before GPU work')
    args = parser.parse_args()
    if args.model != 'v3' and args.ablation != 'full':
        parser.error('--ablation deform_stage3 requires --model v3')
    global OUT, RUN, MODEL, ABLATION
    MODEL = args.model
    ABLATION = args.ablation
    if MODEL == 'v3':
        OUT = ROOT / f'Checkpoint/v3_{ABLATION}_continuous600_x4_seed42'
        RUN = ROOT / f'Runs/v3_{ABLATION}_continuous600_x4_seed42'
    RUN.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (RUN / 'runner.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--model', MODEL,
                                      '--ablation', ABLATION, *(['--wait-for-cave'] if args.wait_for_cave else [])],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=ROOT.parent.parent)
        print(f'Launched supervisor PID {child.pid}; logs: {RUN}')
        return
    with (RUN / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'status.json').exists() or (OUT.exists() and any(OUT.iterdir())):
            raise FileExistsError('Existing experiment; refusing duplicate launch')
        try:
            if args.wait_for_cave:
                upstream_path = ROOT.parent / 'Experiments/cave_continuous600/status.json'
                while True:
                    upstream = json.loads(upstream_path.read_text())
                    if upstream['phase'] == 'complete':
                        break
                    if upstream['phase'] == 'failed':
                        raise RuntimeError(f'CAVE queue failed: {upstream}')
                    state('waiting_for_cave', upstream=upstream)
                    time.sleep(30)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                              '--format=csv,noheader,nounits'], text=True).strip():
                    state('waiting_for_gpu_idle')
                    time.sleep(30)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            state('training', gpu=torch.cuda.get_device_name(0), output=str(OUT))
            execute(['train', '--device', 'cuda', '--sf', '4', '--batch-size', '1',
                     '--epochs', '600', '--samples', '1000', '--lr', '0.0001',
                     '--seed', '42', '--output', OUT], 'train.log')
            checkpoint = torch.load(OUT / 'last.pth', map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != 600:
                raise RuntimeError('Training did not reach epoch 600')
            del checkpoint
            for sf in (4, 8):
                state('testing', sf=sf, output=str(OUT))
                execute(['test', '--device', 'cuda', '--sf', str(sf), '--checkpoint',
                         OUT / 'last.pth', '--output', OUT], f'test_x{sf}.log')
                report = json.loads((OUT / f'last_test_x{sf}.json').read_text())
                if (report['epoch'] != 600 or report['source_sf'] != 4 or report['sf'] != sf
                        or report['model'] != MODEL or report['ablation'] != ABLATION or len(report['patches']) != 4):
                    raise RuntimeError('Unexpected evaluation metadata')
                if not all(math.isfinite(v) for v in report['mean'].values()):
                    raise RuntimeError('Non-finite evaluation metrics')
            state('complete', output=str(OUT))
        except Exception as exc:
            state('failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()

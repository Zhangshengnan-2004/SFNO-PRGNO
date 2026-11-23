"""Sequential Chikusei SFNO then V3-stage3 training and x4/x8 evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'Runs/comparison_seed42'


def state(phase, **details):
    value = dict(phase=phase, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details)
    temporary = RUN / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(RUN / 'status.json')
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (RUN / 'execution.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, cwd=ROOT.parent.parent)
        print(f'Supervisor PID {process.pid}; {RUN}')
        return
    with (RUN / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'status.json').exists():
            raise FileExistsError('Existing comparison; inspect status before recovery')
        try:
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True).strip():
                state('waiting_for_gpu_idle')
                time.sleep(30)
            for model, ablation in [('sfno', 'full'), ('v3', 'deform_stage3')]:
                state('running', model=model, ablation=ablation)
                subprocess.run([sys.executable, '-u', str(ROOT/'train_and_test.py'),
                                '--model', model, '--ablation', ablation], check=True)
            state('complete')
        except Exception as exc:
            state('failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()

"""Wait for existing P2Fusion work, train continuous SFNO baseline, test x4/x8."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'reproduction/baseline_continuous_600'
TARGET=ROOT/'Checkpoint/v3_balanced_scales/baseline_seed42'
P2STATE=ROOT.parent.parent/'P2Fusion-main/train_after_v3_status.json'


def state(phase,**extra):
    data=dict(phase=phase,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra)
    p=RUN/'status.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(RUN/'status.json')
    print(data,flush=True)


def run(command,log):
    print('RUN', ' '.join(map(str,command)),flush=True)
    with (RUN/log).open('x') as f:
        subprocess.run(list(map(str,command)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'status.json').exists() or TARGET.exists():
            raise FileExistsError('Existing baseline run; refusing duplicate launch')
        try:
            while True:
                d=json.loads(P2STATE.read_text());phase=d['phase']
                if phase=='p2fusion_complete':break
                if phase not in ('waiting_for_v3','p2fusion_preflight','p2fusion_training'):
                    raise RuntimeError(f'P2Fusion queue did not complete: {d}')
                state('waiting_for_p2fusion',upstream=phase)
                time.sleep(30)
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            state('training',target=str(TARGET))
            run([sys.executable,'-u','Train.py','--model','sfno','--data_path','Data/Train',
                 '--epochs','600','--batch_size','1','--patch_size','64','--lr','0.0001',
                 '--sf','4','--trainset_num','1000','--num_workers','2','--seed','42',
                 '--save_every','600','--ablation-lr-schedule','--save_dir',TARGET],'train.log')
            checkpoint=torch.load(TARGET/'last.pth',map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==600
            for sf in (4,8):
                state('testing',sf=sf)
                suffix='' if sf==4 else '_x8'
                cmd=[sys.executable,'-u','Test.py','--model','sfno','--checkpoint',TARGET/'last.pth',
                     '--data_path','Data/Test','--sf',sf,'--seed','1','--report',TARGET/f'metrics{suffix}.json']
                if sf==8:cmd+=['--cross-scale']
                run(cmd,f'test{suffix}.log')
                d=json.loads((TARGET/f'metrics{suffix}.json').read_text())
                assert d['epoch']==600 and d['sf']==sf and d['source_sf']==4 and len(d['per_scene'])==12
                assert all(math.isfinite(d['mean'][k]) for k in ('psnr','sam','ergas'))
            run([sys.executable,'run_v3_ablation.py','--output',TARGET.parent,'--summarize'],'summary.log')
            state('complete')
        except Exception as e:
            state('failed',error=str(e));raise


if __name__=='__main__':main()

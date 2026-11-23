"""Resume four existing epoch200 models to epoch400, then test x4 and x8."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parent
SPECS = [
    ('sfno', 'train_x4', 'Train.py', 'Test.py', 'sfno'),
    ('v1', 'train_x4_prgno', 'Train.py', 'Test.py', 'sfno_prgno'),
    ('v2', 'train_x4_prgno_v2', 'Train_V2.py', 'Test_V2.py', 'sfno_prgno_v2'),
    ('v3', 'train_x4_prgno_v3', 'Train_V3.py', 'Test_V3.py', 'sfno_prgno_v3'),
]
RUN = ROOT / 'reproduction/continue_200_to_400'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def run_command(command, log):
    print('RUN:', ' '.join(map(str,command)),flush=True)
    with log.open('x') as stream:
        subprocess.run(list(map(str,command)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    if RUN.exists(): raise FileExistsError(f'Refusing to overwrite {RUN}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    historical={}
    for label,folder,train,test,model in SPECS:
        source=ROOT/'Checkpoint'/folder
        target=ROOT/'Checkpoint'/(folder+'_e400')
        if target.exists():raise FileExistsError(target)
        checkpoint=torch.load(source/'last.pth',map_location='cpu',weights_only=False)
        if checkpoint['epoch']!=200:raise ValueError(f'{folder}: expected epoch200')
        config=checkpoint.get('config',{})
        for key,value in dict(sf=4,batch_size=1,patch_size=64,trainset_num=1000,seed=42).items():
            if config.get(key)!=value:raise ValueError(f'{folder}: unexpected {key}={config.get(key)}')
        # Every file in each original experiment is hashed; no originals are written.
        for p in source.rglob('*'):
            if p.is_file():historical[str(p.relative_to(ROOT))]=digest(p)
    RUN.mkdir(parents=True)
    (RUN/'original_sha256.json').write_text(json.dumps(historical,indent=2)+'\n')
    status={label:{'state':'queued'} for label,*_ in SPECS}
    def persist(): (RUN/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    persist()
    for label,folder,train,test,model in SPECS:
        target=ROOT/'Checkpoint'/(folder+'_e400')
        source=ROOT/'Checkpoint'/folder/'last.pth'
        try:
            status[label]={'state':'training','source_epoch':200,'target_epoch':400,'output':str(target.relative_to(ROOT))};persist()
            run_command([sys.executable,'-u',train,'--data_path','Data/Train',
                         '--epochs',400,'--batch_size',1,'--patch_size',64,'--lr',1e-4,
                         '--sf',4,'--trainset_num',1000,'--num_workers',2,'--seed',42,
                         '--model',model,'--resume',source,'--save_dir',target,
                         '--resume-lr',5e-5,'--lr-decay-epochs',300,350,'--lr-gamma',0.5],
                        RUN/(label+'_train.log'))
            # Keep experiment-local log next to weights after successful training.
            import shutil
            shutil.copyfile(RUN/(label+'_train.log'),target/'train.log')
            finished=torch.load(target/'last.pth',map_location='cpu',weights_only=False)
            if finished['epoch']!=400:raise RuntimeError('Final epoch is not400')
            status[label]['state']='testing';persist()
            for sf in (4,8):
                suffix='' if sf==4 else '_x8'
                command=[sys.executable,'-u',test,'--data_path','Data/Test','--checkpoint',target/'last.pth',
                         '--model',model,'--sf',sf,'--seed',1,'--report',target/f'metrics{suffix}.json']
                if sf==8:command+=['--cross-scale']
                run_command(command,target/f'test{suffix}.log')
            status[label]['state']='complete';persist()
        except Exception:
            status[label]['state']='failed';status[label]['error']=traceback.format_exc();persist()
            print(status[label]['error'],flush=True)
    changed=[p for p,h in historical.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    (RUN/'integrity.json').write_text(json.dumps({'originals_unchanged':not changed,'changed':changed},indent=2)+'\n')
    if changed or any(x['state']!='complete' for x in status.values()):raise RuntimeError('Check status/integrity reports')
    print('All four models completed epoch400 and x4/x8 tests. Original epoch200 artifacts unchanged.',flush=True)


if __name__=='__main__': main()

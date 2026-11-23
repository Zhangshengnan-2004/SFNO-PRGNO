"""Wait for all epoch400 work, then continue four models to epoch600 and test."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from run_continue_400 import ROOT, SPECS, digest, run_command

RUN = ROOT/'reproduction/continue_400_to_600'
PREVIOUS = ROOT/'reproduction/continue_200_to_400'
SERVICE = 'sfno-continue-400-recovery.service'


def verify_report(path, epoch, sf):
    data=json.loads(path.read_text())
    if data.get('epoch')!=epoch or data.get('sf')!=sf or len(data.get('per_scene',[]))!=12:
        raise ValueError(f'Unexpected test protocol: {path}')
    if data.get('source_sf')!=4 or bool(data.get('cross_scale'))!=(sf==8):
        raise ValueError(f'Unexpected training/test scale: {path}')
    if not all(math.isfinite(data['mean'][k]) for k in ('psnr','sam','ergas')):
        raise ValueError(f'Nonfinite metrics: {path}')
    return data


def checkpoint_epoch(path, expected):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('epoch')!=expected:raise ValueError(f'Expected epoch{expected}: {path}')
    return checkpoint


def write_status(status):
    temp=RUN/'status.tmp'
    temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(RUN/'status.json')


def main():
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    if (RUN/'status.json').exists():raise FileExistsError('Existing600 run; refusing overwrite')
    RUN.mkdir(parents=True,exist_ok=True)
    status={'phase':'waiting_for_all_400','target_epoch':600,
            'models':{label:{'state':'queued'} for label,*_ in SPECS}}
    write_status(status)
    try:
        # A simple systemd After= dependency alone does not wait for training completion.
        while True:
            result=subprocess.run(['systemctl','--user','show',SERVICE,'--property=ActiveState','--value'],
                                  capture_output=True,text=True,check=True)
            active=result.stdout.strip()
            if active in ('active','activating','reloading','deactivating'):
                time.sleep(30);continue
            if active!='inactive':raise RuntimeError(f'400 service state is {active}; inspect400 logs')
            break
        previous=json.loads((PREVIOUS/'status.json').read_text())
        if any(previous.get(label,{}).get('state')!='complete' for label,*_ in SPECS):
            raise RuntimeError('Not all four epoch400 runs and tests are complete;600 not started')
        historical=json.loads((PREVIOUS/'original_sha256.json').read_text())
        for label,folder,train,test,model in SPECS:
            source=ROOT/'Checkpoint'/(folder+'_e400')
            target=ROOT/'Checkpoint'/(folder+'_e600')
            if target.exists():raise FileExistsError(target)
            c=checkpoint_epoch(source/'last.pth',400)
            if c.get('config',{}).get('sf')!=4:raise ValueError('Only x4-trained checkpoints allowed')
            for sf in (4,8):verify_report(source/('metrics.json' if sf==4 else 'metrics_x8.json'),400,sf)
            for p in source.rglob('*'):
                if p.is_file():historical[str(p.relative_to(ROOT))]=digest(p)
        changed=[p for p,h in historical.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
        if changed:raise RuntimeError(f'Original files changed: {changed}')
        (RUN/'original_sha256.json').write_text(json.dumps(historical,indent=2)+'\n')
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        status['phase']='running';write_status(status)
        for label,folder,train,test,model in SPECS:
            source=ROOT/'Checkpoint'/(folder+'_e400')/'last.pth'
            target=ROOT/'Checkpoint'/(folder+'_e600')
            try:
                status['models'][label]={'state':'training','source_epoch':400,'target_epoch':600,
                                          'output':str(target.relative_to(ROOT))};write_status(status)
                log=RUN/(label+'_train.log')
                run_command([sys.executable,'-u',train,'--data_path','Data/Train','--epochs',600,
                             '--batch_size',1,'--patch_size',64,'--lr',1e-4,'--sf',4,'--trainset_num',1000,
                             '--num_workers',2,'--seed',42,'--model',model,'--resume',source,'--save_dir',target,
                             '--resume-lr',1.25e-5,'--lr-decay-epochs',500,550,'--lr-gamma',.5],log)
                shutil.copyfile(log,target/'train.log')
                checkpoint_epoch(target/'last.pth',600)
                status['models'][label]['state']='testing';write_status(status)
                for sf in (4,8):
                    suffix='' if sf==4 else '_x8'
                    command=[sys.executable,'-u',test,'--data_path','Data/Test','--checkpoint',target/'last.pth',
                             '--model',model,'--sf',sf,'--seed',1,'--report',target/f'metrics{suffix}.json']
                    if sf==8:command+=['--cross-scale']
                    run_command(command,target/f'test{suffix}.log')
                    verify_report(target/f'metrics{suffix}.json',600,sf)
                status['models'][label]['state']='complete';write_status(status)
            except Exception:
                status['models'][label]['state']='failed'
                status['models'][label]['error']=traceback.format_exc();write_status(status)
                print(status['models'][label]['error'],flush=True)
        changed=[p for p,h in historical.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
        (RUN/'integrity.json').write_text(json.dumps({'originals_unchanged':not changed,'changed':changed},indent=2)+'\n')
        if changed or any(x['state']!='complete' for x in status['models'].values()):
            raise RuntimeError('Some600 tasks incomplete or original artifacts changed')
        rows=[]
        for label,folder,*_ in SPECS:
            for epoch in (400,600):
                for sf in (4,8):
                    path=ROOT/'Checkpoint'/f'{folder}_e{epoch}'/('metrics.json' if sf==4 else 'metrics_x8.json')
                    report=verify_report(path,epoch,sf)
                    rows.append(dict(model=label,epoch=epoch,test_sf=sf,**report['mean']))
        with (RUN/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['model','epoch','test_sf','psnr','sam','ergas'])
            writer.writeheader();writer.writerows(rows)
        status['phase']='complete';write_status(status)
        print('All four models trained600 epochs; x4/x8 tests complete;200/400 artifacts unchanged.',flush=True)
    except Exception:
        status['phase']='failed';status['error']=traceback.format_exc();write_status(status);raise


if __name__=='__main__':main()

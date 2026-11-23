"""Recover interrupted continuation without repeating completed training."""
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import torch
from run_continue_400 import ROOT, RUN, SPECS, digest, run_command


def main():
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    status=json.loads((RUN/'status.json').read_text())
    def persist(): (RUN/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    # Finish unfinished training first; completed models only need their missing tests.
    ordered=sorted(SPECS,key=lambda s:s[0] not in ('v2','v3'))
    for label,folder,train,test,model in ordered:
        target=ROOT/'Checkpoint'/(folder+'_e400')
        existing=target/'last.pth'
        source=existing if existing.exists() else ROOT/'Checkpoint'/folder/'last.pth'
        try:
            c=torch.load(source,map_location='cpu',weights_only=False)
            epoch=int(c['epoch'])
            if epoch>400:raise RuntimeError('Unexpected checkpoint epoch')
            if epoch<400:
                status[label]={'state':'training','resumed_epoch':epoch,'target_epoch':400,'output':str(target.relative_to(ROOT))};persist()
                log=RUN/(label+f'_resume_from_{epoch}.log')
                command=[sys.executable,'-u',train,'--data_path','Data/Train','--epochs',400,
                         '--batch_size',1,'--patch_size',64,'--lr',1e-4,'--sf',4,'--trainset_num',1000,
                         '--num_workers',2,'--seed',42,'--model',model,'--resume',source,'--save_dir',target,
                         '--resume-lr',5e-5,'--lr-decay-epochs',300,350,'--lr-gamma',.5]
                run_command(command,log)
                shutil.copyfile(log,target/'train_resume.log')
                c=torch.load(existing,map_location='cpu',weights_only=False)
                if c['epoch']!=400:raise RuntimeError('Training incomplete')
            status[label]={'state':'testing','training_epoch':400,'target_epoch':400,'output':str(target.relative_to(ROOT))};persist()
            for sf in (4,8):
                suffix='' if sf==4 else '_x8'
                report=target/f'metrics{suffix}.json'
                if report.exists():
                    data=json.loads(report.read_text())
                    if data.get('epoch')==400 and data.get('sf')==sf:continue
                    raise RuntimeError(f'Unexpected existing report: {report}')
                log=target/f'test{suffix}.log'
                if log.exists():
                    archived=target/f'test{suffix}_interrupted.log'
                    if archived.exists():raise FileExistsError(archived)
                    log.rename(archived)
                command=[sys.executable,'-u',test,'--data_path','Data/Test','--checkpoint',existing,
                         '--model',model,'--sf',sf,'--seed',1,'--report',report]
                if sf==8:command+=['--cross-scale']
                run_command(command,log)
            status[label]['state']='complete';persist()
        except Exception:
            status[label]={'state':'failed','error':traceback.format_exc()};persist()
            print(status[label]['error'],flush=True)
    historical=json.loads((RUN/'original_sha256.json').read_text())
    changed=[p for p,h in historical.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    (RUN/'integrity.json').write_text(json.dumps({'originals_unchanged':not changed,'changed':changed},indent=2)+'\n')
    if changed or any(s['state']!='complete' for s in status.values()):raise RuntimeError('Some tasks incomplete; see status.json')
    print('All models at epoch400, all x4/x8 reports saved; originals unchanged.',flush=True)


if __name__=='__main__':main()

"""Run six fresh SFNO jobs sequentially; preserve experiments and aggregate seed statistics."""
import argparse
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from protocol import ROOT,DEFAULTS,load_protocol,sha256,atomic_json,output_dir

RUN=ROOT/'Runs'
JOBS=[(dataset,seed) for seed in (42,43,44) for dataset in ('pavia','chikusei')]


def state(phase,**extra):
    d=dict(phase=phase,pid=os.getpid(),updated_at=datetime.now().astimezone().isoformat(),**extra)
    atomic_json(RUN/'status.json',d);print(json.dumps(d),flush=True)


def summarize():
    rows=[]
    for dataset,seed in JOBS:
        for sf in (4,8):
            path=output_dir(dataset,seed)/f'last_test_x{sf}.json'
            if not path.exists():continue
            d=json.loads(path.read_text())
            if d.get('protocol_sha256')!=sha256(ROOT/'protocol.json'):
                raise ValueError(f'Report belongs to a different frozen protocol: {path}')
            if d['epoch']!=DEFAULTS['epochs'] or d['dataset']!=dataset or d['seed']!=seed or d['sf']!=sf:
                raise ValueError(f'Unexpected report metadata: {path}')
            if not all(math.isfinite(v) for v in d['mean'].values()):raise ValueError('Nonfinite metrics')
            rows.append(dict(dataset=dataset,seed=seed,sf=sf,**d['mean']))
    if rows:
        with (RUN/'per_seed.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[];lines=['# SFNO corrected protocol: three training seeds','',
        '先对每个训练种子的测试图块取均值，再对seed42/43/44取均值和样本标准差（ddof=1）。未齐三种子时不生成三种子标准差。','',
        '| Dataset | Scale | Completed seeds | PSNR ↑ | SAM ↓ | ERGAS ↓ |',
        '|---|---|---|---|---|---|']
    for dataset in ('pavia','chikusei'):
        for sf in (4,8):
            selected=[r for r in rows if r['dataset']==dataset and r['sf']==sf]
            if not selected:continue
            n=len(selected);mean={k:statistics.mean(r[k] for r in selected) for k in ('psnr','sam_degrees','ergas')}
            std={k:statistics.stdev(r[k] for r in selected) for k in mean} if n==3 else None
            groups.append(dict(dataset=dataset,sf=sf,seeds=[r['seed'] for r in selected],n=n,mean=mean,sample_std=std))
            values=[f'{mean[k]:.4f} ± {std[k]:.4f}' if std else f'{mean[k]:.4f} (partial)' for k in mean]
            lines.append(f'| {dataset} | x{sf} | {n}/3 | '+ ' | '.join(values)+' |')
    atomic_json(RUN/'summary.json',dict(groups=groups,completed_reports=len(rows),required_reports=12))
    (RUN/'summary.md').write_text('\n'.join(lines)+'\n')
    if len(rows)==12:
        with (RUN/'mean_std.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['dataset','sf','metric','mean','sample_std_ddof1'])
            for group in groups:
                for metric in group['mean']:w.writerow([group['dataset'],group['sf'],metric,group['mean'][metric],group['sample_std'][metric]])
    return len(rows)


def gpu_idle(job):
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        state('waiting_for_gpu_idle',job=job);time.sleep(30)


def execute(args,log):
    with log.open('x') as f:
        subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args();RUN.mkdir(exist_ok=True)
    if args.summarize:summarize();return
    if args.detach:
        if (RUN/'status.json').exists() and not args.resume:raise FileExistsError('Existing queue; no duplicate launch')
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        with (RUN/f'execution_{stamp}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*(['--resume'] if args.resume else [])],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        print(f'Queue PID {child.pid}; {RUN}');return
    with (RUN/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'status.json').exists() and not args.resume:raise FileExistsError('Existing queue')
        try:
            p=load_protocol();validation=json.loads((ROOT/'validation.json').read_text())
            if validation['status']!='passed' or not validation.get('technical_preflight_passed') or not validation.get('runtime_quality_guards') or not validation.get('shared_pr_accumulation_verified') or validation['protocol_sha256']!=sha256(ROOT/'protocol.json'):
                raise ValueError('Preflight for exact frozen protocol required')
            for dataset,seed in JOBS:
                job=f'{dataset}_seed{seed}';out=output_dir(dataset,seed);logs=RUN/job;logs.mkdir(exist_ok=True)
                if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError(out)
                metadata_path=out/'checkpoint_metadata.json'
                completed=json.loads(metadata_path.read_text())['epoch'] if metadata_path.exists() else 0
                if completed<DEFAULTS['epochs']:
                    if out.exists() and any(out.iterdir()) and not (out/'last.pth').exists():
                        raise RuntimeError('Partial run without checkpoint; inspect before restarting in a new directory')
                    gpu_idle(job);state('training',job=job,output=str(out),completed_jobs=sum((output_dir(d,s)/'complete.json').exists() for d,s in JOBS),total_jobs=6)
                    arguments=['train','--dataset',dataset,'--seed',str(seed),'--output',str(out),'--device','cuda']
                    if completed:arguments+=['--resume',str(out/'last.pth')]
                    execute(arguments,logs/f'train_from_epoch{completed}_{time.time_ns()}.log')
                    completed=json.loads(metadata_path.read_text())['epoch']
                if completed!=DEFAULTS['epochs']:raise ValueError('Training did not reach fixed budget')
                for sf in (4,8):
                    if (out/f'last_test_x{sf}.json').exists():continue
                    gpu_idle(job);state('testing',job=job,sf=sf)
                    execute(['test','--dataset',dataset,'--seed',str(seed),'--checkpoint',str(out/'last.pth'),
                             '--sf',str(sf),'--output',str(out),'--device','cuda'],logs/f'test_x{sf}_{time.time_ns()}.log')
                count=summarize();atomic_json(out/'complete.json',dict(dataset=dataset,seed=seed,epoch=completed,scales=[4,8]))
                state('job_complete',job=job,completed_reports=count,total_reports=12)
            if summarize()!=12:raise ValueError('Missing final reports')
            state('complete',completed_jobs=6,completed_reports=12)
        except Exception as exc:
            state('failed',error=str(exc));raise


if __name__=='__main__':main()

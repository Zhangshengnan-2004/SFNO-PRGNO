"""Paired baseline vs Stage3 seeds 42/43/44; continuous 600 epochs, x4/x8."""
import argparse,csv,fcntl,hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Checkpoint/v3_balanced_scales'
RUN=ROOT/'reproduction/baseline_stage3_three_seeds'
UPSTREAM=ROOT.parent/'Pavia/Runs/v3_full_continuous600_x4_seed42/status.json'
SEEDS=(42,43,44)
MODELS={'baseline':('Train.py','Test.py','sfno',[]),
        'deform_stage3':('Train_V3.py','Test_V3.py','sfno_prgno_v3',['--ablation','deform_stage3'])}
METRICS=('psnr','sam','ergas')


def report(model,seed,sf):
    path=OUT/f'{model}_seed{seed}'/('metrics.json' if sf==4 else 'metrics_x8.json')
    d=json.loads(path.read_text())
    expected=MODELS[model][2]
    if not (d['epoch']==600 and d['sf']==sf and d['source_sf']==4 and d['seed']==1
            and d['model']==expected and bool(d['cross_scale'])==(sf==8) and len(d['per_scene'])==12):
        raise ValueError(f'Invalid protocol: {path}')
    if model=='deform_stage3' and d.get('ablation')!='deform_stage3':raise ValueError(path)
    if not all(math.isfinite(d['mean'][k]) for k in METRICS):raise ValueError(path)
    return d


def summarize():
    rows=[];missing=[]
    for model in MODELS:
        for seed in SEEDS:
            for sf in (4,8):
                try:d=report(model,seed,sf)
                except FileNotFoundError:
                    missing.append(f'{model}/seed{seed}/x{sf}');continue
                rows.append(dict(model=model,seed=seed,sf=sf,**d['mean']))
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','seed','sf',*METRICS]);w.writeheader();w.writerows(rows)
    result=dict(seeds=list(SEEDS),std_definition='sample standard deviation ddof=1 across training seeds; each seed metric is mean over 12 test scenes',
                complete=not missing,missing=missing,summary=[])
    if not missing:
        for model in MODELS:
            for sf in (4,8):
                group=[r for r in rows if r['model']==model and r['sf']==sf]
                item=dict(model=model,sf=sf,n=3)
                for k in METRICS:
                    values=[r[k] for r in group];item[k+'_mean']=statistics.mean(values);item[k+'_std']=statistics.stdev(values)
                result['summary'].append(item)
        paired=[]
        for sf in (4,8):
            item=dict(sf=sf)
            for k in METRICS:
                deltas=[next(r[k] for r in rows if r['model']=='deform_stage3' and r['seed']==seed and r['sf']==sf)-next(r[k] for r in rows if r['model']=='baseline' and r['seed']==seed and r['sf']==sf) for seed in SEEDS]
                item[k+'_delta_mean']=statistics.mean(deltas);item[k+'_delta_std']=statistics.stdev(deltas)
            paired.append(item)
        result['paired_stage3_minus_baseline']=paired
        with (RUN/'mean_std.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result['summary'][0]));w.writeheader();w.writerows(result['summary'])
        lines=['| 模型 | 尺度 | PSNR ↑ | SAM ↓ | ERGAS ↓ |','|---|---|---|---|---|']
        for r in result['summary']:
            vals=[f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" for k in METRICS]
            lines.append('| '+ ' | '.join([r['model'],f"×{r['sf']}",*vals])+' |')
        (RUN/'mean_std.md').write_text('\n'.join(lines)+'\n\n3 training seeds; sample std (ddof=1).\n')
    (RUN/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def state(phase,**details):
    d=dict(phase=phase,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**details)
    p=RUN/'status.tmp';p.write_text(json.dumps(d,indent=2)+'\n');p.replace(RUN/'status.json');print(d,flush=True)


def checkpoint_check(model,seed):
    import torch
    d=torch.load(OUT/f'{model}_seed{seed}/last.pth',map_location='cpu',weights_only=False)
    c=d['config']
    expected=dict(model=MODELS[model][2],seed=seed,epochs=600,sf=4,batch_size=1,patch_size=64,trainset_num=1000,lr=1e-4,ablation_lr_schedule=True,amp=False)
    if model=='deform_stage3':expected['ablation']='deform_stage3'
    if d['epoch']!=600 or any(c.get(k)!=v for k,v in expected.items()):raise ValueError(f'Checkpoint protocol mismatch: {model}, seed={seed}')


def execute(command,log):
    print('RUN',' '.join(map(str,command)),flush=True)
    with (RUN/log).open('x') as f:
        subprocess.run(list(map(str,command)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.summarize:
        d=summarize();print('Complete:',d['complete'],'Missing:',d['missing']);return
    with (RUN/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'status.json').exists():raise FileExistsError('Existing queue; refusing duplicate launch')
        try:
            # Preserve original artifacts; fail before launch if new destinations exist.
            for seed in (43,44):
                for model in MODELS:
                    if (OUT/f'{model}_seed{seed}').exists():raise FileExistsError(OUT/f'{model}_seed{seed}')
            for model in MODELS:
                checkpoint_check(model,42)
                for sf in (4,8):report(model,42,sf)
            historical={}
            for model in MODELS:
                for name in ('last.pth','metrics.json','metrics_x8.json'):
                    p=OUT/f'{model}_seed42'/name;historical[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            (RUN/'seed42_sha256.json').write_text(json.dumps(historical,indent=2)+'\n')
            summarize()
            while True:
                upstream=json.loads(UPSTREAM.read_text())
                if upstream['phase']=='complete':break
                if upstream['phase']=='failed':raise RuntimeError(f'Pavia upstream failed: {upstream}')
                pid=upstream.get('pid')
                if pid:os.kill(pid,0)
                state('waiting_for_pavia',upstream=upstream['phase'])
                time.sleep(30)
            # Also avoid any unrelated compute process already using the single GPU.
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
                state('waiting_for_gpu_idle');time.sleep(30)
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            for seed in (43,44):
                for model,(train,test,model_name,extra) in MODELS.items():
                    target=OUT/f'{model}_seed{seed}'
                    state('training',model=model,seed=seed,target=str(target))
                    execute([sys.executable,'-u',train,'--model',model_name,'--data_path','Data/Train',
                        '--epochs',600,'--batch_size',1,'--patch_size',64,'--lr',1e-4,'--sf',4,
                        '--trainset_num',1000,'--num_workers',2,'--seed',seed,'--save_every',600,
                        '--ablation-lr-schedule','--save_dir',target,*extra],f'{model}_seed{seed}_train.log')
                    checkpoint_check(model,seed)
                    for sf in (4,8):
                        state('testing',model=model,seed=seed,sf=sf)
                        suffix='' if sf==4 else '_x8'
                        cmd=[sys.executable,'-u',test,'--model',model_name,'--data_path','Data/Test',
                             '--checkpoint',target/'last.pth','--sf',sf,'--seed',1,'--report',target/f'metrics{suffix}.json']
                        if sf==8:cmd+=['--cross-scale']
                        execute(cmd,f'{model}_seed{seed}_test{suffix}.log');report(model,seed,sf)
                    summarize()
            if not summarize()['complete']:raise RuntimeError('Incomplete three-seed results')
            for path,digest in historical.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Original changed: {path}')
            state('complete')
        except Exception as e:
            state('failed',error=str(e));raise


if __name__=='__main__':main()

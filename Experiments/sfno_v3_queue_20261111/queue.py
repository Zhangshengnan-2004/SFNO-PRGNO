"""Immutable outer queue: adopt running Chikusei42, then interleave SFNO/V3 by seed."""
import argparse,csv,fcntl,hashlib,json,math,os,signal,statistics,subprocess,sys,time
from contextlib import ExitStack
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ROOTS={'sfno':ROOT.parent/'sfno_restart_20261111','v3_stage3':ROOT.parent/'v3_stage3_restart_20261111'}
JOBS=[(model,dataset,seed) for seed in (42,43,44) for model in ('sfno','v3_stage3') for dataset in ('pavia','chikusei')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def proc_args(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_text().split('\0')
    except (FileNotFoundError,ProcessLookupError):return []


def proc_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


def alive(pid,args,start):
    return proc_start(pid)==start and proc_args(pid)==args


def state(phase,**extra):
    value=dict(phase=phase,pid=os.getpid(),updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),**extra)
    save(ROOT/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
    active_path=ROOT.parent/'active_sfno.json'
    active=json.loads(active_path.read_text()) if active_path.exists() else {}
    active.update(combined_queue=str(ROOT),status_file=str(ROOT/'status.json'),queue_pid=os.getpid(),phase=phase,
                  model_experiments={k:str(v) for k,v in ROOTS.items()},updated_at=value['updated_at'])
    if extra.get('job'):
        active['current_job']=extra['job']
        if extra['job'][0] in ROOTS:active['experiment']=str(ROOTS[extra['job'][0]])
    save(active_path,active)


def output(model,dataset,seed):
    suffix='sfno' if model=='sfno' else 'v3_stage3'
    return ROOTS[model]/'Checkpoint'/f'{dataset}_{suffix}_x4_seed{seed}'


def verify(model,plan):
    root=ROOTS[model];p=json.loads((root/'protocol.json').read_text())
    if sha(root/'protocol.json')!=plan['protocol_sha256'][model]:raise ValueError('Frozen protocol changed:'+model)
    for name,digest in p['source_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen source changed:'+str(root/name))
    for spec in p['datasets'].values():
        if sha(root/spec['response_file'])!=spec['response_sha256']:raise ValueError('Frozen SRF changed')
    audit='paper_alignment' if model=='sfno' else 'lineage'
    if sha(root/(audit+'.json'))!=p[audit+'_sha256']:raise ValueError('Frozen provenance changed')
    return p


def gpu_idle(job):
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        state('waiting_for_gpu_idle',job=job);time.sleep(20)


def execute(model,args,log,phase,job):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:
        child=subprocess.Popen([sys.executable,'-u',str(ROOTS[model]/args[0]),*args[1:]],cwd=ROOTS[model],stdout=f,stderr=subprocess.STDOUT)
        state(phase,job=job,worker_pid=child.pid,worker_cmd=proc_args(child.pid),worker_start=proc_start(child.pid),log=str(log))
        if child.wait()!=0:raise RuntimeError('Worker failed; checkpoint/log retained: '+str(log))


def adopt_current():
    path=ROOTS['sfno']/'Runs/status.json';old=json.loads(path.read_text());supervisor=old['pid']
    if old['phase']!='training' or old['job']!='chikusei_seed42':raise ValueError('Expected active Chikusei SFNO seed42')
    if str(ROOTS['sfno']/'run_queue.py') not in proc_args(supervisor):raise ValueError('Original supervisor identity changed')
    candidates=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=proc_args(int(p.name))
            if str(ROOTS['sfno']/'run.py') in args and 'train' in args and '--dataset' in args and args[args.index('--dataset')+1]=='chikusei' and args[args.index('--seed')+1]=='42':
                candidates.append((int(p.name),args))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    if len(candidates)!=1:raise RuntimeError(f'Expected one running target worker: {candidates}')
    pid,args=candidates[0];start=proc_start(pid)
    # Freeze only the dispatcher; the GPU training child is never signalled.
    os.kill(supervisor,signal.SIGSTOP)
    try:
        if not alive(pid,args,start):raise RuntimeError('Target worker changed during handoff')
        record=dict(original_supervisor_pid=supervisor,worker_pid=pid,worker_args=args,worker_start=start,
                    adopted_job=['sfno','chikusei',42],original_status=old)
        save(ROOT/'handoff.json',record)
        os.kill(supervisor,signal.SIGTERM);os.kill(supervisor,signal.SIGCONT)
    except BaseException:
        try:os.kill(supervisor,signal.SIGCONT)
        except ProcessLookupError:pass
        raise
    old.update(phase='managed_by_combined_queue',manager_pid=os.getpid(),manager_status=str(ROOT/'status.json'))
    save(path,old)
    return record


def wait_worker(record):
    while alive(record['worker_pid'],record['worker_args'],record['worker_start']):
        state('adopted_training',job=['sfno','chikusei',42],worker_pid=record['worker_pid'],output=str(output('sfno','chikusei',42)))
        time.sleep(20)
    metadata=output('sfno','chikusei',42)/'checkpoint_metadata.json'
    if not metadata.exists() or json.loads(metadata.read_text())['epoch']!=600:
        raise RuntimeError('Adopted Chikusei42 did not finish600 epochs; inspect original training log')


def ensure_validation(model,plan):
    root=ROOTS[model];path=root/'validation.json'
    if model=='v3_stage3' and not path.exists():
        probe=json.loads((root/'cpu_probe.json').read_text())
        if probe['status']!='passed' or probe['protocol_sha256']!=plan['protocol_sha256'][model]:raise ValueError('CPU validation missing')
        gpu_idle([model,'technical_preflight'])
        execute(model,['validate.py'],root/'Runs'/f'validation_{time.time_ns()}.log','validating',[model,'technical_preflight'])
    v=json.loads(path.read_text())
    required=['technical_preflight_passed','runtime_quality_guards','shared_pr_accumulation_verified','full_test_entry_and_logs']
    required+=['exact_resume_model_and_adam'] if model=='sfno' else ['resume_model_and_adam_within_tolerance']
    if v['status']!='passed' or v['protocol_sha256']!=plan['protocol_sha256'][model] or not all(v.get(k) for k in required):
        raise ValueError('Exact frozen protocol technical preflight failed: '+model)


def reports(model,dataset,seed,plan):
    root=output(model,dataset,seed);rows=[]
    for sf in (4,8):
        path=root/f'last_test_x{sf}.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        if (d['protocol_sha256'],d['model'],d['dataset'],d['seed'],d['sf'],d['epoch'])!=(plan['protocol_sha256'][model],model,dataset,seed,sf,600):
            raise ValueError('Mixed or incomplete report:'+str(path))
        if not (root/('test.log' if sf==4 else 'test_x8.log')).exists():raise ValueError('Report exists but test log missing')
        if not all(math.isfinite(v) for v in d['mean'].values()):raise ValueError('Nonfinite report')
        rows.append(dict(model=model,dataset=dataset,seed=seed,sf=sf,**d['mean']))
    return rows


def summarize(plan):
    rows=[r for model,dataset,seed in JOBS for r in reports(model,dataset,seed,plan)]
    if rows:
        with (ROOT/'per_seed.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[];lines=['# SFNO / V3-stage3 matched experiments','',
        '| Dataset | Model | Scale | Seeds | PSNR | SAM | ERGAS |','|---|---|---|---|---|---|---|']
    for dataset in ('pavia','chikusei'):
        for model in ('sfno','v3_stage3'):
            for sf in (4,8):
                selected=[r for r in rows if (r['dataset'],r['model'],r['sf'])==(dataset,model,sf)]
                if not selected:continue
                mean={k:statistics.mean(r[k] for r in selected) for k in ('psnr','sam_degrees','ergas')}
                std={k:statistics.stdev(r[k] for r in selected) for k in mean} if len(selected)==3 else None
                groups.append(dict(dataset=dataset,model=model,sf=sf,n=len(selected),mean=mean,sample_std=std))
                values=[f'{mean[k]:.4f} +/- {std[k]:.4f}' if std else f'{mean[k]:.4f} (partial)' for k in mean]
                lines.append(f'| {dataset} | {model} | x{sf} | {len(selected)}/3 | '+' | '.join(values)+' |')
    save(ROOT/'summary.json',dict(completed_reports=len(rows),required_reports=24,groups=groups))
    (ROOT/'summary.md').write_text('\n'.join(lines)+'\n')
    return len(rows)


def recover_workers():
    scripts={str(root/script) for root in ROOTS.values() for script in ('run.py','validate.py')}
    while True:
        found=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                if any(a in scripts for a in proc_args(int(p.name))):found.append(int(p.name))
            except PermissionError:pass
        if not found:return
        state('recovering_existing_workers',worker_pids=found);time.sleep(20)


def run(plan,args,record):
    if record:wait_worker(record)
    if args.resume:recover_workers()
    for model,dataset,seed in JOBS:
        verify(model,plan);job=[model,dataset,seed];out=output(*job)
        if len(reports(*job,plan))==2:continue
        ensure_validation(model,plan)
        logs=ROOTS[model]/'Runs'/f'{dataset}_seed{seed}';logs.mkdir(parents=True,exist_ok=True)
        meta=out/'checkpoint_metadata.json';completed=json.loads(meta.read_text())['epoch'] if meta.exists() else 0
        if completed<600:
            if (out/'stop_reason.json').exists():raise RuntimeError('Prior quality stop requires review:'+str(out/'stop_reason.json'))
            if out.exists() and any(out.iterdir()) and not (out/'last.pth').exists():raise RuntimeError('Partial output lacks checkpoint:'+str(out))
            gpu_idle(job)
            command=['run.py','train','--dataset',dataset,'--seed',str(seed),'--output',str(out),'--device','cuda']
            if completed:command+=['--resume',str(out/'last.pth')]
            execute(model,command,logs/f'train_from_epoch{completed}_{time.time_ns()}.log','training',job)
        metadata=json.loads(meta.read_text())
        if metadata['epoch']!=600 or metadata['config']['protocol_sha256']!=plan['protocol_sha256'][model]:raise ValueError('Wrong final checkpoint')
        for sf in (4,8):
            if (out/f'last_test_x{sf}.json').exists():continue
            gpu_idle(job)
            execute(model,['run.py','test','--dataset',dataset,'--seed',str(seed),'--checkpoint',str(out/'last.pth'),
                           '--output',str(out),'--sf',str(sf),'--device','cuda'],logs/f'test_x{sf}_{time.time_ns()}.log','testing',job)
        if len(reports(*job,plan))!=2:raise RuntimeError('Missing x4/x8 reports')
        save(out/'complete.json',dict(model=model,dataset=dataset,seed=seed,epoch=600,scales=[4,8]))
        subprocess.run([sys.executable,str(ROOTS[model]/'run_queue.py'),'--summarize'],check=True,cwd=ROOTS[model])
        state('job_complete',job=job,completed_reports=summarize(plan),required_reports=24)
    if summarize(plan)!=24:raise RuntimeError('Missing final reports')
    state('complete',completed_jobs=12,completed_reports=24)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--detach',action='store_true');parser.add_argument('--adopt',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    plan=json.loads((ROOT/'queue_plan.json').read_text())
    if sha(Path(__file__))!=plan['queue_source_sha256']:raise ValueError('Queue code changed after freezing')
    for model in ROOTS:verify(model,plan)
    if args.check:
        print(json.dumps(dict(jobs=JOBS,protocols=plan['protocol_sha256'],existing_reports=summarize(plan)),indent=2));return
    if args.detach:
        if (ROOT/'status.json').exists() and not args.resume:raise FileExistsError('Existing combined queue')
        with (ROOT/f'execution_{time.time_ns()}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*(['--adopt'] if args.adopt else []),*(['--resume'] if args.resume else [])],
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        print('Combined queue PID',child.pid);return
    with (ROOT/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            record=adopt_current() if args.adopt else (json.loads((ROOT/'handoff.json').read_text()) if args.resume and (ROOT/'handoff.json').exists() else None)
            with ExitStack() as stack:
                # Retain each original dispatcher's lock to reject duplicate standalone queues.
                for root in ROOTS.values():
                    (root/'Runs').mkdir(exist_ok=True)
                    held=stack.enter_context((root/'Runs/lock').open('a'))
                    for attempt in range(30):
                        try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:
                            if attempt==29:raise
                            time.sleep(.1)
                run(plan,args,record)
        except Exception as exc:state('failed',error=str(exc));raise


if __name__=='__main__':main()

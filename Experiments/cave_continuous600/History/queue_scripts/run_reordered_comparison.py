"""One-time handoff: running CAVE stage3/43 -> Pavia stage3 -> CAVE seed44 pair."""
import fcntl,hashlib,importlib.util,json,math,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def confirm(pid,needle):
    if needle not in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError(f'Unexpected PID {pid}')

def stop_supervisor(pid):
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not alive(pid):return
        time.sleep(.1)
    raise RuntimeError(f'Supervisor did not exit: {pid}')

def cave_tests(c,model,seed):
    c.checkpoint_check(model,seed)
    target=c.OUT/f'{model}_seed{seed}';_,test,model_name,_=c.MODELS[model]
    for sf in (4,8):
        c.state('testing',model=model,seed=seed,sf=sf)
        suffix='' if sf==4 else '_x8'
        cmd=[sys.executable,'-u',test,'--model',model_name,'--data_path','Data/Test',
             '--checkpoint',target/'last.pth','--sf',sf,'--seed',1,'--report',target/f'metrics{suffix}.json']
        if sf==8:cmd+=['--cross-scale']
        c.execute(cmd,f'{model}_seed{seed}_test{suffix}.log');c.report(model,seed,sf)
    c.summarize()

def main():
    c=module('cave_three',ROOT/'run_three_seed_comparison.py')
    p=module('pavia_supervisor',ROOT.parent/'Pavia/train_and_test.py')
    p.MODEL='v3';p.ABLATION='deform_stage3'
    name='v3_deform_stage3_continuous600_x4_seed42'
    p.OUT=p.ROOT/'Checkpoint'/name;p.RUN=p.ROOT/'Runs'/name
    control=c.RUN/'reordered_status.json'
    if control.exists():raise FileExistsError(control)
    confirm(453129,b'run_three_seed_comparison.py');confirm(469551,b'--wait-for-cave')
    confirm(462143,b'--seed\x0043')
    status=json.loads((c.RUN/'status.json').read_text())
    assert status['phase']=='training' and status['model']=='deform_stage3' and status['seed']==43
    assert json.loads((p.RUN/'status.json').read_text())['phase']=='waiting_for_cave'
    for folder in [c.OUT/'baseline_seed44',c.OUT/'deform_stage3_seed44',p.OUT]:
        if folder.exists() and any(folder.iterdir()):raise FileExistsError(folder)
    snapshots={'cave':status,'pavia':json.loads((p.RUN/'status.json').read_text())}
    (c.RUN/'before_reorder.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    # Terminate only idle/waiting supervisors, never their process groups or training child.
    stop_supervisor(469551);stop_supervisor(453129)
    confirm(462143,b'Train_V3.py')
    control.write_text(json.dumps(dict(phase='handoff_complete',pid=os.getpid(),preserved_train_pid=462143,
        order=['CAVE stage3 seed43','Pavia stage3 seed42','CAVE baseline seed44','CAVE stage3 seed44']),indent=2)+'\n')
    with (c.RUN/'lock').open('w') as cl,(p.RUN/'lock').open('w') as pl:
        fcntl.flock(cl,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(pl,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            c.state('training',model='deform_stage3',seed=43,train_pid=462143,next_task='Pavia V3-stage3')
            p.state('waiting_for_cave_seed43',train_pid=462143)
            while alive(462143):time.sleep(15)
            cave_tests(c,'deform_stage3',43)
            c.state('waiting_for_pavia_stage3',next_task='baseline seed44')
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            p.state('training',gpu=torch.cuda.get_device_name(0),output=str(p.OUT))
            p.execute(['train','--device','cuda','--sf','4','--batch-size','1','--epochs','600',
                '--samples','1000','--lr','0.0001','--seed','42','--output',p.OUT],'train.log')
            for sf in (4,8):
                p.state('testing',sf=sf,output=str(p.OUT))
                p.execute(['test','--device','cuda','--sf',str(sf),'--checkpoint',p.OUT/'last.pth','--output',p.OUT],f'test_x{sf}.log')
                d=json.loads((p.OUT/f'last_test_x{sf}.json').read_text())
                assert d['epoch']==600 and d['sf']==sf and d['source_sf']==4 and d['ablation']=='deform_stage3' and len(d['patches'])==4
                assert all(math.isfinite(v) for v in d['mean'].values())
            p.state('complete',output=str(p.OUT))
            for model,(train,_,model_name,extra) in c.MODELS.items():
                target=c.OUT/f'{model}_seed44'
                c.state('training',model=model,seed=44,target=str(target))
                c.execute([sys.executable,'-u',train,'--model',model_name,'--data_path','Data/Train',
                    '--epochs',600,'--batch_size',1,'--patch_size',64,'--lr',1e-4,'--sf',4,
                    '--trainset_num',1000,'--num_workers',2,'--seed',44,'--save_every',600,
                    '--ablation-lr-schedule','--save_dir',target,*extra],f'{model}_seed44_train.log')
                cave_tests(c,model,44)
            assert c.summarize()['complete']
            for file,digest in json.loads((c.RUN/'seed42_sha256.json').read_text()).items():
                assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
            c.state('complete');control.write_text(json.dumps(dict(phase='complete',pid=os.getpid()),indent=2)+'\n')
        except Exception as exc:
            c.state('failed',error=str(exc))
            if json.loads((p.RUN/'status.json').read_text())['phase']!='complete':p.state('failed',error=str(exc))
            control.write_text(json.dumps(dict(phase='failed',error=str(exc)),indent=2)+'\n');raise

if __name__=='__main__':main()

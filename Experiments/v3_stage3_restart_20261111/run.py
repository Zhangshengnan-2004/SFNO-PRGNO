"""Train V3-stage3 from scratch with frozen protocol; final x4/x8 evaluation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from protocol import ROOT,PROTOCOL,DEFAULTS,load_protocol,output_dir,sha256,atomic_json
from data import load_cube,validate_sources,TrainingPatches,test_patches,observations
from metrics import evaluate,RULES
sys.path.insert(0,str(ROOT/'model'))
from SFNO_PRGNO_V3 import SFNO_PRGNO_V3
from matched_spatial import MatchedSpatial
from inference_memory import configure


def reproducible(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    # CUDA grid_sample and antialiased resize backward lack deterministic kernels.
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(4)


def build_model(spec,device):
    # Preserve every layer's constructor initialization, as in the original runner.
    response=np.load(ROOT/spec['response_file'],allow_pickle=False)
    model=SFNO_PRGNO_V3(size=64,hschannels=spec['hsi_channels'],mschannels=spec['msi_channels'],
                       response=response,ablation='deform_stage3',guidance_mode='physics',
                       use_spatial_residual=True,use_spectral_residual=True)
    model.physics.spatial=MatchedSpatial()
    configure(model)
    return model.to(device)


def training_loss(pred,gt):
    valid=gt.norm(dim=1)>1e-8
    count=valid.sum(dim=(1,2))
    if (count==0).any():raise ValueError('All-zero training sample')
    numerator=((pred-gt).abs()*valid[:,None]).sum(dim=(1,2,3))
    return (numerator/(count*gt.shape[1])).mean()


def rng_state(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if device.type!='cuda':raise ValueError('CUDA checkpoint requires CUDA resume')
        torch.cuda.set_rng_state_all(state['cuda'])


def config(args,spec):
    return dict(protocol=PROTOCOL,protocol_sha256=sha256(ROOT/'protocol.json'),dataset=args.dataset,model='v3_stage3',
                ablation='deform_stage3',guidance_mode='physics',seed=args.seed,
                epochs=args.epochs,samples=args.samples,micro_batch=args.micro_batch,effective_batch=args.effective_batch,
                lr=DEFAULTS['lr'],lr_gamma=DEFAULTS['lr_gamma'],lr_decay_epochs=DEFAULTS['lr_decay_epochs'],
                shared_pr_within_effective_batch=False,train_sf=4,test_seed=1,patch_size=64,
                quality_min_optimizer_steps=DEFAULTS['quality_min_optimizer_steps'],
                quality_patience=DEFAULTS['quality_patience'],
                precision='FP32',deterministic=False,deterministic_algorithms='warn_only; CUDA grid_sample backward nondeterministic',
                sdpa='math',tf32=False,workers=0,optimizer='Adam beta0.9,0.999 eps1e-8',
                loss='mean per-image GT-nonzero masked L1',initialization='native V3 constructors including zero heads/offsets; no blanket reinitialization',
                response_file=spec['response_file'],response_sha256=spec['response_sha256'],raw_sha256=spec['raw_sha256'])


def append_row(path,row):
    new=not path.exists()
    with path.open('a',newline='') as f:
        writer=csv.writer(f)
        if new:writer.writerow(['epoch','lr','L1','optimizer_steps','sampled_patches'])
        writer.writerow(row)


def learning_rate(cfg,epoch):
    return cfg['lr']*cfg['lr_gamma']**(epoch//cfg['lr_decay_epochs'])


def forward_rng_state(device):
    return (torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_forward_rng(state,device):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state(state[1],device)


def training_probes(cube,dataset,device):
    saved=rng_state(device)
    try:
        reproducible(1729);ds=TrainingPatches(cube,dataset,4)
        return [ds[i][None].to(device) for i in range(4)]
    finally:restore_rng(saved,device)


def measure_health(model,probes,response,device):
    saved=rng_state(device);was_training=model.training
    try:
        model.eval();rows=[]
        with torch.inference_mode():
            for gt in probes:
                low,msi=observations(gt,4,response);reproducible(1)
                pred=model(low,msi,4)
                bc=F.interpolate(low,size=gt.shape[-2:],mode='bicubic',align_corners=False)
                metric=evaluate(pred,gt,4);bicubic=evaluate(bc,gt,4)
                rows.append(dict(L1=float(training_loss(pred,gt)),bicubic_L1=float(training_loss(bc,gt)),
                                 psnr=metric['psnr'],bicubic_psnr=bicubic['psnr'],
                                 sam_degrees=metric['sam_degrees'],bicubic_sam_degrees=bicubic['sam_degrees'],
                                 ergas=metric['ergas'],bicubic_ergas=bicubic['ergas']))
        return {k:float(np.mean([v[k] for v in rows])) for k in rows[0]}
    finally:
        model.train(was_training);restore_rng(saved,device)


def convergence_policy(health,steps,failures,cfg):
    better=health['L1']<health['bicubic_L1'] and health['psnr']>health['bicubic_psnr']
    if steps<cfg['quality_min_optimizer_steps']:return failures,False,better
    failures=0 if better else failures+1
    return failures,failures>=cfg['quality_patience'],better


def train(args,p,spec,device):
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('Nonempty output; refusing overwrite')
    if args.resume and args.resume.resolve().parent!=out:raise ValueError('Resume output must be checkpoint directory')
    cfg=config(args,spec)
    validate_sources(spec);cube=load_cube(args.dataset)
    response=torch.from_numpy(np.load(ROOT/spec['response_file'],allow_pickle=False)).to(device)
    reproducible(args.seed)
    model=build_model(spec,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8)
    loader=DataLoader(TrainingPatches(cube,args.dataset,args.samples),batch_size=args.micro_batch,num_workers=0,shuffle=False)
    start=0;steps=0;health_failures=0
    out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        if ck['config']!=cfg:raise ValueError('Resume configuration changed')
        if ck['epoch']>=args.epochs:raise ValueError('Training already finished')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        start,steps=ck['epoch'],ck['optimizer_steps']
        health_failures=ck['health_failures']
        csv_path=out/'train.csv'
        if csv_path.exists():
            with csv_path.open() as f:rows=list(csv.reader(f))
            kept=[rows[0]]+[row for row in rows[1:] if int(row[0])<=start]
            if len(kept)==1 or int(kept[-1][0])<start:kept.append(ck['log_row'])
            with csv_path.open('w',newline='') as f:csv.writer(f).writerows(kept)
        else:append_row(csv_path,ck['log_row'])
        restore_rng(ck['rng'],device)
        atomic_json(out/f'resume_epoch{start}.json',dict(epoch=start,checkpoint=str(args.resume)))
        del ck
    else:
        atomic_json(out/'config.json',cfg)
        np.save(out/'response.npy',response.cpu().numpy())
    probes=training_probes(cube,args.dataset,device)
    health=measure_health(model,probes,response,device)
    if not args.resume:
        atomic_json(out/'health_initial.json',health)
        if health['L1']>2:raise RuntimeError('Abnormal initial output scale; queue stopped before training')
    for epoch in range(start,args.epochs):
        lr=learning_rate(cfg,epoch)
        for group in optimizer.param_groups:group['lr']=lr
        model.train();optimizer.zero_grad(set_to_none=True)
        total=0.;sample_count=0;group_count=0;started=time.monotonic()
        for gt in loader:
            gt=gt.to(device);low,msi=observations(gt,4,response)
            # PR shares one spatial permutation across a true batch. Reuse its
            # RNG across microbatches to preserve that behavior under accumulation.
            if group_count==0:shared_rng=forward_rng_state(device)
            else:restore_forward_rng(shared_rng,device)
            pred=model(low,msi,4)
            loss=training_loss(pred,gt)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            (loss*(len(gt)/args.effective_batch)).backward()
            total+=float(loss.detach())*len(gt);sample_count+=len(gt);group_count+=len(gt)
            if group_count==args.effective_batch:
                if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise RuntimeError('Nonfinite gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1;group_count=0
                if any(not torch.isfinite(v).all() for v in model.parameters()):
                    raise RuntimeError('Nonfinite parameter after Adam update')
                if sample_count%500==0:
                    progress=dict(phase='training',epoch=epoch+1,epochs=args.epochs,patches=sample_count,samples=args.samples,
                                  optimizer_steps=steps,lr=lr,L1=total/sample_count,elapsed_seconds=time.monotonic()-started)
                    atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if group_count or sample_count!=args.samples:raise RuntimeError('Incomplete effective batch')
        stop_reason=None
        if epoch==0 and total/sample_count>1:
            stop_reason='First epoch L1 >1: abnormal output scale'
        if steps>=500 and total/sample_count>.1:
            stop_reason='L1 >0.1 after500 Adam updates: severe training error'
        if (epoch+1)%20==0:
            health=measure_health(model,probes,response,device)
            eligible=steps>=cfg['quality_min_optimizer_steps']
            health_failures,quality_stop,better=convergence_policy(health,steps,health_failures,cfg)
            health_status=dict(epoch=epoch+1,optimizer_steps=steps,convergence_guard_active=eligible,
                               better_than_bicubic=better,consecutive_failures=health_failures,**health)
            atomic_json(out/f'health_epoch_{epoch+1}.json',health_status)
            atomic_json(out/'health_latest.json',health_status)
            if quality_stop:
                stop_reason=f'No Bicubic improvement in{health_failures} consecutive checks after warmup'
        row=[epoch+1,lr,total/sample_count,steps,(epoch+1)*args.samples]
        append_row(out/'train.csv',row)
        checkpoint=dict(protocol=PROTOCOL,epoch=epoch+1,optimizer_steps=steps,health_failures=health_failures,model=model.state_dict(),
                        optimizer=optimizer.state_dict(),config=cfg,rng=rng_state(device),log_row=row)
        tmp=out/'last.tmp';torch.save(checkpoint,tmp);tmp.replace(out/'last.pth')
        if (epoch+1)%DEFAULTS['save_every']==0 or epoch+1==args.epochs:
            import shutil
            shutil.copyfile(out/'last.pth',out/f'epoch_{epoch+1}.pth')
        atomic_json(out/'checkpoint_metadata.json',dict(epoch=epoch+1,optimizer_steps=steps,health_failures=health_failures,config=cfg))
        print(f'epoch={epoch+1}/{args.epochs} lr={lr:.9g} L1={total/sample_count:.7f} optimizer_steps={steps}',flush=True)
        if stop_reason:
            atomic_json(out/'stop_reason.json',dict(reason=stop_reason,epoch=epoch+1,optimizer_steps=steps,
                                                   checkpoint_saved=True,health_failures=health_failures))
            raise RuntimeError(stop_reason+'; checkpoint saved and queue stopped')
        if args.stop_after and epoch+1>=args.stop_after:break
    if epoch+1==args.epochs:atomic_json(out/'progress.json',dict(phase='trained',epoch=epoch+1,optimizer_steps=steps))


def test(args,p,spec,device):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    target=out/f'last_test_x{args.sf}.json'
    if target.exists():raise FileExistsError(target)
    before=sha256(args.checkpoint)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['config']
    if ck['protocol']!=PROTOCOL or cfg['protocol_sha256']!=sha256(ROOT/'protocol.json') or cfg['dataset']!=args.dataset:
        raise ValueError('Checkpoint protocol/dataset mismatch')
    if ck['epoch']!=cfg['epochs']:raise ValueError('Only final fixed-budget checkpoint may be evaluated')
    validate_sources(spec);cube=load_cube(args.dataset)
    reproducible(DEFAULTS['test_seed'])
    model=build_model(spec,device);model.load_state_dict(ck['model']);model.eval()
    epoch=ck['epoch'];del ck
    response=torch.from_numpy(np.load(ROOT/spec['response_file'],allow_pickle=False)).to(device)
    # Native SFNO PR samples in eval mode. Same seed and patch order for every training seed.
    reproducible(DEFAULTS['test_seed']);rows=[]
    with torch.inference_mode():
        for geometry,gt,valid in test_patches(cube,args.dataset,args.sf):
            gt=gt.to(device);low,msi=observations(gt,args.sf,response)
            pred=model(low,msi,args.sf)
            metrics=evaluate(pred,gt,args.sf,valid)
            bicubic=F.interpolate(low,size=gt.shape[-2:],mode='bicubic',align_corners=False)
            kept=valid&(gt.norm(dim=1)[0].cpu().numpy()>1e-8)
            rows.append(dict(**geometry,valid_pixels=int(kept.sum()),total_pixels=int(valid.size),
                             **metrics,bicubic=evaluate(bicubic,gt,args.sf,valid),
                             ergas_fixed4_diagnostic=metrics['ergas']*args.sf/4))
            print(f'[{geometry["index"]}] PSNR={metrics["psnr"]:.4f} SAM={metrics["sam_degrees"]:.4f} ERGAS={metrics["ergas"]:.4f}',flush=True)
    if sha256(args.checkpoint)!=before:raise ValueError('Checkpoint changed during evaluation')
    report=dict(protocol=PROTOCOL,protocol_sha256=sha256(ROOT/'protocol.json'),checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=before,epoch=epoch,dataset=args.dataset,seed=cfg['seed'],model='v3_stage3',source_sf=4,sf=args.sf,
                test_seed=DEFAULTS['test_seed'],ablation='deform_stage3',rules=RULES,patches=rows,
                mean={k:float(np.mean([r[k] for r in rows])) for k in ('psnr','sam_degrees','ergas')})
    report['bicubic_mean']={k:float(np.mean([r['bicubic'][k] for r in rows])) for k in report['mean']}
    report['comparison']=dict(psnr_gain=report['mean']['psnr']-report['bicubic_mean']['psnr'],
                              sam_reduction=report['bicubic_mean']['sam_degrees']-report['mean']['sam_degrees'],
                              ergas_reduction=report['bicubic_mean']['ergas']-report['mean']['ergas'])
    report['better_than_bicubic_all_metrics']=all(v>0 for v in report['comparison'].values())
    atomic_json(target,report)
    lines=[f'Checkpoint: {report["checkpoint"]}',f'Epoch: {epoch}',f'Train seed: {cfg["seed"]}; test seed: 1',
           f'Scale: {args.sf}; patches: {len(rows)}']
    for r in rows:lines.append(f'[{r["index"]}] PSNR={r["psnr"]:.4f} SAM={r["sam_degrees"]:.4f} ERGAS={r["ergas"]:.4f}')
    lines += ['',f'Bicubic PSNR = {report["bicubic_mean"]["psnr"]:.4f}; SAM = {report["bicubic_mean"]["sam_degrees"]:.4f}; ERGAS = {report["bicubic_mean"]["ergas"]:.4f}',
              f'PSNR gain over Bicubic = {report["comparison"]["psnr_gain"]:.4f} dB',
              f'Avg PSNR = {report["mean"]["psnr"]:.4f}',f'Avg SAM = {report["mean"]["sam_degrees"]:.4f}',f'Avg ERGAS = {report["mean"]["ergas"]:.4f}']
    with (out/('test.log' if args.sf==4 else 'test_x8.log')).open('x') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[-3:]),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=['train','test'])
    parser.add_argument('--dataset',choices=['pavia','chikusei'],required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=DEFAULTS['epochs'])
    parser.add_argument('--samples',type=int,default=DEFAULTS['samples'])
    parser.add_argument('--micro-batch',type=int,default=DEFAULTS['micro_batch'])
    parser.add_argument('--effective-batch',type=int,default=DEFAULTS['effective_batch'])
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--sf',type=int,choices=[4,8],default=4)
    parser.add_argument('--stop-after',type=int,help='Validation only: stop after absolute epoch while retaining full configuration')
    args=parser.parse_args()
    if min(args.epochs,args.samples,args.micro_batch,args.effective_batch)<=0:parser.error('Positive sizes required')
    if args.samples%args.effective_batch or args.effective_batch%args.micro_batch:parser.error('Samples and effective batch must divide exactly')
    if args.mode=='train' and (args.checkpoint or args.sf!=4):parser.error('Train from scratch or --resume; train scale fixed4')
    if args.mode=='test' and (not args.checkpoint or args.resume):parser.error('Test requires --checkpoint, not --resume')
    args.output=args.output or output_dir(args.dataset,args.seed)
    p=load_protocol();spec=p['datasets'][args.dataset];device=torch.device(args.device)
    if args.mode=='train':train(args,p,spec,device)
    else:test(args,p,spec,device)


if __name__=='__main__':main()

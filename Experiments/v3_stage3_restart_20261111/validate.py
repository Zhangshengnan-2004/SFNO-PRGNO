"""Technical preflight only; convergence is monitored in the six persistent formal jobs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import subprocess
import sys
import time
import argparse
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from protocol import ROOT,DEFAULTS,load_protocol,sha256,atomic_json
from data import load_cube,validate_sources,TrainingPatches,test_patches,observations
import run
from run import reproducible,build_model,training_loss,learning_rate,forward_rng_state,restore_forward_rng,convergence_policy
from metrics import evaluate
from data import degrade
from matched_spatial import MatchedSpatial


def check_spatial():
    reproducible(7);op=MatchedSpatial();errors=[]
    for sf in (4,8):
        x=torch.randn(1,3,32,32,dtype=torch.float64)
        y=torch.randn(1,3,32//sf,32//sf,dtype=torch.float64)
        forward_error=float((op(x,y.shape[-2:])-degrade(x,sf)).abs().max())
        adjoint_error=float(((op(x,y.shape[-2:])*y).sum()-(x*op.adjoint(y,x.shape[-2:])).sum()).abs())
        assert forward_error<1e-10 and adjoint_error<1e-10
        warmed=MatchedSpatial()
        with torch.inference_mode():warmed(x,y.shape[-2:])
        assert all(not torch.is_inference(v) for v in warmed.cache.values())
        differentiable=x.clone().requires_grad_(True)
        warmed(differentiable,y.shape[-2:]).square().mean().backward()
        assert torch.isfinite(differentiable.grad).all()
        errors.append(dict(sf=sf,forward_max_error=forward_error,adjoint_dot_error=adjoint_error,inference_cache_then_backward=True))
    return errors


def probe(device_name):
    p=load_protocol();device=torch.device(device_name);details=[];spatial=check_spatial()
    for dataset in ('pavia','chikusei'):
        spec=p['datasets'][dataset];reproducible(42);model=build_model(spec,device)
        gt=torch.rand(1,spec['hsi_channels'],64,64,device=device)*.5
        r=torch.from_numpy(np.load(ROOT/spec['response_file'])).to(device)
        assert torch.equal(model.physics.spectral.response,r)
        assert [b.MDG.ablation for b in (model.block1,model.block2,model.block3)]==['no_deformable','no_deformable','full']
        low,msi=observations(gt,4,r)
        model.eval()
        with torch.inference_mode():model(low,msi,4)
        assert all(not torch.is_inference(v) for v in model.physics.spatial.cache.values())
        model.train();pred=model(low,msi,4);loss=training_loss(pred,gt);loss.backward()
        assert torch.isfinite(loss) and float(loss)<2
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        assert model.block1.MDG.deformable.offset.weight.grad is None
        assert model.block2.MDG.deformable.offset.weight.grad is None
        active=float(model.block3.MDG.deformable.offset.weight.grad.norm())
        assert active>0
        details.append(dict(dataset=dataset,L1=float(loss.detach()),stage3_offset_gradient_norm=active,inference_then_training_backward=True))
        del model,gt,r,low,msi,pred,loss
        if device.type=='cuda':torch.cuda.empty_cache()
    atomic_json(ROOT/f'{device_name}_probe.json',dict(status='passed',protocol_sha256=sha256(ROOT/'protocol.json'),spatial_operator=spatial,details=details))
    print(device_name,'V3-stage3 forward/backward and matched spatial/adjoint probe passed',flush=True)


def check_policy():
    cfg=DEFAULTS
    bad=dict(L1=.0133,bicubic_L1=.0140,psnr=35.,bicubic_psnr=36.)
    for steps in (200,400,800,1600):
        assert convergence_policy(bad,steps,0,cfg)==(0,False,False)
    failures=0
    for steps,expected in ((2000,False),(2400,False),(2800,True)):
        failures,stop,better=convergence_policy(bad,steps,failures,cfg)
        assert stop==expected and not better
    good=dict(L1=.01,bicubic_L1=.0140,psnr=37.,bicubic_psnr=36.)
    assert convergence_policy(good,2400,2,cfg)==(0,False,True)
    identity=torch.rand(1,3,4,4)
    assert evaluate(identity,identity,4)['ergas']==0
    assert evaluate(torch.zeros_like(identity),identity,4)['sam_degrees']==90


def check_sfno_accumulation(model,gt,response):
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert not any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules())
    model.train();model.zero_grad(set_to_none=True)
    low,msi=observations(gt,4,response);state=forward_rng_state(gt.device)
    training_loss(model(low,msi,4),gt).backward()
    reference={k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None}
    after=forward_rng_state(gt.device);model.zero_grad(set_to_none=True)
    for start in (0,2):
        restore_forward_rng(state,gt.device)
        (training_loss(model(low[start:start+2],msi[start:start+2],4),gt[start:start+2])/2).backward()
    delta=sum((v.grad-reference[k]).double().square().sum() for k,v in model.named_parameters() if k in reference)
    denom=sum(v.double().square().sum() for v in reference.values())
    relative=float((delta/denom).sqrt());assert relative<1e-4,relative
    current=forward_rng_state(gt.device)
    assert torch.equal(current[0],after[0]) and torch.equal(current[1],after[1])
    model.zero_grad(set_to_none=True)
    return dict(true_reference_batch=4,micro_reference_batch=2,gradient_relative_l2_error=relative,pr_rng_equivalent=True)


def check_saved_stop(p):
    # Exercise the production save-before-stop path with a deliberately bad lightweight model.
    class BadScaleModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.tensor(1.5))
        def forward(self,low,msi,sf):
            return self.bias.expand(low.shape[0],102,*msi.shape[-2:])
    out=ROOT/'Preflight'/f'forced_stop_{time.time_ns()}'
    args=SimpleNamespace(output=out,resume=None,dataset='pavia',seed=42,epochs=2,samples=50,
                         micro_batch=5,effective_batch=50,stop_after=None)
    try:
        with patch.object(run,'build_model',side_effect=lambda spec,device:BadScaleModel().to(device)):
            run.train(args,p,p['datasets']['pavia'],torch.device('cuda'))
    except RuntimeError as exc:
        assert 'First epoch L1 >1' in str(exc),str(exc)
    else:raise AssertionError('Expected bad-scale run to stop')
    reason=json.loads((out/'stop_reason.json').read_text())
    ck=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
    assert reason['checkpoint_saved'] and ck['epoch']==1 and ck['optimizer_steps']==1
    return dict(path=str(out),epoch_saved=1,forced_bad_loss_stopped=True)


def main():
    p=load_protocol();check_policy();details=[]
    assert DEFAULTS['epochs']==600 and DEFAULTS['samples']==1000
    assert DEFAULTS['micro_batch']==5 and DEFAULTS['effective_batch']==50
    assert DEFAULTS['epochs']*DEFAULTS['samples']//DEFAULTS['effective_batch']==12000
    for epoch in range(600):assert learning_rate(DEFAULTS,epoch)==2e-4*.95**(epoch//6)
    print('Budget, LR boundaries and update-based convergence policy passed',flush=True)
    for dataset in ('pavia','chikusei'):
        spec=p['datasets'][dataset];validate_sources(spec);cube=load_cube(dataset)
        reproducible(42);model=build_model(spec,torch.device('cuda'))
        optimizer=torch.optim.Adam(model.parameters(),lr=DEFAULTS['lr'])
        ds=TrainingPatches(cube,dataset,50);r=torch.from_numpy(np.load(ROOT/spec['response_file'])).cuda()
        initial=[];torch.cuda.reset_peak_memory_stats()
        for group in range(10):
            gt=torch.stack([ds[group*5+i] for i in range(5)]).cuda()
            if group==0:shared_rng=forward_rng_state(gt.device)
            else:restore_forward_rng(shared_rng,gt.device)
            low,msi=observations(gt,4,r);pred=model(low,msi,4);loss=training_loss(pred,gt)
            assert torch.isfinite(loss) and float(loss)<2;initial.append(float(loss));(loss/10).backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert all(torch.isfinite(v).all() for v in model.parameters())
        peak=int(torch.cuda.max_memory_allocated());del gt,low,msi,pred,loss,optimizer
        accumulation=check_sfno_accumulation(model,torch.stack([ds[i] for i in range(4)]).cuda(),r)
        before=run.rng_state(torch.device('cuda'));was_training=model.training
        probes=run.training_probes(cube,dataset,torch.device('cuda'))
        health=run.measure_health(model,probes,r,torch.device('cuda'))
        after=run.rng_state(torch.device('cuda'))
        assert before['python']==after['python'] and torch.equal(before['torch'],after['torch'])
        assert before['numpy'][0]==after['numpy'][0] and np.array_equal(before['numpy'][1],after['numpy'][1]) and before['numpy'][2:]==after['numpy'][2:]
        assert all(torch.equal(a,b) for a,b in zip(before['cuda'],after['cuda']))
        assert model.training==was_training
        del probes
        model.eval()
        for sf in (4,8):
            _,gt,valid=next(test_patches(cube,dataset,sf));gt=gt.cuda();low,msi=observations(gt,sf,r)
            with torch.inference_mode():
                reproducible(1);pred=model(low,msi,sf)
                reproducible(1);again=model(low,msi,sf)
            assert pred.shape==gt.shape and torch.isfinite(pred).all() and torch.equal(pred,again)
            assert all(np.isfinite(v) for v in evaluate(pred,gt,sf,valid).values())
            print(dataset,sf,'whole image deterministic inference passed',tuple(pred.shape),flush=True)
            del gt,low,msi,pred,again
        details.append(dict(dataset=dataset,micro_batch=5,accumulation=10,finite_effective_batch50_update=True,
                            sfno_gradient_equivalence=accumulation,monitor_rng_and_mode_preserved=True,
                            initial_micro_losses=initial,peak_training_allocated_bytes=peak,technical_probe=health))
        del model,cube,ds,r;torch.cuda.empty_cache()
    # Persist all preflight outputs so an interrupted check never deletes evidence.
    folder=ROOT/'Preflight'/f'resume_{time.time_ns()}';full=folder/'full';split=folder/'split'
    base=[sys.executable,'-u',str(ROOT/'run.py'),'train','--dataset','pavia','--seed','42',
          '--epochs','2','--samples','50','--micro-batch','5','--effective-batch','50','--device','cuda']
    for extra in (['--output',str(full)],['--output',str(split),'--stop-after','1'],
                  ['--output',str(split),'--resume',str(split/'last.pth')]):
        subprocess.run(base+extra,check=True,cwd=ROOT)
    a=torch.load(full/'last.pth',map_location='cpu',weights_only=False)
    b=torch.load(split/'last.pth',map_location='cpu',weights_only=False)
    assert a['epoch']==b['epoch']==2 and a['optimizer_steps']==b['optimizer_steps']==2
    assert a['health_failures']==b['health_failures']==0
    differences=[]
    for k in a['model']:
        av,bv=a['model'][k],b['model'][k]
        assert torch.allclose(av,bv,atol=2e-5,rtol=2e-4),k
        differences.append(float((av-bv).abs().max()))
    for i in a['optimizer']['state']:
        for k,v in a['optimizer']['state'][i].items():
            assert torch.allclose(v,b['optimizer']['state'][i][k],atol=2e-5,rtol=2e-4) if isinstance(v,torch.Tensor) else v==b['optimizer']['state'][i][k]
    print('V3 model/Adam resume equivalence within tolerance passed; max weight diff',max(differences),flush=True)
    for sf in (4,8):
        subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'test','--dataset','pavia','--checkpoint',str(full/'last.pth'),
                        '--output',str(full),'--sf',str(sf),'--device','cuda'],check=True,cwd=ROOT)
        report=json.loads((full/f'last_test_x{sf}.json').read_text());assert len(report['patches'])==4 and report['epoch']==2
    stop_check=check_saved_stop(p)
    report=dict(status='passed',protocol_sha256=sha256(ROOT/'protocol.json'),technical_preflight_passed=True,
                budget_600x1000=True,optimizer_updates_per_seed=12000,paper_known_optimizer_settings=True,
                shared_pr_accumulation_verified=True,runtime_quality_guards=True,
                convergence_policy_warmup_and_patience_verified=True,save_before_stop_verified=stop_check,
                training_convergence_not_yet_assessed=True,
                resume_model_and_adam_within_tolerance=True,resume_max_weight_difference=max(differences),
                resume_tolerance=dict(atol=2e-5,rtol=2e-4),full_test_entry_and_logs=True,persistent_preflight_path=str(folder),details=details)
    atomic_json(ROOT/'validation.json',report);print('All technical preflight checks passed; six formal runs can start',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',choices=['cpu','cuda']);args=parser.parse_args()
    try:
        if args.probe:probe(args.probe)
        else:
            check_spatial();main()
    except Exception as exc:
        import traceback
        atomic_json(ROOT/'validation.json',dict(status='failed',protocol_sha256=sha256(ROOT/'protocol.json'),error=str(exc),traceback=traceback.format_exc()))
        raise

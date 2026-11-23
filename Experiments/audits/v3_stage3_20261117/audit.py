"""Read-only audit of frozen V3-stage3 experiments. Writes only beside this script."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import sys, json, csv, math, hashlib, argparse, traceback, re, statistics
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[2]
ROOT=PROJECT/'Experiments/v3_stage3_restart_20261111'
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import run
from protocol import load_protocol, sha256, DEFAULTS
from data import degrade, observations, load_cube, validate_sources, TrainingPatches, test_patches
from matched_spatial import MatchedSpatial
from DynamicSpatial import DynamicSpatial
from VectorAttention import VectorAttention
from inference_memory import configure
from metrics import evaluate, RULES
from PhysicsGalerkin import PhysicsGuidedGalerkin


def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def close(a,b,atol=1e-6,rtol=1e-5):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
def hash_tensor(t):return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
def intersection(a,b):
 y,x,h,w=a;v,u,k,l=b
 return max(0,min(y+h,v+k)-max(y,v))*max(0,min(x+w,u+l)-max(x,u))


def inventory(p):
 result={};failures=[]
 baseline=json.loads((PROJECT/'Experiments/sfno_restart_20261111/protocol.json').read_text())
 assert p['defaults']==baseline['defaults']
 assert p['datasets']==baseline['datasets']
 for name in ('data.py','metrics.py'):
  assert (ROOT/name).read_bytes()==(PROJECT/'Experiments/sfno_restart_20261111'/name).read_bytes(),name
 result['comparison']={'same_data_protocol':True,'same_metric_source':True,'same_defaults':True}
 result['datasets']={}
 for ds,spec in p['datasets'].items():
  validate_sources(spec)
  r=np.load(ROOT/spec['response_file'],allow_pickle=False)
  assert r.shape==(spec['msi_channels'],spec['hsi_channels']) and np.isfinite(r).all()
  assert (r>=0).all()
  train=spec['train_rectangles_yxhw'];tests=[[*yx,spec['test_size'][str(sf)],spec['test_size'][str(sf)]] for sf in (4,8) for yx in spec['test_origins_yx']]
  assert all(intersection(a,b)==0 for a in train for b in tests)
  result['datasets'][ds]={'hsi_channels':spec['hsi_channels'],'msi_channels':spec['msi_channels'],'response_row_sums':r.sum(1).tolist(),'response_hash_checked':True,'raw_hash_checked':True,'train_test_overlap_pixels':0}
 groups=[]
 report_rows=[]
 for folder in sorted((ROOT/'Checkpoint').iterdir()):
  if not folder.is_dir():continue
  cfg=json.loads((folder/'config.json').read_text());ds=cfg['dataset'];seed=cfg['seed']
  assert cfg['ablation']=='deform_stage3' and cfg['guidance_mode']=='physics'
  assert cfg['protocol_sha256']==sha256(ROOT/'protocol.json')
  rows=list(csv.DictReader((folder/'train.csv').open()))
  assert [int(x['epoch']) for x in rows]==list(range(1,601))
  for i,row in enumerate(rows):
   assert all(math.isfinite(float(x)) for x in row.values())
   assert abs(float(row['lr'])-run.learning_rate(cfg,i))<1e-15
   assert int(row['optimizer_steps'])==(i+1)*20
   assert int(row['sampled_patches'])==(i+1)*1000
  health_files=list(folder.glob('health_epoch_*.json'))
  assert len(health_files)==30
  for f in health_files:
   health=json.loads(f.read_text())
   assert all(math.isfinite(v) for v in health.values() if isinstance(v,(float,int)))
   assert health['optimizer_steps']==health['epoch']*20
   assert not health['consecutive_failures']
  assert not (folder/'stop_reason.json').exists()
  trainlogs=list((ROOT/'Runs'/f'{ds}_seed{seed}').glob('train_from_epoch0_*.log'))
  assert len(trainlogs)==1
  txt=trainlogs[0].read_text()
  problems=[ln for ln in txt.splitlines() if any(x in ln for x in ['Traceback','RuntimeError','Nonfinite','CUDA out of memory'])]
  assert not problems
  epoch_entries=re.findall(r'^epoch=(\d+)/600 lr=([\d.e+\-]+) L1=([\d.e+\-]+) optimizer_steps=(\d+)$',txt,re.M)
  assert [int(x[0]) for x in epoch_entries]==list(range(1,601))
  checkpoint_hash=sha256(folder/'last.pth')
  ck=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
  assert ck['config']==cfg and ck['epoch']==600 and ck['optimizer_steps']==12000
  assert ck['log_row']==[600,float(rows[-1]['lr']),float(rows[-1]['L1']),12000,600000]
  assert all(torch.isfinite(v).all() for v in ck['model'].values() if v.is_floating_point())
  response=ck['model']['physics.spectral.response']
  assert torch.equal(response,torch.from_numpy(np.load(ROOT/p['datasets'][ds]['response_file'])))
  steps=[]
  for state in ck['optimizer']['state'].values():
   for k,v in state.items():
    if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
   steps.append(int(state['step']))
  assert set(steps)=={12000}
  assert {'python','numpy','torch','cuda'}<=ck['rng'].keys()
  net=run.build_model(p['datasets'][ds],torch.device('cpu'))
  net.load_state_dict(ck['model'],strict=True)
  modes=[b.MDG.ablation for b in (net.block1,net.block2,net.block3)]
  assert modes==['no_deformable','no_deformable','full']
  g=float(net.block3.MDG.correction_logit.detach().sigmoid())
  assert torch.equal(ck['model']['block1.MDG.correction_logit'],torch.tensor(-3.))
  assert torch.equal(ck['model']['block2.MDG.correction_logit'],torch.tensor(-3.))
  learn={k:float(ck['model'][k].norm()) for k in ['intermediate_reconstruction.weight','physics.lift.weight','block1.GI.delta_qk.weight','block2.GI.delta_qk.weight','block3.GI.delta_qk.weight','block3.MDG.deformable.offset.weight','block3.MDG.deformable.weight.weight']}
  assert all(v>0 for v in learn.values())
  final=json.loads((folder/'health_latest.json').read_text())
  for sf in (4,8):
   report=json.loads((folder/f'last_test_x{sf}.json').read_text())
   assert report['checkpoint_sha256']==checkpoint_hash
   assert report['seed']==seed and report['test_seed']==1 and report['epoch']==600
   assert report['sf']==sf and report['source_sf']==4 and report['ablation']=='deform_stage3'
   assert report['protocol_sha256']==sha256(ROOT/'protocol.json') and report['rules']==RULES
   assert len(report['patches'])==(4 if ds=='pavia' else 8)
   for key in report['mean']:
    assert abs(report['mean'][key]-np.mean([x[key] for x in report['patches']]))<1e-12
   for i,entry in enumerate(report['patches']):
    assert [entry['y'],entry['x']]==p['datasets'][ds]['test_origins_yx'][i]
    assert entry['size']==p['datasets'][ds]['test_size'][str(sf)]
    assert 0<entry['valid_pixels']<=entry['total_pixels']==entry['size']**2
   assert report['better_than_bicubic_all_metrics']
   report_rows.append({'dataset':ds,'seed':seed,'sf':sf,**report['mean']})
  group={'name':folder.name,'seed':seed,'dataset':ds,'epoch':600,'updates':12000,'patches':600000,'first_l1':float(rows[0]['L1']),'last_l1':float(rows[-1]['L1']),'min_l1':min(float(r['L1']) for r in rows),'last_lr':float(rows[-1]['lr']),'last100_l1_mean':statistics.mean(float(r['L1']) for r in rows[-100:]),'health_psnr':final['psnr'],'health_bicubic_psnr':final['bicubic_psnr'],'health_checks':30,'fatal_errors':problems,'resumed':bool(list(folder.glob('resume_epoch*.json'))),'checkpoint_sha256':checkpoint_hash,'stage3_gate':g,'optimizer_active_tensor_count':len(steps),'learned_weight_norms':learn,'nondeterminism_warnings':sorted(set(re.findall(r'UserWarning: ([^\n]+)',txt)))}
  groups.append(group)
  print('inventory',folder.name,'passed',flush=True)
  del ck,net
 assert len(groups)==6
 for group in json.loads((ROOT/'Runs/summary.json').read_text())['groups']:
  selected=[r for r in report_rows if r['dataset']==group['dataset'] and r['sf']==group['sf']]
  assert len(selected)==3
  for k in group['mean']:
   assert abs(statistics.mean(r[k] for r in selected)-group['mean'][k])<1e-12
   assert abs(statistics.stdev(r[k] for r in selected)-group['sample_std'][k])<1e-12
 result['runs']=groups;result['test_reports']=report_rows
 result['unclipped_claim_mismatch']=('unclipped results also reported' in RULES['prediction'] and 'clip=False' not in (ROOT/'run.py').read_text())
 result['source_comparison']={f.name:(f.read_bytes()==(PROJECT/'CAVE/Model'/f.name).read_bytes()) for f in (ROOT/'model').glob('*.py') if (PROJECT/'CAVE/Model'/f.name).exists()}
 return result


def spatial_checks(device):
 entries=[]
 for sf in (4,8):
  for h,w in [(32,40),(64,64)]:
   torch.manual_seed(123)
   x=torch.randn(1,4,h,w,dtype=torch.float64,device=device,requires_grad=True)
   y=torch.randn(1,4,h//sf,w//sf,dtype=torch.float64,device=device)
   op=MatchedSpatial().to(device)
   with torch.inference_mode():op(x.detach(),y.shape[-2:])
   assert all(not torch.is_inference(v) for v in op.cache.values())
   value=op(x,y.shape[-2:]);ref=degrade(x,sf)
   err=float((value-ref).abs().max().detach())
   adj=float(((value*y).sum()-(x*op.adjoint(y,x.shape[-2:])).sum()).abs().detach())
   assert err<1e-10 and adj<1e-9
   value.square().mean().backward();assert torch.isfinite(x.grad).all()
   entries.append(dict(sf=sf,hr=[h,w],forward_max_error=err,adjoint_dot_error=adj,inference_cache_then_backward=True))
 return entries


def runtime(p,device):
 results={'device':str(device),'torch':torch.__version__,'spatial':spatial_checks(device),'datasets':{}}
 if device.type=='cuda':results['gpu']=torch.cuda.get_device_name(device)
 for ds,spec in p['datasets'].items():
  torch.manual_seed(42)
  net=run.build_model(spec,device)
  ck=torch.load(ROOT/'Checkpoint'/f'{ds}_v3_stage3_x4_seed42'/'last.pth',map_location='cpu',weights_only=False)
  net.load_state_dict(ck['model'],strict=True);del ck
  response=net.physics.spectral.response
  calls={};handles=[]
  def hook(name):
   def call(mod,args,out):calls[name]=calls.get(name,0)+1
   return call
  for j,b in enumerate((net.block1,net.block2,net.block3),1):
   for n,mod in [('regional',b.MDG.regional),('deformable',b.MDG.deformable),('GI',b.GI)]:handles.append(mod.register_forward_hook(hook(f'{j}.{n}')))
  scale_checks=[]
  for sf in (4,8):
   calls.clear();net.eval()
   gt=torch.rand(1,spec['hsi_channels'],16,24,device=device)*.6
   low,msi=observations(gt,sf,response)
   with torch.inference_mode():pred=net(low,msi,sf)
   assert pred.shape==gt.shape and torch.isfinite(pred).all()
   for j in (1,2,3):
    assert calls[f'{j}.regional']==calls[f'{j}.GI']==1
    assert calls.get(f'{j}.deformable',0)==int(j==3)
   scale_checks.append({'sf':sf,'shape':list(pred.shape),'calls':dict(calls)})
  for h in handles:h.remove()
  net.train();net.zero_grad(set_to_none=True)
  loss=run.training_loss(net(low,msi,8),gt);loss.backward()
  assert torch.isfinite(loss)
  assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
  gradnorms={}
  for j,b in enumerate((net.block1,net.block2,net.block3),1):
   for name,mod in [('offset',b.MDG.deformable.offset),('sampling_weight',b.MDG.deformable.weight),('physics_qk',b.GI.delta_qk)]:
    value=mod.weight.grad
    gradnorms[f'{j}.{name}']=None if value is None else float(value.norm())
  assert gradnorms['1.offset'] is None and gradnorms['2.offset'] is None and gradnorms['3.offset']>0
  assert all(gradnorms[f'{j}.physics_qk']>0 for j in (1,2,3))
  # Actual V3 accumulation, with four identical precomputed samples in each comparison.
  gt=torch.rand(4,spec['hsi_channels'],16,16,device=device)*.6
  low,msi=observations(gt,4,response)
  net.zero_grad(set_to_none=True)
  run.training_loss(net(low,msi,4),gt).backward()
  reference={n:v.grad.detach().clone() for n,v in net.named_parameters() if v.grad is not None}
  net.zero_grad(set_to_none=True)
  for start in (0,2): (run.training_loss(net(low[start:start+2],msi[start:start+2],4),gt[start:start+2])/2).backward()
  numerator=sum((v.grad-reference[n]).double().square().sum() for n,v in net.named_parameters() if n in reference)
  denominator=sum(v.double().square().sum() for v in reference.values())
  rel=float((numerator/denominator).sqrt());assert rel<1e-4,rel
  # The per-image mask normalizes each sample first; accumulation must respect sample count.
  manual=[]
  pred=net(low,msi,4)
  for j in range(4):
   mask=gt[j].norm(dim=0)>1e-8
   manual.append((pred[j,:,mask]-gt[j,:,mask]).abs().mean())
  close(run.training_loss(pred,gt),torch.stack(manual).mean())
  va=net.VI1_1.eval();a=torch.randn(1,spec['hsi_channels']+spec['msi_channels'],130,7,device=device)
  with torch.no_grad():native=va._original_forward(a);chunked=va(a)
  maxerr=float((native-chunked).abs().max());close(native,chunked,atol=2e-6,rtol=1e-5)
  net.zero_grad(set_to_none=True)
  results['datasets'][ds]={'scales':scale_checks,'backward_finite':True,'stage_gradients':gradnorms,'gradient_accumulation_relative_l2':rel,'masked_l1_matches_manual':True,'chunked_vector_attention_max_error':maxerr}
  print('runtime',device,ds,'passed',flush=True)
  del net,gt,low,msi,pred,reference,va,a,native,chunked
  if device.type=='cuda':torch.cuda.empty_cache()
 return results


def general_cache_check():
 torch.manual_seed(7)
 op=DynamicSpatial();x=torch.rand(1,3,16,16)
 with torch.inference_mode():op(x,(4,4))
 count=sum(torch.is_inference(t) for otf,entries in op._cache.values() for t in [otf]+[v for a in entries for v in a])
 try:op(x.clone().requires_grad_(True),(4,4)).square().mean().backward()
 except RuntimeError as exc:
  assert 'Inference tensors cannot be saved for backward' in str(exc)
  return {'bug_reproduced':True,'inference_tensors_in_cache':count,'error':str(exc),'used_in_current_restart':False}
 return {'bug_reproduced':False,'inference_tensors_in_cache':count,'used_in_current_restart':False}


def cave_checks():
 root=PROJECT/'Experiments/cave_continuous600'
 records=[]
 for seed in (42,43,44):
  folder=root/'Checkpoint'/f'cave_v3_stage3_x4_seed{seed}'
  cfg=json.loads((folder/'config.json').read_text())
  rows=list(csv.DictReader((folder/'train_log.csv').open()))
  assert [int(r['epoch']) for r in rows]==list(range(1,601))
  assert all(math.isfinite(float(r['train_l1'])) for r in rows)
  assert cfg['seed']==seed and cfg['resume'] is None and cfg['ablation']=='deform_stage3'
  records.append({'seed':seed,'epochs':600,'first_l1':float(rows[0]['train_l1']),'last_l1':float(rows[-1]['train_l1']),'batch_size':cfg['batch_size'],'amp':cfg['amp']})
 train=PROJECT/'CAVE/Data/Train/Train.txt';test=PROJECT/'CAVE/Data/Test/Test.txt'
 train_names=set(train.read_text().split());test_names=set(test.read_text().split())
 assert len(train_names)==20 and len(test_names)==12 and not train_names&test_names
 fit=json.loads((root/'Source/physics/cave_effective_response_report.json').read_text())
 assert set(fit['fit_scenes'])<=train_names and not set(fit['fit_scenes'])&test_names
 return {'runs':records,'train_scenes':len(train_names),'test_scenes':len(test_names),'train_test_overlap':[],'spectral_response_fit_uses_only_training_scenes':True}


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
 torch.set_num_threads(4)
 run.reproducible(42)
 p=load_protocol()
 result={'frozen_protocol_and_source_hashes':'passed','protocol_sha256':sha256(ROOT/'protocol.json')}
 if args.device=='cpu':
  result['inventory']=inventory(p)
  result['general_dynamicspatial_cache']=general_cache_check()
  result['cave']=cave_checks()
 result['runtime']=runtime(p,torch.device(args.device))
 # Verify that no frozen source was changed while auditing.
 load_protocol()
 save(OUT/f'{args.device}_audit.json',result)
 print('AUDIT PASSED; result',OUT/f'{args.device}_audit.json',flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(OUT/'audit_error.json',{'error':str(exc),'traceback':traceback.format_exc()})
  raise

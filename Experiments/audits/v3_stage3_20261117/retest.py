"""Re-run seed42 evaluation using the frozen production test entry, in a new audit folder."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[2]
ROOT=PROJECT/'Experiments/v3_stage3_restart_20261111'
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import run
from metrics import evaluate
from protocol import load_protocol,sha256
p=load_protocol();comparisons=[]
original_evaluate=run.evaluate
for dataset in ('pavia','chikusei'):
 for sf in (4,8):
  target=OUT/'gpu_retest'/f'{dataset}_seed42'
  checkpoint=ROOT/'Checkpoint'/f'{dataset}_v3_stage3_x4_seed42'/'last.pth'
  diagnostic=[]
  def with_diagnostics(pred,gt,sf,sensor_valid=None):
   clipped=original_evaluate(pred,gt,sf,sensor_valid)
   diagnostic.append({'kind':'model' if len(diagnostic)%2==0 else 'bicubic','unclipped':original_evaluate(pred,gt,sf,sensor_valid,clip=False),'clipped':clipped,'raw_min':float(pred.min()),'raw_max':float(pred.max()),'fraction_outside_0_1':float(((pred<0)|(pred>1)).float().mean())})
   return clipped
  run.evaluate=with_diagnostics
  args=SimpleNamespace(output=target,checkpoint=checkpoint,sf=sf,dataset=dataset)
  run.test(args,p,p['datasets'][dataset],torch.device('cuda'))
  result=json.loads((target/f'last_test_x{sf}.json').read_text())
  stored=json.loads((checkpoint.parent/f'last_test_x{sf}.json').read_text())
  errors={k:abs(result['mean'][k]-stored['mean'][k]) for k in stored['mean']}
  assert max(errors.values())<1e-5,errors
  unclipped_mean={k:float(np.mean([x['unclipped'][k] for x in diagnostic if x['kind']=='model'])) for k in stored['mean']}
  data={'dataset':dataset,'sf':sf,'mean_absolute_errors':errors,'matches_original_within_1e_5':True,'clipped_mean':result['mean'],'unclipped_mean':unclipped_mean,'per_patch':diagnostic}
  (target/f'unclipped_diagnostics_x{sf}.json').write_text(json.dumps(data,indent=2)+'\n')
  comparisons.append(data)
  print('RETEST MATCH',dataset,sf,errors,flush=True)
  torch.cuda.empty_cache()
(OUT/'retest_summary.json').write_text(json.dumps(comparisons,indent=2)+'\n')
load_protocol()
print('Four production retests passed; original checkpoints and reports retained.',flush=True)

"""Evaluate remaining 12 x4-to-x8 ablations; resume completed reports safely."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json,os,subprocess,sys,csv,statistics,re
ROOT=Path(__file__).resolve().parents[1]
V={'baseline':['--model','sfno'],'spatial':['--model','sfno_prgno','--disable-spectral-residual'],'spectral':['--model','sfno_prgno','--disable-spatial-residual'],'capacity':['--model','sfno_prgno','--guidance-mode','learned']}
def report(v,s):return ROOT/'physics'/f'{"v1" if v=="dual" else v}_x4_to_x8_seed{s}.json'
def run(v,s):
 p=report(v,s)
 if p.exists():
  d=json.loads(p.read_text());assert d['source_sf']==4 and d['sf']==8 and len(d['per_scene'])==12
  return f'SKIP {v} {s}'
 cp=ROOT/'Checkpoint/ablation/x4'/f'{v}_seed{s}'/'last.pth'
 if v=='baseline' and s==42:cp=ROOT/'Checkpoint/train_x4/last.pth'
 assert cp.is_file(),cp
 cmd=[sys.executable,'-u',str(ROOT/'Test.py'),'--data_path',str(ROOT/'Data/Test'),'--checkpoint',str(cp),'--sf','8','--cross-scale','--seed','1','--report',str(p),*V[v]]
 env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 with (ROOT/f'test_{v}_x4_to_x8_seed{s}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 return f'DONE {v} {s}: {json.loads(p.read_text())["mean"]}'
def summarize():
 rows=[]
 for v in ['baseline','spatial','spectral','dual','capacity']:
  values={4:[],8:[]}
  for s in (42,43,44):
   p=ROOT/'Checkpoint/ablation/x4'/f'{v}_seed{s}'/'metrics.json'
   if p.exists():d=json.loads(p.read_text())['mean']
   else:
    assert s==42 and v in ('baseline','dual')
    log=ROOT/('test_x4_200.log' if v=='baseline' else 'test_prgno_x4.log')
    nums=re.findall(r'Avg (?:PSNR|SAM|ERGAS)\s*=\s*([\d.]+)',log.read_text())
    d=dict(zip(['psnr','sam','ergas'],map(float,nums[-3:])))
   values[4].append(d);values[8].append(json.loads(report(v,s).read_text())['mean'])
  row={'Model':{'baseline':'Baseline (SFNO)','spatial':'Spatial-only','spectral':'Spectral-only','dual':'Dual (V1)','capacity':'Learned Guidance'}[v]}
  for sf,rs in values.items():
   for k in ['psnr','sam','ergas']:
    vals=[r[k] for r in rs];row[f'x{sf}_{k}']=f'{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}'
  rows.append(row)
 with (ROOT/'physics/v1_ablation_complete.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# V1 消融完整表','','训练seed42/43/44：均值 ± 样本标准差；推理seed=1。×8为×4权重跨尺度推理，无微调，ERGAS倍率系数100/8。×4的seed42 baseline/dual使用旧日志四位小数。','','| '+' | '.join(rows[0])+' |','|'+'|'.join(['---']*len(rows[0]))+'|']+['| '+' | '.join(r.values())+' |' for r in rows]
 (ROOT/'physics/v1_ablation_complete.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(run,v,s) for v in V for s in (42,43,44)]
  for j in as_completed(jobs):print(j.result(),flush=True)
 summarize()

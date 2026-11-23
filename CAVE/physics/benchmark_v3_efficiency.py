import sys,time,json,gc
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'Model'))
from SFNO_PRGNO_V3 import SFNO_PRGNO_V3
torch.set_num_threads(4)
rows=[]
for variant in ('full','no_deformable','no_regional','fixed_gate'):
 folder=root/'Checkpoint'/('train_x4_prgno_v3_e600' if variant=='full' else f'ablation_v3/{variant}_seed42')
 ck=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
 model=SFNO_PRGNO_V3(response=ck['model']['physics.spectral.response'],ablation=variant).cuda().eval()
 model.load_state_dict(ck['model'])
 gates=[float(b.MDG.correction_logit.sigmoid()) for b in (model.block1,model.block2,model.block3)]
 for sf in (4,8):
  ms=torch.rand(1,31,256//sf,256//sf,device='cuda');rgb=torch.rand(1,3,256,256,device='cuda')
  with torch.inference_mode():
   for _ in range(5): model(ms,rgb,sf)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   times=[]
   for _ in range(20):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();model(ms,rgb,sf);b.record();b.synchronize();times.append(a.elapsed_time(b))
   row=dict(variant=variant,sf=sf,median_ms=sorted(times)[len(times)//2],peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,gates=gates)
   rows.append(row);print(row,flush=True)
 del model,ck,ms,rgb;gc.collect();torch.cuda.empty_cache()
report=dict(gpu=torch.cuda.get_device_name(),protocol='FP32 eval inference_mode batch=1 HR=256x256; warmup=5, measured=20; CUDA events; synthetic inputs; no I/O',results=rows)
p=root/'physics/v3_efficiency_benchmark.json';p.write_text(json.dumps(report,indent=2));print(p)

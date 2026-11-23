"""Same-checkpoint-family FP32 GPU inference benchmark; no model edits."""
import gc,json,statistics,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'Source'),str(ROOT/'Source/Model')]
from SFNO import SFNO
from SFNO_PRGNO_V3 import SFNO_PRGNO_V3

def main():
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:raise RuntimeError('Other GPU compute jobs active: '+active)
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    rows=[]
    for name,folder,variant in [('baseline','baseline_seed42',None),('full_v3','full_seed42','full'),('stage3','deform_stage3_seed42','deform_stage3')]:
        path=ROOT/'Checkpoint'/({'baseline_seed42':'cave_sfno_x4_seed42','deform_stage3_seed42':'cave_v3_stage3_x4_seed42'}.get(folder,folder))/'last.pth'
        if name == 'full_v3' and not path.exists():
            print('Skipping full_v3: its historical checkpoint was deleted before migration',flush=True)
            continue
        ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==600
        model=SFNO() if variant is None else SFNO_PRGNO_V3(response=ck['model']['physics.spectral.response'],ablation=variant)
        model.load_state_dict(ck['model'],strict=True)
        total=sum(p.numel() for p in model.parameters())
        # Parameters used on the actual forward path; disabled branches remain registered.
        pred=model(torch.rand(1,31,4,4),torch.rand(1,3,16,16),4)
        pred.square().mean().backward()
        active_params=sum(p.numel() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True);del pred,ck
        model=model.cuda().eval()
        for sf in (4,8):
            ms=torch.rand(1,31,256//sf,256//sf,device='cuda');rgb=torch.rand(1,3,256,256,device='cuda')
            with torch.inference_mode():
                for _ in range(10):model(ms,rgb,sf)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                allocated=torch.cuda.memory_allocated();times=[];walls=[];round_medians=[]
                for _ in range(3):
                    group=[]
                    for _ in range(10):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        wall=time.perf_counter();start.record();model(ms,rgb,sf);end.record();end.synchronize()
                        walls.append((time.perf_counter()-wall)*1000);group.append(start.elapsed_time(end))
                    times+=group;round_medians.append(statistics.median(group))
                row=dict(model=name,sf=sf,registered_params=total,forward_used_params=active_params,
                    cuda_median_ms=statistics.median(times),cuda_mean_ms=statistics.mean(times),
                    cuda_std_ms=statistics.stdev(times),round_medians_ms=round_medians,
                    synchronized_wall_median_ms=statistics.median(walls),
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                    incremental_peak_MiB=(torch.cuda.max_memory_allocated()-allocated)/2**20,
                    checkpoint=str(path))
                rows.append(row);print(json.dumps(row),flush=True)
            del ms,rgb
        del model;gc.collect();torch.cuda.empty_cache()
    report=dict(skipped_models=['full_v3 (historical checkpoint previously deleted)'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,batch=1,HR_size=[256,256],
        precision='FP32, TF32 disabled, no autocast',mode='eval + inference_mode',
        warmup=10,timed_calls=30,rounds=3,threads=4,includes='model forward only; no image I/O, no host-device transfer',
        caveat='Synthetic inputs, one device and size; wall median includes dispatch and synchronization. Forward-used params counted via gradient presence on CPU x4 probe; registered disabled branches are not removed.',results=rows)
    (OUT/'efficiency.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',OUT/'efficiency.json',flush=True)
if __name__=='__main__':main()

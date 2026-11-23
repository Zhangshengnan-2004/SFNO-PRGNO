"""Non-overlapping Stage3 module timing and separate allocation probes."""
import argparse,collections,csv,json,statistics,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Source'),str(ROOT/'Source/Model')]
from SFNO_PRGNO_V3 import SFNO_PRGNO_V3


def targets(model):
    result=[]
    for i in (1,2,3):result.append((f'VI1_{i}','VectorAttention',getattr(model,f'VI1_{i}')))
    for i in (1,2,3):
        block=getattr(model,f'block{i}')
        result.extend([(f'block{i}.regional','Regional',block.MDG.regional),
                       (f'block{i}.GI','GI',block.GI)])
    result += [('block3.deformable','Deformable',model.block3.MDG.deformable),
               ('physics','Physics',model.physics)]
    return result


def event():return torch.cuda.Event(enable_timing=True)


def profile(model,ms,rgb,sf,iterations):
    modules=targets(model)
    expected={'VectorAttention':3,'Regional':3,'Deformable':1,'Physics':3,'GI':3}
    with torch.inference_mode():
        for _ in range(10):model(ms,rgb,sf)
        torch.cuda.synchronize()
        clean=[]
        for _ in range(iterations):
            a,b=event(),event();a.record();model(ms,rgb,sf);b.record();b.synchronize();clean.append(a.elapsed_time(b))
        # Time probes: synchronize only once at the end of each whole forward.
        pending={};records=[];handles=[]
        def pre(name,group):
            def hook(module,args):
                a=event();a.record();pending[name]=(group,a)
            return hook
        def post(name):
            def hook(module,args,output):
                group,a=pending.pop(name);b=event();b.record();records.append((name,group,a,b))
            return hook
        for name,group,module in modules:
            handles.extend([module.register_forward_pre_hook(pre(name,group)),module.register_forward_hook(post(name))])
        groups=collections.defaultdict(list);individual=collections.defaultdict(list);hooked=[]
        try:
            for _ in range(iterations):
                records.clear();a,b=event(),event();a.record();model(ms,rgb,sf);b.record();b.synchronize()
                hooked.append(a.elapsed_time(b));counts=collections.Counter(r[1] for r in records)
                assert dict(counts)==expected,counts
                sums=collections.defaultdict(float)
                for name,group,start,end in records:
                    value=start.elapsed_time(end);individual[name].append(value);sums[group]+=value
                for group,value in sums.items():groups[group].append(value)
        finally:
            for h in handles:h.remove()
        # Allocation probes are separate: barriers and peak resets would distort timings.
        starts={};memory=collections.defaultdict(list);handles=[]
        def mempre(name):
            def hook(module,args):
                torch.cuda.synchronize();starts[name]=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            return hook
        def mempost(name):
            def hook(module,args,output):
                torch.cuda.synchronize();before=starts.pop(name)
                memory[name].append(dict(entry_MiB=before/2**20,
                    additional_peak_MiB=max(0,torch.cuda.max_memory_allocated()-before)/2**20,
                    retained_delta_MiB=(torch.cuda.memory_allocated()-before)/2**20))
            return hook
        for name,_,module in modules:
            handles.extend([module.register_forward_pre_hook(mempre(name)),module.register_forward_hook(mempost(name))])
        try:
            for _ in range(3):model(ms,rgb,sf)
        finally:
            for h in handles:h.remove()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();model(ms,rgb,sf);torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()/2**20
        entries=[];full=statistics.median(hooked)
        for group in expected:
            names=[name for name,g,_ in modules if g==group]
            entries.append(dict(group=group,calls_per_forward=expected[group],
                median_ms=statistics.median(groups[group]),std_ms=statistics.stdev(groups[group]),
                percent_of_instrumented_forward=100*statistics.median(groups[group])/full,
                max_single_call_additional_peak_MiB=max(v['additional_peak_MiB'] for n in names for v in memory[n])))
        return dict(sf=sf,clean_forward_median_ms=statistics.median(clean),
            instrumented_forward_median_ms=full,whole_forward_peak_allocated_MiB=peak,groups=entries,
            individual_calls={n:dict(median_ms=statistics.median(v),calls_per_forward=len(v)//iterations,
                memory_probes=memory[n]) for n,v in individual.items()})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--size',type=int,default=256);parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'module_profile')
    args=parser.parse_args()
    if args.size%8 or args.iterations<2:parser.error('size must be divisible by 8 and iterations >=2')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU has another compute task; profiling aborted')
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    checkpoint=ROOT/'Checkpoint/cave_v3_stage3_x4_seed42/last.pth'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=SFNO_PRGNO_V3(response=ck['model']['physics.spectral.response'],ablation='deform_stage3').cuda().eval()
    model.load_state_dict(ck['model'],strict=True);del ck
    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,checkpoint=str(checkpoint),
        HR_size=[args.size,args.size],batch=1,precision='FP32; TF32 disabled',warmup=10,iterations=args.iterations,
        timing='CUDA events, inclusive per module; groups are non-nested, summed per forward then median',
        memory='separate 3-pass probes; maximum extra allocated bytes above module-entry baseline; includes output and temporaries, excludes already-allocated weights and inputs; group peaks must NOT be summed',
        scope='Physics covers model.physics only; intermediate HSI reconstruction head, initial/final projections, interpolation and residual arithmetic are in unassigned time; inference only, not training',results=[])
    args.output.mkdir(parents=True,exist_ok=True)
    for sf in (4,8):
        ms=torch.rand(1,31,args.size//sf,args.size//sf,device='cuda');rgb=torch.rand(1,3,args.size,args.size,device='cuda')
        result['results'].append(profile(model,ms,rgb,sf,args.iterations))
        print(json.dumps(result['results'][-1]['groups'],indent=2),flush=True)
        if args.trace and sf==4:
            # One forward with named regions, including operation-level allocation attribution.
            scopes=collections.defaultdict(list);handles=[]
            def pre(name):
                def h(module,inputs):
                    ctx=torch.profiler.record_function('MODULE::'+name);ctx.__enter__();scopes[name].append(ctx)
                return h
            def post(name):
                def h(module,inputs,output):scopes[name].pop().__exit__(None,None,None)
                return h
            for name,_,module in targets(model):handles += [module.register_forward_pre_hook(pre(name)),module.register_forward_hook(post(name))]
            try:
                with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                    model(ms,rgb,sf);torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output/'trace_x4.json'))
                (args.output/'operators_x4.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
            finally:
                for h in handles:h.remove()
        del ms,rgb
    (args.output/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['| 模块 | 每次前向调用次数 | ×4耗时 ms | ×4占比 | ×8耗时 ms | ×8占比 | 最大单次额外峰值 MiB |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for a,b in zip(result['results'][0]['groups'],result['results'][1]['groups']):
        lines.append(f"| {a['group']} | {a['calls_per_forward']} | {a['median_ms']:.2f} | {a['percent_of_instrumented_forward']:.1f}% | {b['median_ms']:.2f} | {b['percent_of_instrumented_forward']:.1f}% | {max(a['max_single_call_additional_peak_MiB'],b['max_single_call_additional_peak_MiB']):.2f} |")
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'summary.md',flush=True)
if __name__=='__main__':main()

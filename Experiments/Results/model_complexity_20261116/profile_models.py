"""CPU forward complexity: multiply-add=2 FLOPs; analytic supplementary estimates."""
import sys, json, csv, math, importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'CAVE/Model'),str(ROOT/'CAVE'),str(WORK/'SSRNET-master/comparison')]
from SFNO import SFNO
from SFNO_PRGNO_V3 import SFNO_PRGNO_V3
from adapter import ComparisonSSRNET

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class CounterMode(FlopCounterMode):
    def __init__(self):
        super().__init__(display=False);self.extra=Counter();self.omitted=Counter()
    def _count_flops(self,packet,out,args,kwargs):
        super()._count_flops(packet,out,args,kwargs)
        name=str(packet)
        if packet in self.flop_registry:return out
        outs=[out] if isinstance(out,torch.Tensor) else [v for v in out if isinstance(v,torch.Tensor)] if isinstance(out,(tuple,list)) else []
        n=sum(v.numel() for v in outs)
        if name in ('aten._fft_c2c','aten._fft_r2c','aten._fft_c2r'):
            x,dims=args[:2];size=math.prod(x.shape[d] for d in dims)
            # Complex FFT ~5Nlog2N; real transforms ~half. Estimate, not exact kernel count.
            value=5*x.numel()*math.log2(size)
            if name=='aten._fft_r2c':value*=.5
            if name=='aten._fft_c2r':value=2.5*n*math.log2(math.prod(out.shape[d] for d in dims))
            self.extra['fft_estimate']+=value
        elif name=='aten.grid_sampler_2d':self.extra['bilinear_grid_sample_estimate']+=9*n
        elif 'upsample_bicubic2d' in name:self.extra['bicubic_interpolation_estimate']+=35*n
        elif 'upsample_bilinear2d' in name:
            # antialias filter depends on reduction ratio: conservative separable footprint estimate
            x=args[0]; outsize=out.shape[-2:]
            taps=[max(2,math.ceil(2*x.shape[-2+i]/outsize[i])) for i in range(2)] if '_aa' in name else [2,2]
            self.extra['bilinear_interpolation_estimate']+=(2*sum(taps)-1)*n
        else:self.omitted[name]+=1
        return out

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    outdir=Path(__file__).parent;rows=[];details=[]
    for dataset,bands in [('cave',31),('pavia',102),('chikusei',128)]:
        protocol=None
        if dataset=='cave':response=np.load(ROOT/'CAVE/physics/cave_effective_response.npz')['response']
        else:
            response=np.load(ROOT/f'Experiments/v3_stage3_restart_20261111/response_{dataset}.npy')
            protocol=load('matched_spatial',ROOT/'Experiments/v3_stage3_restart_20261111/matched_spatial.py')
        mschannels=response.shape[0]
        for label in ['SFNO','SPGFNet (V3-stage3)','SSR-NET']:
            if label=='SFNO':model=SFNO(size=64,hschannels=bands,mschannels=mschannels)
            elif label.startswith('SPGFNet'):
                model=SFNO_PRGNO_V3(size=64,hschannels=bands,mschannels=mschannels,response=response,ablation='deform_stage3')
                if protocol:model.physics.spatial=protocol.MatchedSpatial()
            else:model=ComparisonSSRNET(bands,mschannels)
            if label=='SSR-NET':ckpt=WORK/f'SSRNET-master/comparison/Checkpoint/{dataset}_ssrnet_x4_seed42/last.pth'
            else:
                tag='sfno' if label=='SFNO' else 'v3_stage3'
                folder='cave_continuous600' if dataset=='cave' else ('sfno_restart_20261111' if label=='SFNO' else 'v3_stage3_restart_20261111')
                ckpt=ROOT/f'Experiments/{folder}/Checkpoint/{dataset}_{tag}_x4_seed42/last.pth'
            model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model'],strict=True)
            model.eval();total=sum(p.numel() for p in model.parameters())
            for sf in [4,8]:
                low=torch.randn(1,bands,64//sf,64//sf);msi=torch.randn(1,mschannels,64,64)
                used=set();handles=[]
                def hook(module,inputs,output):
                    used.update(id(p) for p in module.parameters(recurse=False))
                for module in model.modules():handles.append(module.register_forward_hook(hook))
                with torch.no_grad():
                    model(low,msi,sf) # warm caches, kernels are setup cost
                    with CounterMode() as counter:pred=model(low,msi,sf)
                for h in handles:h.remove()
                assert pred.shape==(1,bands,64,64) and torch.isfinite(pred).all()
                active=sum(p.numel() for p in model.parameters() if id(p) in used)
                if label.startswith('SPGFNet'):active+=1 # correction_logit directly used by parent
                core=counter.get_total_flops();extra=sum(counter.extra.values())
                row=dict(dataset=dataset,model=label,scale=sf,hr_size='64x64',params=total,params_M=total/1e6,
                         executed_params=active,core_GFLOPs=core/1e9,supplement_estimate_GFLOPs=extra/1e9,
                         major_estimate_GFLOPs=(core+extra)/1e9)
                rows.append(row);details.append(dict(**row,supplement=dict(counter.extra),unaccounted_ops=dict(counter.omitted),core_ops={str(k):v for k,v in counter.flop_counts['Global'].items()}))
                print(row,flush=True)
    (outdir/'details.json').write_text(json.dumps(details,indent=2))
    with (outdir/'complexity.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lines=['| Dataset | Model | Params (M) | Core ×4 (G) | Core ×8 (G) | Major estimate ×4 (G) | Major estimate ×8 (G) |','|---|---|---:|---:|---:|---:|---:|']
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2];lines.append(f"| {a['dataset']} | {a['model']} | {a['params_M']:.6f} | {a['core_GFLOPs']:.4f} | {b['core_GFLOPs']:.4f} | {a['major_estimate_GFLOPs']:.4f} | {b['major_estimate_GFLOPs']:.4f} |")
    (outdir/'summary.md').write_text('\n'.join(lines)+'\n\nBatch=1, HR=64×64, FP32 CPU, multiply-add=2 FLOPs. Core: PyTorch 2.6 FlopCounterMode supported operators (mainly convolution/matrix products). Major estimate adds approximate FFT, interpolation and grid sampling costs. Neither column is a complete FLOPs count; elementwise, normalization, activation, reduction, softmax, pooling and indexing costs are not fully included. details.json lists unaccounted operators. Warm forward excludes cached geometry setup. Weights do not affect these static counts. SPGFNet is mapped to deform_stage3. Params includes registered but unused modules.\n')
if __name__=='__main__':main()

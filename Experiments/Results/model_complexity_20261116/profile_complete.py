"""Algorithmic FLOP accounting under documented conventions; see complete_rules/RULES.md."""
import sys, json, csv, math, importlib.util, hashlib
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
    """Explicit algorithmic convention, fail closed for unclassified operators."""
    ZERO=set("cat split im2col view unsqueeze squeeze permute clone t randperm sort slice index select transpose expand _unsafe_view _to_copy lift_fresh new_zeros stack arange zeros_like copy_ view_as_real _conj detach alias empty empty_like contiguous reshape _local_scalar_dense is_same_size".split())
    def __init__(self):
        super().__init__(display=False); self.extra=Counter();self.omitted=Counter();self.calls=Counter()
    def _count_flops(self,packet,out,args,kwargs):
        super()._count_flops(packet,out,args,kwargs)
        name=str(packet).removeprefix('aten.');self.calls[name]+=1
        if packet in self.flop_registry:
            # Built-in GEMM/convolution formulas count multiply-add=2, but omit bias.
            bias=None
            if name=='convolution':bias=args[2]
            if name in ('addmm','baddbmm'):bias=args[0]
            if bias is not None:self.extra['bias_add']+=out.numel()
            return out
        tensors=[out] if isinstance(out,torch.Tensor) else [v for v in out if isinstance(v,torch.Tensor)] if isinstance(out,(tuple,list)) else []
        y=tensors[0] if tensors else None;n=y.numel() if y is not None else 0
        x=args[0] if args else None
        cost=0;category=name
        if name in self.ZERO:return out
        if name in ('add','add_','sub','sub_','mul','mul_','div','div_'):
            if y is not None and (y.is_floating_point() or y.is_complex()):
                if y.is_complex():
                    c=sum(isinstance(a,torch.Tensor) and a.is_complex() for a in args[:2])
                    unit=(6 if c==2 else 2) if name.startswith('mul') else 2 if name.startswith(('add','sub')) else 11
                else:unit=1
                cost=n*unit
        elif name=='scatter_add_':cost=args[2].numel()
        elif name in ('sum','mean','std'):
            dims=args[1] if len(args)>1 else None
            dims=tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims,int) else tuple(dims)
            k=math.prod(x.shape[d] for d in dims);groups=x.numel()//k
            cost=groups*((k-1) if name=='sum' else k if name=='mean' else 4*k+1)
        elif name in ('native_layer_norm','native_group_norm'):
            if name=='native_layer_norm':k=math.prod(args[1]);affine=args[2] is not None;bias=args[3] is not None
            else:k=args[4]//args[6]*args[5];affine=args[1] is not None;bias=args[2] is not None
            groups=x.numel()//k
            cost=groups*(5*k+2)+x.numel()*(int(affine)+int(bias))
        elif name=='_scaled_dot_product_flash_attention_for_cpu':
            q,k,v=args[:3];batch=math.prod(q.shape[:-2]);l,d=q.shape[-2:];length=k.shape[-2]
            cost=2*batch*l*length*(d+v.shape[-1])+batch*l*length+batch*l*(5*length-2)
            category='SDPA_matmul_scale_softmax'
        elif name=='_softmax':
            k=x.shape[args[1]];cost=x.numel()//k*(5*k-2) # max comparisons + subtract + exp + sum + divide
        elif name in ('relu','relu_','leaky_relu','leaky_relu_'):cost=n*(2 if 'leaky' in name else 1)
        elif name=='gelu':cost=5*n # exact GELU: .5*x*(1+erf(x/sqrt(2)))
        elif name=='sigmoid':cost=4*n
        elif name in ('log','tanh','exp','sqrt','rsqrt','neg','abs','floor','ceil'):cost=n
        elif name=='_adaptive_avg_pool2d':
            ih,iw=x.shape[-2:];oh,ow=y.shape[-2:]
            ky=sum(math.ceil((i+1)*ih/oh)-math.floor(i*ih/oh) for i in range(oh))
            kx=sum(math.ceil((i+1)*iw/ow)-math.floor(i*iw/ow) for i in range(ow))
            cost=math.prod(x.shape[:-2])*ky*kx
        elif name in ('_fft_c2c','_fft_r2c','_fft_c2r'):
            dims=args[1];size=math.prod(x.shape[d] for d in dims)
            cost=5*x.numel()*math.log2(size)
            if name!='_fft_c2c':cost*=.5
            if len(args)>2 and args[2]!=0:cost+=2*n if y.is_complex() else n
            category='FFT_radix2_convention'
        elif name=='grid_sampler_2d':
            assert args[2]==0 and args[3]==1 and args[4]==False
            # Per output position: coordinate transform 6, fraction 2, complement 2,
            # four 2-D weights 4; each channel weighted sum 4mul+3add.
            cost=7*n+14*args[1].numel()//2
            category='bilinear_grid_sample'
        elif 'upsample_bicubic2d' in name:
            cost=31*n;category='bicubic_weighted_sum'
        elif 'upsample_bilinear2d' in name:
            if '_aa' not in name:cost=7*n
            else:
                ih,iw=x.shape[-2:];oh,ow=y.shape[-2:]
                def taps(i,o):
                    scale=i/o;support=max(scale,1.)
                    return sum(max(0,min(i,int((j+.5)*scale+support+.5))-max(0,int((j+.5)*scale-support+.5))) for j in range(o))
                cost=math.prod(x.shape[:-2])*(ih*(2*taps(iw,ow)-ow)+ow*(2*taps(ih,oh)-oh))
            category='bilinear_weighted_sum'
        else:self.omitted[str(packet)]+=1
        self.extra[category]+=cost
        return out

def main():
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.mha.set_fastpath_enabled(False)
    outdir=Path(__file__).parent/'complete_rules';outdir.mkdir(exist_ok=True);rows=[];details=[]
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
            checkpoint_sha_before=hashlib.sha256(ckpt.read_bytes()).hexdigest()
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
                assert not counter.omitted, dict(counter.omitted)
                assert pred.shape==(1,bands,64,64) and torch.isfinite(pred).all()
                active=sum(p.numel() for p in model.parameters() if id(p) in used)
                # Parent hooks include correction_logit and MultiheadAttention direct parameters.
                core=counter.get_total_flops();extra=sum(counter.extra.values())
                row=dict(dataset=dataset,model=label,scale=sf,hr_size='64x64',params=total,params_M=total/1e6,
                         core_GFLOPs=core/1e9,additional_GFLOPs=extra/1e9,
                         algorithmic_GFLOPs=(core+extra)/1e9)
                rows.append(row);details.append(dict(**row,supplement=dict(counter.extra),unaccounted_ops=dict(counter.omitted),checkpoint=str(ckpt),checkpoint_sha256=checkpoint_sha_before,operator_calls=dict(counter.calls),core_ops={str(k):v for k,v in counter.flop_counts['Global'].items()}))
                print(row,flush=True)
            assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==checkpoint_sha_before, 'Checkpoint changed during profiling'
    (outdir/'details.json').write_text(json.dumps(details,indent=2))
    with (outdir/'complexity.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lines=['| Dataset | Model | Params (M) | Core ×4 (G) | Core ×8 (G) | Algorithmic ×4 (G) | Algorithmic ×8 (G) |','|---|---|---:|---:|---:|---:|---:|']
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2];lines.append(f"| {a['dataset']} | {a['model']} | {a['params_M']:.6f} | {a['core_GFLOPs']:.4f} | {b['core_GFLOPs']:.4f} | {a['algorithmic_GFLOPs']:.4f} | {b['algorithmic_GFLOPs']:.4f} |")
    (outdir/'summary.md').write_text('\n'.join(lines)+'\n\nSee RULES.md for exact scope and formulas.\n')
if __name__=='__main__':main()

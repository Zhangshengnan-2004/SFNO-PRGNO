"""Independent SSRNET closed form, coverage, kernel-mode equivalence checks."""
import json
from pathlib import Path
import torch
from profile_complete import CounterMode,ComparisonSSRNET
BASE=Path(__file__).parent
rows=json.loads((BASE/'complete_rules/details.json').read_text())
checks=[]
for c in (31,102,128):
    # Three C->C 3x3 convolutions, bias+ReLU each; two residual adds;
    # bilinear LR upsampling, plus two spatial and one spectral differences.
    expected=3*4096*c*c*9*2+3*4096*c*2+2*4096*c+7*4096*c+2*c*63*64+(c-1)*4096
    found=[r for r in rows if r['model']=='SSR-NET' and r['params']==27*c*c+3*c+1]
    assert len(found)==2
    for r in found:
        assert round(r['algorithmic_GFLOPs']*1e9)==expected,(r,expected)
        checks.append(dict(test='SSRNET independent closed-form count',channels=c,scale=r['scale'],flops=expected))
assert len(rows)==18 and all(not r['unaccounted_ops'] for r in rows)
# Verify counted attention decomposition is numerically equivalent to optimized eval.
torch.manual_seed(9);torch.set_num_threads(2)
m=torch.nn.MultiheadAttention(64,4,batch_first=True).eval();x=torch.randn(1,64,64)
with torch.no_grad():
    torch.backends.mha.set_fastpath_enabled(True);a=m(x,x,x,need_weights=False)[0]
    torch.backends.mha.set_fastpath_enabled(False)
    with CounterMode() as counter:b=m(x,x,x,need_weights=False)[0]
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    assert not counter.omitted,dict(counter.omitted)
checks.append(dict(test='MHA optimized and counted paths agree',max_abs_error=(a-b).abs().max().item()))
# Check convolution+bias separately against a rectangular grouped convolution.
m=torch.nn.Conv2d(4,6,3,groups=2,bias=True).eval();x=torch.randn(1,4,7,9)
with torch.no_grad(),CounterMode() as c:y=m(x)
expected=y.numel()*(2*(4//2)*3*3+1)
assert c.get_total_flops()+sum(c.extra.values())==expected
checks.append(dict(test='grouped conv+bias independent count',flops=expected))
(BASE/'complete_rules/verification.json').write_text(json.dumps(dict(passed=True,configurations=18,unclassified_ops=0,checks=checks),indent=2))
print('PASS: 18 configurations covered; SSRNET closed form; grouped convolution+bias; attention equivalence')

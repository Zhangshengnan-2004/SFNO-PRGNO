"""Supplement MHF forward FLOPs using the existing explicit algorithmic rules.

No GPU, training, checkpoint loading or modifications to experiment sources.
"""
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
WORKSPACE = PROJECT.parent
OLD = PROJECT / 'Experiments/Results/model_complexity_20261116'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# profile_complete imports the SSR adapter as "adapter"; load MHF under a distinct name.
base = load_module('existing_complete_counter', OLD / 'profile_complete.py')
adapter_path = WORKSPACE / 'MHF-net-master/comparison/adapter.py'
mhf = load_module('mhf_adapter_for_complexity', adapter_path)


class MHFCounter(base.CounterMode):
    ZERO = base.CounterMode.ZERO | {'constant_pad_nd'}

    def __init__(self):
        super().__init__()
        self.convolution_reference = 0
        self.convolution_calls = []

    def _count_flops(self, packet, out, args, kwargs):
        name = str(packet).removeprefix('aten.')
        if name == 'var_mean':
            # Population mean+variance: k + k + k + (k-1) + 1 = 4k per group.
            self.calls[name] += 1
            self.extra['population_var_mean'] += 4 * args[0].numel()
            return out
        if name == 'convolution':
            x, weight, bias = args[:3]
            transposed = bool(args[6])
            spatial = x.shape[2:] if transposed else out.shape[2:]
            expected = 2 * x.shape[0] * math.prod(spatial) * weight.numel()
            self.convolution_reference += expected
            self.convolution_calls.append({
                'input': list(x.shape), 'weight': list(weight.shape),
                'output': list(out.shape), 'transposed': transposed,
                'groups': args[8], 'flops_without_bias': expected,
            })
        return super()._count_flops(packet, out, args, kwargs)


def verify_counter():
    x = torch.ones(1, 2, 2, 3)
    weight = torch.ones(2, 3, 3, 3)
    with MHFCounter() as counter:
        full = torch.nn.functional.conv_transpose2d(x, weight, stride=2)
        cropped = full[..., :4, :6]
    assert full.shape == (1, 3, 5, 7) and cropped.shape == (1, 3, 4, 6)
    assert counter.get_total_flops() == 648
    assert not counter.omitted
    moment_input = torch.ones(2, 3, 4, 5)
    with MHFCounter() as moments:
        torch.var_mean(moment_input, dim=(0, 2, 3), correction=0)
    assert moments.extra['population_var_mean'] == 4 * 120
    assert not moments.omitted
    norm = mhf.MomentNorm(3, 0.05)
    with torch.no_grad(), MHFCounter() as normalized:
        norm(moment_input)
    assert sum(normalized.extra.values()) == 966
    assert not normalized.omitted
    return {'transpose_convolution_then_crop_flops': 648,
            'population_var_mean_flops_120_elements': 480,
            'complete_moment_norm_120_elements': 966}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    verification = verify_counter()
    rows, details = [], []
    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    for dataset, bands, msi in [('cave', 31, 3), ('pavia', 102, 4), ('chikusei', 128, 3)]:
        config_path = WORKSPACE / f'MHF-net-master/comparison/Checkpoint/{dataset}_mhf_x4_seed42/config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else None
        if config:
            assert config['rank'] == 12 and config['stages'] == 20
            expected_hash = config.get('adapter_sha256') or config.get('source_sha256', {}).get(str(adapter_path))
            assert expected_hash == adapter_hash
        model = mhf.ComparisonMHF(bands, msi, rank=12, stages=20).eval()
        params = sum(p.numel() for p in model.parameters())
        for sf in (4, 8):
            low = torch.randn(1, bands, 64 // sf, 64 // sf)
            high = torch.randn(1, msi, 64, 64)
            with torch.no_grad():
                model(low, high, sf)
                with MHFCounter() as counter:
                    output = model(low, high, sf)
            assert output.shape == (1, bands, 64, 64) and torch.isfinite(output).all()
            assert not counter.omitted, dict(counter.omitted)
            core = counter.get_total_flops()
            assert core == counter.convolution_reference
            transposed = [c for c in counter.convolution_calls if c['transposed']]
            expected_transposed = 19 * 2 * bands * bands * 9 * (16**2 + 32**2 + (8**2 if sf == 8 else 0))
            assert len(transposed) == (38 if sf == 4 else 57)
            assert sum(c['flops_without_bias'] for c in transposed) == expected_transposed
            assert counter.calls['var_mean'] == 126
            assert counter.calls['constant_pad_nd'] == 38
            additional = sum(counter.extra.values())
            row = dict(dataset=dataset, model='MHF-Net', scale=sf, hr_size='64x64',
                       hsi_channels=bands, msi_channels=msi, params=params, params_M=params / 1e6,
                       core_GFLOPs=core / 1e9, additional_GFLOPs=additional / 1e9,
                       algorithmic_GFLOPs=(core + additional) / 1e9,
                       training_config_present=config is not None)
            rows.append(row)
            details.append(dict(**row, supplement=dict(counter.extra),
                                unaccounted_ops=dict(counter.omitted), operator_calls=dict(counter.calls),
                                convolution_calls=counter.convolution_calls,
                                config_path=str(config_path) if config else None,
                                weights='constructor weights; forward path is weight-independent; no training checkpoint accessed'))
            print(row, flush=True)
        del model
    assert hashlib.sha256(adapter_path.read_bytes()).hexdigest() == adapter_hash
    result = dict(torch=torch.__version__, device='cpu', batch=1, hr_size=[64, 64],
                  multiply_add_flops=2, rank=12, stages=20,
                  inherited_rules=str(OLD / 'complete_rules/RULES.md'),
                  added_rules={'population_var_mean': '4*k per population group, correction=0',
                               'constant_pad_nd': '0; deterministic padding/data movement'},
                  adapter_sha256=adapter_hash, base_counter_sha256=hashlib.sha256((OLD / 'profile_complete.py').read_bytes()).hexdigest(),
                  checks=verification, results=details)
    (OUT / 'mhf_details.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUT / 'mhf_complexity.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()

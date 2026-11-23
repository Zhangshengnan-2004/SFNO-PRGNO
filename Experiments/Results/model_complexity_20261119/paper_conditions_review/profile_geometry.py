"""Actual CPU profiles under documented test geometries, NOT Table 1 replication.

The paper does not disclose its FLOPs accounting protocol. This script preserves
that distinction and never rescales the published 61.20/468.62/331.85 values.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import torch

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[3]
OLD = PROJECT / 'Experiments/Results/model_complexity_20261116'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('documented_flop_rules', OLD / 'profile_complete.py')
memory = load('documented_memory_mode', PROJECT / 'Experiments/v3_stage3_restart_20261111/inference_memory.py')


class GeometryCounter(base.CounterMode):
    ZERO = base.CounterMode.ZERO | {'equal', 'floor_divide', 'constant_pad_nd'}

    def _count_flops(self, packet, output, args, kwargs):
        name = str(packet).removeprefix('aten.')
        if name == 'equal':
            assert isinstance(output, bool)
        if name == 'floor_divide':
            assert not output.is_floating_point() and not output.is_complex()
        if name == '_convolution' and args[2] is not None:
            self.extra['bias_add'] += output.numel()
        return super()._count_flops(packet, output, args, kwargs)


def sha(path):
    with path.open('rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


def profile(kind, dataset, size):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    bands, msi = {'cave': (31, 3), 'pavia': (102, 4), 'chikusei': (128, 3)}[dataset]
    sf = 4
    original_code = None
    if kind == 'author':
        assert dataset == 'cave'
        checkpoint = PROJECT / 'CAVE/Checkpoint/BestModel_ts.pt'
        model = torch.jit.load(str(checkpoint), map_location='cpu').eval()
        original_code = dict(top=model.code, ADE=model.VI1_1.code,
                             PR={k: getattr(model, k).PR.code for k in ('block1', 'block2', 'block3')})
    else:
        if kind == 'sfno':
            model = base.SFNO(size=64, hschannels=bands, mschannels=msi)
        else:
            response = base.np.load(PROJECT / 'CAVE/physics/cave_effective_response.npz')['response'] if dataset == 'cave' else base.np.load(PROJECT / f'Experiments/v3_stage3_restart_20261111/response_{dataset}.npy')
            model = base.SFNO_PRGNO_V3(size=64, hschannels=bands, mschannels=msi,
                                     response=response, ablation='deform_stage3')
            if dataset != 'cave':
                spatial = load('frozen_spatial_for_profile', PROJECT / 'Experiments/v3_stage3_restart_20261111/matched_spatial.py')
                model.physics.spatial = spatial.MatchedSpatial()
        folder = 'cave_continuous600' if dataset == 'cave' else f'{kind}_restart_20261111'
        checkpoint = PROJECT / f'Experiments/{folder}/Checkpoint/{dataset}_{kind}_x4_seed42/last.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval()
        memory.configure(model)
    before = sha(checkpoint)
    params = sum(p.numel() for p in model.parameters())
    low = torch.rand(1, bands, size//sf, size//sf)
    high = torch.rand(1, msi, size, size)
    # Create fixed geometry caches outside the measured forward. No complete warm
    # forward is needed, and no cached inference tensors enter a training model.
    if kind == 'v3_stage3':
        reference = torch.zeros(1, 1, size, size)
        with torch.no_grad():
            spatial = model.physics.spatial
            if hasattr(spatial, '_geometry'):
                spatial._geometry((size, size), (size//sf, size//sf), reference)
            else:
                spatial.geometry((size, size), (size//sf, size//sf), reference)
        del reference
    started = time.monotonic()
    print(f'START {kind} {dataset} HR={size} bands={bands}/{msi}', flush=True)
    with torch.no_grad(), GeometryCounter() as counter:
        pred = model(low, high) if kind == 'author' else model(low, high, sf)
    assert not counter.omitted, dict(counter.omitted)
    assert tuple(pred.shape) == (1, bands, size, size) and torch.isfinite(pred).all()
    assert sha(checkpoint) == before
    row = dict(model=kind, dataset=dataset, batch=1, hsi_channels=bands, msi_channels=msi,
               sf=sf, HR=[size, size], LR=[size//sf, size//sf], params=params, params_M=params/1e6,
               core_FLOPs=counter.get_total_flops(), additional_FLOPs=sum(counter.extra.values()),
               core_GFLOPs=counter.get_total_flops()/1e9,
               algorithmic_GFLOPs=(counter.get_total_flops()+sum(counter.extra.values()))/1e9,
               elapsed_seconds=time.monotonic()-started, checkpoint=str(checkpoint), checkpoint_sha256=before,
               source='author released TorchScript' if kind == 'author' else 'current local experiment implementation',
               input_basis='author default Test.py and exported trace example' if dataset == 'cave' and size == 512 else 'paper Section 4.1 test image size' if size != 64 else 'controlled small-input comparison',
               paper_Table1_conditions_confirmed=False, MAC_FLOPs=2,
               counting='explicit algorithmic rules; FFT/interpolation are documented theoretical costs',
               row_chunked_VectorAttention=(kind != 'author' and size > 128),
               unclassified_ops=dict(counter.omitted), supplemental_ops=dict(counter.extra),
               operator_calls=dict(counter.calls), core_ops={str(k): v for k, v in counter.flop_counts['Global'].items()},
               torch=torch.__version__, device='cpu', original_code=original_code)
    path = OUT / f'{dataset}_{kind}_hr{size}.json'
    path.write_text(json.dumps(row, indent=2) + '\n')
    print('DONE', {k: row[k] for k in ('model', 'dataset', 'HR', 'params_M', 'core_GFLOPs', 'algorithmic_GFLOPs', 'elapsed_seconds')}, flush=True)
    del model, low, high, pred
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--model', choices=['author', 'sfno', 'v3_stage3'])
    parser.add_argument('--dataset', choices=['cave', 'pavia', 'chikusei'])
    parser.add_argument('--size', type=int)
    args = parser.parse_args()
    if args.all:
        jobs = [('author', 'cave', 64), ('author', 'cave', 512)]
        jobs += [(model, dataset, size) for dataset, size in [('cave', 512), ('pavia', 192), ('chikusei', 500)]
                 for model in ('sfno', 'v3_stage3')]
        for model, dataset, size in jobs:
            command = [sys.executable, str(Path(__file__).resolve()), '--model', model,
                       '--dataset', dataset, '--size', str(size)]
            subprocess.run(command, check=True)
    else:
        assert args.model and args.dataset and args.size
        assert args.size % 4 == 0
        profile(args.model, args.dataset, args.size)


if __name__ == '__main__':
    main()

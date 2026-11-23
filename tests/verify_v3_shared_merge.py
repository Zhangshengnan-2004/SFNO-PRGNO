"""CPU equivalence checks for merging V3's shared V2 implementation.

The frozen experiment is the pre-merge reference. This script never imports a
reference model through CAVE/Model, never changes checkpoint tensors, and does
not take an optimizer step. It writes only its JSON result (default: /tmp).

Example, from /home/zsn/projects::

    PYTHONDONTWRITEBYTECODE=1 /home/zsn/miniconda3/envs/sfno/bin/python \
        SFNO-PRGNO/tests/verify_v3_shared_merge.py --output /tmp/v3_merge.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
import types

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np
import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ATOL = 1e-6
RTOL = 1e-5


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def reset_rng(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_hashes():
    return {
        "torch_cpu": tensor_hash(torch.get_rng_state()),
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
    }


def setup_cpu():
    torch.set_num_threads(2)
    # The same explicit attention path is used in both isolated implementations.
    torch.backends.mha.set_fastpath_enabled(False)


def isolated_package(directory, alias):
    """Load relative imports under a unique package, without sys.path fallback."""
    directory = Path(directory).resolve()
    for name in list(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]
    package = types.ModuleType(alias)
    package.__path__ = [str(directory)]
    package.__package__ = alias
    package.__spec__ = importlib.machinery.ModuleSpec(alias, loader=None, is_package=True)
    sys.modules[alias] = package
    return importlib.import_module(alias + ".SFNO_PRGNO_V3")


def specs(root):
    experiment = root / "Experiments/v3_stage3_restart_20261111"
    with np.load(root / "CAVE/physics/cave_effective_response.npz", allow_pickle=False) as response:
        cave_response = response["response"].copy()
    return {
        "cave": {
            "response": cave_response,
            "checkpoint": root / "Experiments/cave_continuous600/Checkpoint/cave_v3_stage3_x4_seed42/last.pth",
        },
        "pavia": {
            "response": np.load(experiment / "response_pavia.npy", allow_pickle=False),
            "checkpoint": experiment / "Checkpoint/pavia_v3_stage3_x4_seed42/last.pth",
        },
        "chikusei": {
            "response": np.load(experiment / "response_chikusei.npy", allow_pickle=False),
            "checkpoint": experiment / "Checkpoint/chikusei_v3_stage3_x4_seed42/last.pth",
        },
    }


def construct(module, response, ablation="deform_stage3"):
    msi, hsi = response.shape
    return module.SFNO_PRGNO_V3(
        size=64, hschannels=hsi, mschannels=msi, response=response,
        ablation=ablation, guidance_mode="physics",
        use_spatial_residual=True, use_spectral_residual=True,
    ).cpu()


def exact_state(reference, candidate, context):
    old, new = reference.state_dict(), candidate.state_dict()
    assert tuple(old) == tuple(new), f"{context}: state_dict paths/order changed"
    for name in old:
        assert old[name].dtype == new[name].dtype, f"{context}/{name}: dtype changed"
        assert torch.equal(old[name], new[name]), f"{context}/{name}: initial tensor changed"
    return {"state_tensors": len(old), "state_sha256": state_hash(old),
            "registered_parameters": sum(p.numel() for p in reference.parameters())}


def initialization_pair(old_module, new_module, response, ablation, context):
    reset_rng()
    old = construct(old_module, response, ablation)
    old_rng = rng_hashes()
    reset_rng()
    new = construct(new_module, response, ablation)
    new_rng = rng_hashes()
    assert old_rng == new_rng, f"{context}: constructor RNG consumption changed"
    summary = exact_state(old, new, context)
    summary["rng_after_constructor"] = old_rng
    return old, new, summary


def inputs(response, sf, height=16, width=24):
    assert height % sf == width % sf == 0
    msi, hsi = response.shape
    generator = torch.Generator().manual_seed(20261123 + sf + height + width)
    low = torch.rand(1, hsi, height // sf, width // sf, generator=generator)
    high = torch.rand(1, msi, height, width, generator=generator)
    return low, high


def close(reference, candidate, context):
    assert reference.shape == candidate.shape, f"{context}: shape changed"
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all(), context
    torch.testing.assert_close(candidate, reference, atol=ATOL, rtol=RTOL, msg=context)
    return float((candidate - reference).abs().max())


def forward_pair(old, new, response, sf, height=16, width=24):
    low, high = inputs(response, sf, height, width)
    old.eval(); new.eval()
    # Inference-mode caching is an existing, separate DynamicSpatial limitation;
    # no_grad keeps cached constants usable for the subsequent backward test.
    with torch.no_grad():
        reset_rng(81)
        old_out = old(low, high, sf)
        old_rng = rng_hashes()
        reset_rng(81)
        new_out = new(low, high, sf)
        new_rng = rng_hashes()
    assert old_rng == new_rng, "forward RNG changed"
    error = close(old_out, new_out, f"forward x{sf} {height}x{width}")
    return {"sf": sf, "lr_shape": list(low.shape), "msi_shape": list(high.shape),
            "output_shape": list(old_out.shape), "max_abs_error": error,
            "rng_equal": True}


def backward_pair(old, new, response):
    low, high = inputs(response, 4)
    low_old = low.detach().clone().requires_grad_(True)
    high_old = high.detach().clone().requires_grad_(True)
    low_new = low.detach().clone().requires_grad_(True)
    high_new = high.detach().clone().requires_grad_(True)
    old.train(); new.train()
    old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
    reset_rng(82)
    old_output = old(low_old, high_old, 4)
    old_loss = old_output.square().mean()
    old_loss.backward()
    old_rng = rng_hashes()
    reset_rng(82)
    new_output = new(low_new, high_new, 4)
    new_loss = new_output.square().mean()
    new_loss.backward()
    new_rng = rng_hashes()
    assert old_rng == new_rng, "backward RNG changed"
    close(old_output, new_output, "backward forward output")
    old_parameters = dict(old.named_parameters())
    new_parameters = dict(new.named_parameters())
    assert tuple(old_parameters) == tuple(new_parameters), "parameter paths changed"
    errors = []
    absent = []
    for name, parameter in old_parameters.items():
        a, b = parameter.grad, new_parameters[name].grad
        assert (a is None) == (b is None), f"gradient participation changed: {name}"
        if a is None:
            absent.append(name)
        else:
            errors.append(close(a, b, "gradient " + name))
    assert errors, "no trainable gradients were exercised"
    input_errors = {
        "lr_hsi": close(low_old.grad, low_new.grad, "LR input gradient"),
        "hr_msi": close(high_old.grad, high_new.grad, "MSI input gradient"),
    }
    # A backward pass is deliberately not followed by any optimizer update.
    exact_state(old, new, "state after backward")
    return {"sf": 4, "loss": float(old_loss.detach()),
            "parameter_gradients_compared": len(errors), "none_gradients": absent,
            "gradient_max_abs_error": max(errors), "input_gradient_max_errors": input_errors,
            "rng_equal": True, "optimizer_steps": 0}


def modes(model):
    return [block.MDG.ablation for block in (model.block1, model.block2, model.block3)]


def expected_modes(ablation):
    if ablation.startswith("deform_stage"):
        enabled = int(ablation[-1])
        return ["full" if j == enabled else "no_deformable" for j in (1, 2, 3)]
    return [ablation] * 3


def legacy_probe(directory, root):
    """Run only in a fresh subprocess, so top-level imports cannot cross trees."""
    setup_cpu()
    sys.path.insert(0, str(Path(directory).resolve()))
    module = importlib.import_module("SFNO_PRGNO_V3")
    spec = specs(root)["cave"]
    reset_rng()
    model = construct(module, spec["response"])
    initialized = state_hash(model.state_dict())
    initialized_rng = rng_hashes()
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    low, high = inputs(spec["response"], 4, 8, 12)
    with torch.no_grad():
        result = model(low, high, 4)
    assert torch.isfinite(result).all()
    return {"state_sha256": initialized, "rng_after_constructor": initialized_rng,
            "loaded_checkpoint_output_sha256": tensor_hash(result),
            "output_shape": list(result.shape), "ablations": list(module.ABLATIONS)}


def check_legacy(reference_dir, candidate_dir, root):
    results = []
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for directory in (reference_dir, candidate_dir):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--legacy-probe", str(directory),
             "--project-root", str(root)], text=True, capture_output=True,
            env=environment, check=False, timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(f"legacy import failed for {directory}:\n{completed.stderr}\n{completed.stdout}")
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1], "legacy sys.path import behavior changed"
    return {"reference_and_candidate_equal": True, "fresh_subprocesses": 2, **results[0]}


def verify(args):
    setup_cpu()
    reference_dir = args.reference_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    root = args.project_root.resolve()
    assert reference_dir != candidate_dir, "reference and candidate must be isolated directories"
    reference_hashes = {str(p.relative_to(reference_dir)): file_hash(p)
                        for p in reference_dir.rglob("*.py")}
    old_module = isolated_package(reference_dir, "_v3_merge_reference")
    new_module = isolated_package(candidate_dir, "_v3_merge_candidate")
    assert tuple(old_module.ABLATIONS) == tuple(new_module.ABLATIONS), "ABLATIONS changed"
    data = specs(root)
    result = {"status": "running", "torch": torch.__version__, "device": "cpu", "threads": 2,
              "reference_dir": str(reference_dir), "candidate_dir": str(candidate_dir),
              "seed": SEED, "atol": ATOL, "rtol": RTOL,
              "package_import": True, "ablations": [], "datasets": {}}
    state_paths = None
    for ablation in old_module.ABLATIONS:
        old, new, record = initialization_pair(old_module, new_module, data["cave"]["response"],
                                              ablation, "ablation " + ablation)
        if state_paths is None:
            state_paths = tuple(old.state_dict())
        assert tuple(old.state_dict()) == tuple(new.state_dict()) == state_paths
        assert modes(old) == modes(new) == expected_modes(ablation)
        record.update(ablation=ablation, stage_modes=modes(old),
                      forward=forward_pair(old, new, data["cave"]["response"], 4, 8, 12))
        # fixed_grid deliberately differs from the actual x8 observation grid.
        if ablation == "fixed_grid":
            record["forward_x8"] = forward_pair(old, new, data["cave"]["response"], 8, 8, 16)
        result["ablations"].append(record)
        print("ablation passed:", ablation, flush=True)
        del old, new
    for dataset, spec in data.items():
        old, new, initialized = initialization_pair(old_module, new_module, spec["response"],
                                                  "deform_stage3", dataset)
        checkpoint_path = spec["checkpoint"]
        before = file_hash(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 600, f"unexpected checkpoint epoch: {checkpoint_path}"
        state = checkpoint["model"]
        for model in (old, new):
            loaded = model.load_state_dict(state, strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
        record = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": before,
                  "epoch": checkpoint["epoch"], "strict_load": True, "initialization": initialized,
                  "forwards": [forward_pair(old, new, spec["response"], sf) for sf in (4, 8)],
                  "backward": backward_pair(old, new, spec["response"])}
        for name, value in old.state_dict().items():
            assert torch.equal(value, state[name]), f"checkpoint tensor changed during verification: {name}"
        assert file_hash(checkpoint_path) == before, f"checkpoint file changed: {checkpoint_path}"
        result["datasets"][dataset] = record
        print("checkpoint/forward/backward passed:", dataset, flush=True)
        del old, new, checkpoint, state
    result["legacy_import"] = check_legacy(reference_dir, candidate_dir, root)
    assert reference_hashes == {str(p.relative_to(reference_dir)): file_hash(p)
                                for p in reference_dir.rglob("*.py")}, "frozen reference source changed"
    result["reference_source_hashes"] = reference_hashes
    result["candidate_source_hashes"] = {str(p.relative_to(candidate_dir)): file_hash(p)
                                          for p in candidate_dir.rglob("*.py")}
    result["status"] = "passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/verify_v3_shared_merge.json"))
    parser.add_argument("--legacy-probe", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.legacy_probe is not None:
        print(json.dumps(legacy_probe(args.legacy_probe, args.project_root.resolve())))
        return 0
    args.reference_dir = args.reference_dir or args.project_root / "Experiments/v3_stage3_restart_20261111/model"
    args.candidate_dir = args.candidate_dir or args.project_root / "CAVE/Model"
    destination = args.output.resolve()
    protected = [args.reference_dir.resolve(), args.candidate_dir.resolve(),
                 (args.project_root / "Experiments/v3_stage3_restart_20261111").resolve(),
                 (args.project_root / "Experiments/cave_continuous600").resolve()]
    if destination.suffix != ".json" or any(destination.is_relative_to(path) for path in protected):
        parser.error("output must be a JSON outside model and frozen experiment directories")
    started = time.monotonic()
    try:
        result = verify(args)
    except Exception as error:
        result = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        traceback.print_exc()
    result["elapsed_seconds"] = time.monotonic() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(destination),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

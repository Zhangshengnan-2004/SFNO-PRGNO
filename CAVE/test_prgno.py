"""CPU-friendly verification suite for SFNO-PRGNO V1."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

# Some isolated Windows runners omit identity variables required lazily by
# torch._dynamo when an optimizer is first constructed.
os.environ.setdefault("USERNAME", "sfno_test")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.cwd() / ".torch_cache"))

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Model"))


def _install_timm_shim_if_needed():
    """The local torch 2.11 CPU build cannot import installed timm; SFNO uses only these two helpers."""
    # Do not probe-import timm: a failed import can poison torch.compiler in
    # the affected environment before an optimizer is constructed.
    for name in list(sys.modules):
        if name == "timm" or name.startswith("timm."):
            del sys.modules[name]
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)
        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            return x.div(keep) * x.new_empty(shape).bernoulli_(keep)

    layers.DropPath = DropPath
    layers.to_2tuple = lambda x: x if isinstance(x, tuple) else (x, x)
    timm.models, models.layers = models, layers
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})
    return True


TIMM_SHIM = _install_timm_shim_if_needed()
from SFNO import SFNO
from SFNO_PRGNO import SFNO_PRGNO
from PhysicsGalerkin import CAVESpatialDegradation, EffectiveSpectralResponse
from Utils import para_setting


def relative_inner_product_error(lhs, rhs):
    return float((lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12))


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path, default=ROOT / "physics" / "cave_effective_response.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "Checkpoint" / "train_x4" / "last.pth")
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(7)
    response = np.load(args.response, allow_pickle=False)["response"]
    report = {"timm_shim_used": TIMM_SHIM, "tests": {}}

    # Adjoint checks in float64.
    for sf in (4, 8):
        spatial = CAVESpatialDegradation(sf).double()
        x = torch.randn(2, 31, args.size, args.size, dtype=torch.float64)
        y = torch.randn_like(spatial(x))
        error = relative_inner_product_error((spatial(x) * y).sum(),
                                             (x * spatial.adjoint(y, x.shape[-2:])).sum())
        report["tests"][f"spatial_adjoint_sf{sf}"] = {"relative_error": error, "passed": error < 1e-10}
        reference_x = np.random.default_rng(sf).standard_normal((31, args.size, args.size))
        fft_b, _ = para_setting("gaussian_blur", sf, [args.size, args.size], 2.0)
        reference_blur = np.fft.ifft2(np.fft.fft2(reference_x, axes=(-2, -1)) * fft_b,
                                     axes=(-2, -1)).real
        offset = sf // 2 - 1
        reference_lr = reference_blur[:, offset::sf, offset::sf]
        actual_lr = spatial(torch.from_numpy(reference_x).unsqueeze(0)).squeeze(0).numpy()
        max_error = float(np.max(np.abs(reference_lr - actual_lr)))
        report["tests"][f"dataset_db_match_sf{sf}"] = {
            "max_absolute_error": max_error,
            "note": "Difference is limited to the dataset's float32 PSF conversion",
            "passed": max_error < 1e-6}
    spectral = EffectiveSpectralResponse(torch.from_numpy(response)).double()
    x = torch.randn(2, 31, args.size, args.size, dtype=torch.float64)
    y = torch.randn(2, 3, args.size, args.size, dtype=torch.float64)
    error = relative_inner_product_error((spectral(x) * y).sum(), (x * spectral.adjoint(y)).sum())
    report["tests"]["spectral_adjoint"] = {"relative_error": error, "passed": error < 1e-12}

    baseline = SFNO(size=args.size)
    baseline_params = count_parameters(baseline)
    report["baseline_parameters"] = baseline_params
    for sf in (4, 8):
        model = SFNO_PRGNO(size=args.size, response=response, scale_factor=sf)
        model.eval()
        ms = torch.rand(1, 31, args.size // sf, args.size // sf)
        rgb = torch.rand(1, 3, args.size, args.size)
        with torch.no_grad():
            torch.manual_seed(11)
            baseline_output = baseline(ms, rgb, sf)
            torch.manual_seed(11)
            output = model(ms, rgb, sf)
        passed = output.shape == baseline_output.shape == (1, 31, args.size, args.size)
        report["tests"][f"forward_shape_sf{sf}"] = {"shape": list(output.shape), "passed": passed}
        if sf == 4:
            report["prgno_parameters"] = count_parameters(model)

    # Zero initialization and a real backward pass through physics -> Delta Q/K.
    model = SFNO_PRGNO(size=args.size, response=response, scale_factor=4)
    zero_ok = all(torch.count_nonzero(block.GI.delta_qk.weight).item() == 0 and
                  torch.count_nonzero(block.GI.delta_qk.bias).item() == 0
                  for block in (model.block1, model.block2, model.block3))
    report["tests"]["delta_qk_zero_initialization"] = {"passed": zero_ok}
    model.train()
    ms = torch.rand(1, 31, args.size // 4, args.size // 4)
    rgb = torch.rand(1, 3, args.size, args.size)
    target = torch.rand(1, 31, args.size, args.size)
    output = model(ms, rgb, 4)
    torch.nn.functional.l1_loss(output, target).backward()
    grad_norms = [float(block.GI.delta_qk.weight.grad.norm())
                  for block in (model.block1, model.block2, model.block3)]
    report["tests"]["backward_delta_qk"] = {"gradient_norms": grad_norms,
                                                    "passed": all(value > 0 for value in grad_norms)}

    # One optimizer step is the requested minimum training smoke test.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.block1.GI.delta_qk.weight.detach().clone()
    optimizer.step()
    changed = not torch.equal(before, model.block1.GI.delta_qk.weight.detach())
    report["tests"]["one_step_training"] = {"loss": float(torch.nn.functional.l1_loss(output, target).detach()),
                                                    "parameter_changed": changed, "passed": changed}

    # Check baseline checkpoint loading without touching the checkpoint.
    if args.checkpoint.is_file():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        fresh = SFNO_PRGNO(size=args.size, response=response, scale_factor=4)
        missing, unexpected = fresh.load_sfno_state_dict(state)
        expected_prefixes = ("intermediate_reconstruction.", "physics.",
                             "block1.GI.delta_qk.", "block2.GI.delta_qk.", "block3.GI.delta_qk.")
        only_new_missing = all(key.startswith(expected_prefixes) for key in missing)
        report["tests"]["baseline_checkpoint_loading"] = {
            "missing_keys": missing, "unexpected_keys": unexpected,
            "passed": only_new_missing and not unexpected,
        }
        baseline_loaded = SFNO(size=args.size).eval()
        baseline_loaded.load_state_dict(state)
        fresh.eval()
        probe_ms = torch.rand(1, 31, args.size // 4, args.size // 4)
        probe_rgb = torch.rand(1, 3, args.size, args.size)
        with torch.no_grad():
            torch.manual_seed(101)
            baseline_value = baseline_loaded(probe_ms, probe_rgb, 4)
            torch.manual_seed(101)
            prgno_value = fresh(probe_ms, probe_rgb, 4)
        max_error = float((baseline_value - prgno_value).abs().max())
        report["tests"]["checkpoint_initial_behavior"] = {
            "max_absolute_error": max_error, "passed": max_error < 1e-6}
    else:
        report["tests"]["baseline_checkpoint_loading"] = {
            "passed": False, "skipped": True, "reason": f"Missing {args.checkpoint}"}

    report["parameter_difference"] = report["prgno_parameters"] - baseline_params
    report["all_passed"] = all(item.get("passed", False) for item in report["tests"].values())
    output_file = ROOT / "physics" / "prgno_test_report.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output_file}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

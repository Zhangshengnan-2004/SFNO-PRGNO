"""Fit and audit a fixed effective 3x31 spectral response from CAVE training data.

The saved matrix is a training-set fitted effective response, not a measured
sensor spectral response. Validation scenes are held out from fitting; test
scenes are used only for the final diagnostic report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=root / "Data" / "Train")
    parser.add_argument("--test-dir", type=Path, default=root / "Data" / "Test")
    parser.add_argument("--output", type=Path, default=root / "physics" / "cave_effective_response.npz")
    parser.add_argument("--report", type=Path, default=root / "physics" / "cave_effective_response_report.json")
    parser.add_argument("--validation-scenes", type=int, default=4)
    parser.add_argument("--smooth-lambda", type=float, default=1e-4)
    parser.add_argument("--sample-stride", type=int, default=4,
                        help="Pixel stride used for fitting; metrics always use all pixels")
    return parser.parse_args()


def names(directory: Path, split: str):
    return (directory / f"{split}.txt").read_text().splitlines()


def load_pair(directory: Path, name: str):
    hsi = np.asarray(sio.loadmat(directory / "HSI" / f"{name}.mat")["hsi"], dtype=np.float64)
    rgb = np.asarray(sio.loadmat(directory / "RGB" / f"{name}.mat")["rgb"], dtype=np.float64)
    if hsi.shape[:2] != rgb.shape[:2] or hsi.shape[-1] != 31 or rgb.shape[-1] != 3:
        raise ValueError(f"Unexpected shapes for {name}: HSI={hsi.shape}, RGB={rgb.shape}")
    if not (np.isfinite(hsi).all() and np.isfinite(rgb).all()):
        raise ValueError(f"Non-finite values in {name}")
    return hsi, rgb


def accumulate(directory: Path, scene_names, stride: int):
    xtx = np.zeros((31, 31), dtype=np.float64)
    xty = np.zeros((31, 3), dtype=np.float64)
    count = 0
    ranges = []
    for name in scene_names:
        hsi, rgb = load_pair(directory, name)
        x = hsi[::stride, ::stride].reshape(-1, 31)
        y = rgb[::stride, ::stride].reshape(-1, 3)
        xtx += x.T @ x
        xty += x.T @ y
        count += x.shape[0]
        ranges.append({"scene": name, "hsi_min": float(hsi.min()), "hsi_max": float(hsi.max()),
                       "rgb_min": float(rgb.min()), "rgb_max": float(rgb.max())})
    return xtx / count, xty / count, count, ranges


def fit_nonnegative(xtx, xty, smooth_lambda: float):
    d2 = np.diff(np.eye(31), n=2, axis=0)
    penalty = d2.T @ d2
    response = np.empty((3, 31), dtype=np.float64)
    solver = []
    for channel in range(3):
        b = xty[:, channel]
        def objective(w):
            return 0.5 * w @ xtx @ w - b @ w + 0.5 * smooth_lambda * (w @ penalty @ w)
        def gradient(w):
            return xtx @ w - b + smooth_lambda * penalty @ w
        initial = np.maximum(np.linalg.lstsq(xtx + smooth_lambda * penalty + 1e-10*np.eye(31), b, rcond=None)[0], 0)
        result = minimize(objective, initial, jac=gradient, bounds=[(0, None)] * 31,
                          method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-15})
        if not result.success:
            raise RuntimeError(f"Response fit failed for RGB channel {channel}: {result.message}")
        response[channel] = result.x
        solver.append({"channel": channel, "iterations": int(result.nit), "message": str(result.message)})
    return response, solver


def alignment_diagnostic(directory: Path, scene_names, max_shift=2):
    """Compare zero-shift RGB/HSI luminance correlation against nearby shifts."""
    scores = {f"{dy},{dx}": [] for dy in range(-max_shift, max_shift + 1)
              for dx in range(-max_shift, max_shift + 1)}
    for name in scene_names:
        hsi, rgb = load_pair(directory, name)
        a = hsi.mean(-1)[::4, ::4]
        b = rgb.mean(-1)[::4, ::4]
        for key in scores:
            dy, dx = map(int, key.split(","))
            shifted = np.roll(b, (dy, dx), axis=(0, 1))
            scores[key].append(float(np.corrcoef(a.ravel(), shifted.ravel())[0, 1]))
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    best = max(means, key=means.get)
    return {"best_shift_dy_dx": [int(v) for v in best.split(",")],
            "zero_shift_correlation": means["0,0"], "best_correlation": means[best]}


def metrics(directory: Path, scene_names, response):
    absolute_sum = squared_sum = target_squared_sum = 0.0
    count = 0
    per_scene = []
    for name in scene_names:
        hsi, rgb = load_pair(directory, name)
        prediction = np.einsum("oc,hwc->hwo", response, hsi)
        error = prediction - rgb
        n = error.size
        mae = float(np.abs(error).mean())
        rmse = float(np.sqrt(np.square(error).mean()))
        relative = float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(rgb.ravel()), 1e-12))
        per_scene.append({"scene": name, "mae": mae, "rmse": rmse, "relative_l2": relative})
        absolute_sum += np.abs(error).sum()
        squared_sum += np.square(error).sum()
        target_squared_sum += np.square(rgb).sum()
        count += n
    return {"mae": float(absolute_sum / count), "rmse": float(np.sqrt(squared_sum / count)),
            "relative_l2": float(np.sqrt(squared_sum / target_squared_sum)), "per_scene": per_scene}


def main():
    args = parse_args()
    train_names = names(args.train_dir, "Train")
    test_names = names(args.test_dir, "Test")
    if not 0 < args.validation_scenes < len(train_names):
        raise ValueError("validation-scenes must leave at least one fitting scene")
    fit_names = train_names[:-args.validation_scenes]
    validation_names = train_names[-args.validation_scenes:]
    xtx, xty, sample_count, ranges = accumulate(args.train_dir, fit_names, args.sample_stride)
    response, solver = fit_nonnegative(xtx, xty, args.smooth_lambda)

    report = {
        "label": "training-set fitted effective spectral response (not a measured sensor response)",
        "fit_scenes": fit_names,
        "validation_scenes": validation_names,
        "test_scenes": test_names,
        "fit_sample_count": sample_count,
        "sample_stride": args.sample_stride,
        "nonnegative": True,
        "smooth_lambda": args.smooth_lambda,
        "smoothness_note": "A weak second-difference penalty stabilizes an effective response while preserving nonnegativity.",
        "ranges": ranges,
        "alignment": alignment_diagnostic(args.train_dir, train_names),
        "solver": solver,
        "fit": metrics(args.train_dir, fit_names, response),
        "validation": metrics(args.train_dir, validation_names, response),
        "test": metrics(args.test_dir, test_names, response),
        "response_row_sums": response.sum(axis=1).tolist(),
        "response_min": float(response.min()),
        "response_max": float(response.max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, response=response.astype(np.float32),
             fit_scenes=np.asarray(fit_names), validation_scenes=np.asarray(validation_names),
             smooth_lambda=np.asarray(args.smooth_lambda), label=np.asarray(report["label"]))
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("label", "alignment", "fit", "validation", "test",
                                                    "response_row_sums", "response_min", "response_max")}, indent=2))
    print(f"Saved response: {args.output}")
    print(f"Saved report:   {args.report}")


if __name__ == "__main__":
    main()

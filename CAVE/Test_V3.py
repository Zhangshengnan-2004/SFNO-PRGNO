import os
import sys
import argparse
import json
import random
from pathlib import Path

import torch
import torch.utils.data as tud
import numpy as np
from numpy_compat import install as install_numpy_compat

install_numpy_compat()

from Utils import *
from CAVE_Dataset import cave_dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Model"))
from SFNO_PRGNO_V3 import SFNO_PRGNO_V3, ABLATIONS


# ==========================
# 测试参数
# ==========================

parser = argparse.ArgumentParser(
    description="SFNO-PRGNO V3 CAVE Testing"
)

parser.add_argument(
    "--data_path",
    type=str,
    default="./Data/Test/"
)

parser.add_argument("--ablation", choices=ABLATIONS, default=None)
parser.add_argument("--cross-scale", action="store_true", help="Keep V3 learned weights; evaluate on target observation grid")
parser.add_argument("--report", type=Path, default=None)

parser.add_argument("--model", choices=("sfno_prgno_v3",), default="sfno_prgno_v3")
parser.add_argument("--response", type=Path,
                    default=ROOT / "physics" / "cave_effective_response.npz")
parser.add_argument("--guidance-mode", choices=("physics", "learned"), default=None)
parser.add_argument("--disable-spatial-residual", action="store_true", default=None)
parser.add_argument("--disable-spectral-residual", action="store_true", default=None)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="./Checkpoint/train_x4_prgno_v3/last.pth"
)

parser.add_argument(
    "--sf",
    type=int,
    default=4
)

parser.add_argument(
    "--gpu",
    type=int,
    default=0
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1
)

parser.add_argument(
    "--sizeI",
    type=int,
    default=512
)

parser.add_argument(
    "--testset_num",
    type=int,
    default=12
)

parser.add_argument(
    "--seed",
    type=int,
    default=1
)

parser.add_argument(
    "--kernel_type",
    type=str,
    default="gaussian_blur"
)

opt = parser.parse_args()
if opt.report is not None and opt.report.exists():
    raise FileExistsError(f"Refusing to overwrite {opt.report}")
torch.set_num_threads(4)

print(opt)


# ==========================
# GPU
# ==========================

device = torch.device(
    f"cuda:{opt.gpu}"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
if opt.batch_size != 1:
    raise ValueError("Evaluation currently requires batch_size=1")


# ==========================
# 加载测试数据
# ==========================

data_path = os.path.join(
    opt.data_path, ""
)

file_list = loadpath(
    os.path.join(data_path, "Test.txt"),
    shuffle=False
)

HR_HSI, HR_MSI = prepare_data(
    data_path,
    file_list,
    len(file_list)
)

dataset = cave_dataset(
    opt,
    HR_HSI,
    HR_MSI,
    istrain=False
)

test_loader = tud.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=False
)


# ==========================
# 加载自己训练的模型
# ==========================

print("Loading checkpoint:", opt.checkpoint)

checkpoint = torch.load(
    opt.checkpoint,
    map_location="cpu",
    weights_only=False
)

# Restore experiment semantics; state tensors alone cannot identify an ablation.
config = checkpoint.get("config", {})
saved_ablation = config.get("ablation", "full")
if opt.ablation is not None and opt.ablation != saved_ablation:
    raise ValueError("Requested ablation differs from checkpoint")
opt.ablation = saved_ablation
for key, default in (("guidance_mode", "physics"), ("disable_spatial_residual", False),
                     ("disable_spectral_residual", False)):
    saved_value = config.get(key, default)
    requested = getattr(opt, key)
    if requested is not None and requested != saved_value:
        raise ValueError(f"Test configuration differs from checkpoint: {key}")
    setattr(opt, key, saved_value)
# 模型初始化参数需要与训练时保持一致
if opt.disable_spatial_residual and opt.disable_spectral_residual:
    raise ValueError("Cannot disable both spatial and spectral residuals")
model = SFNO_PRGNO_V3(
    ablation=opt.ablation,
    size=opt.sizeI,
    response_path=opt.response,
    scale_factor=opt.sf,
    use_spatial_residual=not opt.disable_spatial_residual,
    use_spectral_residual=not opt.disable_spectral_residual,
    guidance_mode=opt.guidance_mode,
).to(device)

state = dict(checkpoint["model"])
source_sf = checkpoint.get("config", {}).get("sf")
if opt.cross_scale:
    if opt.model not in ("sfno_prgno_v3",) or source_sf is None:
        raise ValueError("Cross-scale testing requires a V2 checkpoint with config.sf")
    # V3 computes target-grid degradation dynamically; no buffer replacement.
    print(f"Cross-scale: trained x{source_sf} -> tested x{opt.sf}; learned weights unchanged")
elif source_sf is not None and int(source_sf) != opt.sf and opt.model != "sfno":
    raise ValueError("Checkpoint scale differs; explicitly use --cross-scale for V2")
model.load_state_dict(state, strict=True)

model.eval()

print("模型权重加载成功")
print("训练轮数:", checkpoint["epoch"])


# ==========================
# 测试
# ==========================

psnr_total = 0.0
sam_total = 0.0
ergas_total = 0.0

count = 0
per_scene = []

with torch.no_grad():

    for j, (LR, RGB, HR) in enumerate(test_loader):

        LR = LR.to(device)
        RGB = RGB.to(device)

        # SFNO模型推理
        output = model(
            LR,
            RGB,
            opt.sf
        )

        output = output.clamp(0, 1)

        result = (
            output.cpu()
            .squeeze(0)
            .numpy()
            .transpose(1, 2, 0)
        )

        gt = (
            HR.cpu()
            .squeeze(0)
            .clamp(0, 1)
            .numpy()
            .transpose(1, 2, 0)
        )

        # 计算评价指标
        psnr = cal_psnr(result, gt)

        sam = compute_sam(result, gt)

        ergas = compute_ergas(gt, result, scale_factor=opt.sf)

        psnr_total += psnr
        sam_total += sam
        ergas_total += ergas

        per_scene.append({"scene": file_list[j], "psnr": float(psnr),
                          "sam": float(sam), "ergas": float(ergas)})
        count += 1

        print(
            f"[{count}/{len(test_loader)}] "
            f"PSNR={psnr:.4f} "
            f"SAM={sam:.4f} "
            f"ERGAS={ergas:.4f}"
        )


# ==========================
# 平均指标
# ==========================

print("\n========== Test Results ==========")

print("Scale:", opt.sf)

print("Test Images:", count)

print("Avg PSNR = %.4f" % (psnr_total / count))

print("Avg SAM = %.4f" % (sam_total / count))

print("Avg ERGAS = %.4f" % (ergas_total / count))

print("==================================")

if opt.report is not None:
    opt.report.parent.mkdir(parents=True, exist_ok=True)
    opt.report.write_text(json.dumps({
        "checkpoint": opt.checkpoint, "epoch": checkpoint["epoch"],
        "model": opt.model, "ablation": opt.ablation, "guidance_mode": opt.guidance_mode,
        "spatial_residual": opt.model != "sfno" and opt.guidance_mode == "physics" and not opt.disable_spatial_residual,
        "spectral_residual": opt.model != "sfno" and opt.guidance_mode == "physics" and not opt.disable_spectral_residual,
        "device": str(device), "ergas_scale_factor": opt.sf,
        "sf": opt.sf, "source_sf": source_sf, "cross_scale": opt.cross_scale,
        "rebuilt_buffers": [],
        "seed": opt.seed, "per_scene": per_scene,
        "mean": {"psnr": float(psnr_total/count), "sam": float(sam_total/count),
                 "ergas": float(ergas_total/count)},
    }, indent=2) + "\n", encoding="utf-8")


# Always save a readable log alongside the tested checkpoint.
from test_result_log import write_result_log
write_result_log({
    "checkpoint": opt.checkpoint, "epoch": checkpoint["epoch"], "sf": opt.sf,
    "per_scene": per_scene,
    "mean": {"psnr": float(psnr_total/count), "sam": float(sam_total/count),
             "ergas": float(ergas_total/count)},
})

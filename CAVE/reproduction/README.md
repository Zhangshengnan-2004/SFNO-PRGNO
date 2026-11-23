# SFNO 复现与 V1 可复现实验包

SFNO-PRGNO 为主实验工程；SFNO-main 为原 SFNO 本地复现工程，完整保留其代码、数据和权重。包含 V1/SFNO 训练权重、optimizer/scaler/config、三个种子的五组消融、×4与×4→×8结果、CAVE训练/测试数据、拟合响应和依赖记录。主工程仅保留 SFNO 和 PRGNO V1。

## 解压与环境
在新目录解压压缩包，先运行 python reproduction/verify.py 核对全部文件。现有 sfno 环境可直接使用；新环境可参考 environment.yml、requirements.freeze.txt 和 conda-explicit.txt。它们记录打包时环境，不含环境二进制或显卡驱动，不能保证与每次历史训练环境完全一致。CUDA版PyTorch需匹配软件源和驱动；历史配置中的旧绝对路径仅作记录，实际运行采用下面的相对路径。

## 重现现有权重测试
进入 SFNO-PRGNO/CAVE，运行以下命令（python 使用已配置的 sfno 环境）：

```bash
python Test.py --model sfno --checkpoint Checkpoint/train_x4/last.pth --data_path Data/Test --sf 4 --seed 1 --report reproduced_baseline_x4.json
python Test.py --model sfno_prgno --checkpoint Checkpoint/train_x4_prgno/last.pth --data_path Data/Test --sf 4 --seed 1 --report reproduced_v1_x4.json
python Test.py --model sfno_prgno --checkpoint Checkpoint/train_x4_prgno/last.pth --data_path Data/Test --sf 8 --cross-scale --seed 1 --report reproduced_v1_x8.json
python scripts/complete_v1_cross_scale.py
```

complete_v1_cross_scale.py 检测已完成报告会跳过；要实际重跑，先将 physics 下 baseline/spatial/spectral/capacity_x4_to_x8_seed*.json 移到其他位置，再执行脚本（不要删除参考结果）。Dual跨尺度测试seed43/44使用 Checkpoint/ablation/x4/dual_seed43/last.pth、dual_seed44/last.pth。

## 从头训练
```bash
python Train.py --model sfno --data_path Data/Train --epochs 200 --batch_size 1 --patch_size 64 --lr 0.0001 --sf 4 --seed 42 --trainset_num 1000 --num_workers 2 --save_dir Checkpoint/reproduce_sfno_seed42
python Train.py --model sfno_prgno --data_path Data/Train --epochs 200 --batch_size 1 --patch_size 64 --lr 0.0001 --sf 4 --seed 42 --trainset_num 1000 --num_workers 2 --save_dir Checkpoint/reproduce_v1_seed42
```

其余种子使用43/44；spatial加 --disable-spectral-residual，spectral加 --disable-spatial-residual，capacity加 --guidance-mode learned。每组选择新的save_dir，避免覆盖。恢复训练用 --resume 原last.pth，--epochs是目标总轮数，保持模型、倍率、seed、分支设置一致。完整预训练权重包含优化器状态，训练全流程未在本次打包时重跑。

## 结果口径
完整三种子表：SFNO-PRGNO/CAVE/physics/v1_ablation_complete.csv 和 .md。seed42 baseline/dual 的×4来自旧日志四位小数，其他指标见对应 metrics.json。×8是×4权重直接测试，已知退化算子按×8重建，无微调。ERGAS×8系数100/8。固定推理seed=1，GPU/CPU及软件差异可能造成数值差别。capacity旧报告残差标记可能为true，但learned模式实际绕过残差计算。

本包保证已保存文件可校验和现有权重可加载；不承诺重新训练逐位复现，也不将历史结果与作者论文数值视为相同实验。

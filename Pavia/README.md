> 2026-11-13：本目录旧版SFNO/V3-stage3实验权重、训练日志及测试结果已按用户要求删除；原始数据及代码保留。下方协议和命令仅作历史说明，不能据此恢复已删除权重。当前训练与结果见[实验索引](../KEPT_RESULTS_20261110.md)。

# Pavia Centre 合成 MSI 基线

本流程复用 `CAVE/Model/SFNO.py`，使用 102 通道 HSI 和 3 通道合成 MSI。
它不是论文 Pavia 结果的严格复现：当前只有原始 HSI，缺少论文使用的 MSI 和精确划分。
不使用 CAVE 权重，从头训练。CAVE 模型默认通道和参数名称保持兼容。

## 协议 pavia_synthetic_v1

- 输入：`Data/Pavia.mat` 中的 `pavia`，尺寸必须为 `(1096,715,102)`。
- 使用固定除数 8000 归一化，不根据测试区域拟合参数。
- 以下坐标均为 Python 0 起始坐标，区间右端不包含。
- 训练：行 `[0,704)`，所有 715 列。随机抽取 64×64 块，随机旋转和水平/垂直翻转。
- 行 `[704,712)` 留空作为隔离区域。
- 测试：四个 192×192 块，左上角为 `(712,0)`、`(712,192)`、`(904,0)`、`(904,192)`。
  这些坐标是本基线自行选择的，不是论文提供的坐标。其余区域不使用。
- 合成 MSI：在波段下标 0–101 上构造三个归一化高斯响应，中心为 20、50、80，标准差为 12。
  每行权重之和为 1。它们不是实际相机响应，不宣称为真实 RGB；矩阵保存为 `response.npy`。
- LR-HSI：s×s、σ=2 的归一化高斯核；偶数核坐标以 `(s−1)/2` 为中心。
  每个训练/测试图块独立做周期边界卷积，左/上补 `s/2−1`，右/下补 `s/2`，从下标 `s/2−1` 开始间隔 s 采样。
- 无噪声。L1 损失；Adam，β=(0.9,0.999)，ε=1e−8。
- 为后续 SFNO/PRGNO 对照，对齐 `CAVE/run_continuous_baseline.py` 的连续 600 轮训练预算：
  batch=1、64×64、每轮 1000 个随机块、seed=42、FP32、初始 LR=1e−4。
- 学习率按绝对完成轮数衰减，与 CAVE 的 `--ablation-lr-schedule` 相同：
  第 1–200 轮 1e−4；201–300 轮 5e−5；301–350 轮 2.5e−5；
  351–500 轮 1.25e−5；501–550 轮 6.25e−6；551–600 轮 3.125e−6。
  不再使用旧版每轮乘 0.95。旧检查点不可直接用于这组对照实验。
- 复用现有模型的默认初始化。论文提到 He 初始化，但未明确逐层规则，此处不声称初始化完全一致。
- 每轮原子更新 `last.pth`，默认每 200 轮保留独立检查点；不根据测试结果选择模型。
- `--resume` 恢复模型、Adam、Python/NumPy/PyTorch/CUDA 随机状态和绝对轮数；
  `--epochs 600` 表示总共 600 轮，而不是再训练 600 轮。恢复时须使用相同数据、尺度、batch、采样数、学习率与种子。
- 这里对齐的是训练预算与学习率；Pavia 的合成 MSI、空间划分和归一化仍遵循上述独立协议，
  退化与指标尚未统一为 CAVE 的同一实现。未来 PRGNO 必须复用本 Pavia 协议进行比较。
- 测试不裁剪输出值；PSNR 为各波段 PSNR 均值，峰值为 1；SAM 单位为度（忽略零范数像素）；
  ERGAS 使用 `100/s × sqrt(mean(MSE_band / mean_GT_band²))`。报告四块的逐块结果及算术均值。

## 运行

在 `/home/zsn/projects` 下：

```bash
# CPU 前向、反向和一次参数更新，不保存文件
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Pavia/run.py smoke --device cpu

# 在可用 CUDA 的机器从头连续训练 600 轮
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Pavia/run.py train --device cuda --sf 4 --batch-size 1 --epochs 600

# 同一 x4 模型分别评估 x4 和跨尺度 x8
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Pavia/run.py test --device cuda --sf 4 --checkpoint SFNO-PRGNO/Pavia/Checkpoint/sfno_continuous600_x4_seed42/last.pth
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Pavia/run.py test --device cuda --sf 8 --checkpoint SFNO-PRGNO/Pavia/Checkpoint/sfno_continuous600_x4_seed42/last.pth
```

`--output` 指定独立实验目录。训练拒绝覆盖非空目录，测试拒绝覆盖同名 JSON。
若另训 x8，必须为其指定新的 `--output`。配置保存数据 SHA256、种子、版本和参数。
用户终端已验证 CUDA 可用；代理执行环境未检测到 CUDA，完整训练需在用户终端启动。

中断后使用相同配置恢复：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Pavia/run.py train --device cuda --epochs 600 --resume SFNO-PRGNO/Pavia/Checkpoint/sfno_continuous600_x4_seed42/last.pth
```

若旧版训练仍在运行，应先在其终端按 Ctrl+C 停止，保留旧目录；新版使用独立的 `sfno_continuous600_x4_seed42` 目录从头训练。
入口支持 SFNO 和完整 V3，见下文。


## Pavia V3 对照实验

使用 `SFNO_PRGNO_V3` 的 `full` 版本（三个阶段均有可变形校正），不加载 CAVE 或 Pavia SFNO 权重。
训练数据、划分、合成 MSI、退化、L1、优化器、学习率日程、batch=1、每轮 1000 块和总计 600 轮均沿用上述 SFNO 基线。
V3 的谱残差使用生成 MSI 的已知 3×102 响应（属于已知 SRF 条件）；空间残差采用与本协议退化匹配的 FFT 算子及其伴随。
不修改已有 SFNO 的实验文件；CAVE V3 的默认通道保持为 31+3。

后台训练并在完成后自动进行 ×4、×8 测试：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Pavia/train_and_test.py --model v3 --detach
```

训练结果：`Checkpoint/v3_full_continuous600_x4_seed42/`。
日志及状态：`Runs/v3_full_continuous600_x4_seed42/` 中的 `train.log`、`test_x4.log`、`test_x8.log`、`status.json`。
自动测试输出与 CAVE 相同的逐块指标及均值文本格式，JSON 指标另存于检查点目录。

中断后恢复训练（自动调度器目前不负责恢复）：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Pavia/run.py train --model v3 --device cuda --epochs 600 --resume SFNO-PRGNO/Pavia/Checkpoint/v3_full_continuous600_x4_seed42/last.pth
```

## V3-stage3：仅最后阶段启用可变形分支

模型入口已支持 `--model v3 --ablation deform_stage3`；三个阶段均保留区域聚合和GI，只有第三阶段执行可变形修正。使用102波段HSI及本协议的已知合成3通道响应，空间退化与伴随同完整V3。与现有baseline/完整V3一致，从头连续600轮、seed42、×4训练、batch1、64×64 patch、1000样本/轮、Adam+L1及相同学习率日程，完成后评估四个固定192×192测试块的×4/×8。

2026-11-06已设置后台任务：等待CAVE三种子队列全部训练测试完成、GPU空闲后开始，不与其并行。

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Pavia/train_and_test.py --model v3 --ablation deform_stage3 --wait-for-cave --detach
```

已启动后不要重复执行。输出：`Checkpoint/v3_deform_stage3_continuous600_x4_seed42/`；日志及状态：`Runs/v3_deform_stage3_continuous600_x4_seed42/`。其中 `train.log`、`test_x4.log`、`test_x8.log` 按阶段生成。模型权重 `last.pth`，指标 `last_test_x4.json` 和 `last_test_x8.json`。

验证：102波段64×64 CPU smoke前反向通过；×4/×8前反向及有限梯度检查通过；确认Stage1/2偏移层无梯度、Stage3偏移层有梯度。


2026-11-06 队列调整：按用户要求，当前 CAVE Stage3 seed43 完成训练和测试后，先运行 Pavia V3-stage3 seed42，再运行 CAVE baseline seed44、Stage3 seed44。当前训练子进程未中断或重启。新调度器为 Experiments/cave_continuous600/History/queue_scripts/run_reordered_comparison.py，总日志为 Experiments/cave_continuous600/History/three_seed_queue/reordered_execution.log；原有各任务 status.json 持续更新。旧队列入口不要重复启动。


测试日志约定（2026-11-06）：自动训练队列完成训练后执行×4/×8测试，测试入口同时在权重目录保存test.log和test_x8.log，包含checkpoint路径、轮数、逐图/块指标及最终均值。保留已有日志，避免截断正在重定向的stdout。历史缺失日志依据已保存JSON报告补齐并标明来源，不重新推理；原Runs/reproduction日志仍保留。

> 2026-11-13：本文是原始位置消融计划。SFNO baseline与V3-stage3的三种子训练/测试已迁移至Experiments/cave_continuous600；下方旧位置仅为历史记录。

# 兼顾 ×4/×8 的 V3 提效实验

目标：减少可变形采样次数，观察两个尺度的 PSNR/SAM/ERGAS 与推理耗时的共同变化，不预设精度会提高。

| 变体 | Stage 1 | Stage 2 | Stage 3 | 每次前向 grid_sample 调用 |
|---|---|---|---|---|
| deform_stage1 | Regional + gated Deformable | Regional | Regional | 9 |
| deform_stage2 | Regional | Regional + gated Deformable | Regional | 9 |
| deform_stage3 | Regional | Regional | Regional + gated Deformable | 9 |
| full | Regional + gated Deformable | Regional + gated Deformable | Regional + gated Deformable | 27 |

所有变体保留空间与光谱物理引导、GI、前端 VectorAttention、原始门控公式；以相同初始化流程保留禁用分支注册参数。减少调用数不等于整体加速三倍，需实测。

协议：seed=42，×4 从头连续训练600轮，batch=1，patch=64，1000 patches/epoch，Adam + L1，AMP关闭。沿用绝对epoch学习率计划。完成后用last.pth测试×4/×8，测试seed=1。完整V3对照也连续训练，排除原有分段重启差异。

独立目录：Checkpoint/v3_balanced_scales；总日志：v3_balanced_scales_execution.log。2026-11-03已启动，顺序为stage1、stage2、stage3、full。

评估时同时报告两个尺度全部三项指标，以及相同输入大小下的GPU耗时和峰值显存；先检查是否存在兼顾两尺度且更快的方案，若存在指标取舍则明确列出，不只按PSNR决定优劣。单种子结果是探索性证据，正式结论应补多种子；测试集探索后的最终结论需要独立验证。

```bash
python run_v3_ablation.py --variants deform_stage1 deform_stage2 deform_stage3 full --output Checkpoint/v3_balanced_scales --execute
python run_v3_ablation.py --output Checkpoint/v3_balanced_scales --summarize
```

CPU检查已通过：所有变体×4/×8前反向、各阶段分支启用与梯度检查；原V3采样几何与序列化回归通过。

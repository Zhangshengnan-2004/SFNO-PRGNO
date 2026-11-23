# V3 消融实验

状态：代码与 CPU 检查已完成。2026-11-02 已在沙箱外启动 no_deformable、no_regional、fixed_gate 三组（seed=42），单卡顺序执行；沙箱内 CUDA 不可见。总日志：CAVE/v3_ablation_three_execution.log。其余组尚未启动，尚无新增最终指标。

## 实验矩阵

| 名称 | 相对完整 V3 的唯一干预 | 目的 |
|---|---|---|
| full | 无 | 同协议完整对照 |
| no_deformable | Mix = Regional(F) | 验证可变形修正分支 |
| no_regional | Mix = F + sigmoid(α)·(Deformable(F)−F) | 验证区域聚合，保留原修正门控 |
| fixed_gate | α 固定为 −3，修正权重约 0.0474 | 验证学习阶段级修正强度的作用 |
| fixed_grid | 区域分支和可变形采样都固定使用 HR/4 观测网格；物理算子仍按实际尺寸 | 验证混合模块的跨尺度网格适配 |
| spatial_only | 去除光谱观测残差 | 验证光谱约束 |
| spectral_only | 去除空间观测残差 | 验证空间约束 |
| learned_guidance | 用 lift(X̂) 代替观测残差反投影 | 同引导头容量的纯学习引导对照 |

fixed_grid 在 ×4 训练期间与 full 相同，差异集中在 ×8 推理；不能将它解释为物理算子消融。no_regional 保留小门控，衡量的是当前公式中区域主分支的贡献，非重新调参后的最佳纯可变形网络。learned_guidance 仍保留可学习 Q/K 调制，不等同于完全移除 GI。

为保持原 V3 权重键和初始化流程，禁用分支仍注册但不参与前向，梯度为空；登记参数总数不能作为这些变体的有效容量或计算量。尚不包含等计算量替代分支对照。

## 公平比较协议

- 全部变体从头训练，不加载完整 V3 的 e600 权重。
- CAVE 原 20 训练 / 12 测试场景；训练 ×4；HR patch=64，batch=1，1000 patches/epoch。
- Adam + L1，AMP 关闭，训练种子先用 42；正式报告补齐 43、44，报告均值与标准差。
- 与现有续训阶段对齐：1–200 epoch LR=1e-4；201–300=5e-5；301–350=2.5e-5；351–500=1.25e-5；501–550=6.25e-6；551–600=3.125e-6。
- 单次连续训练至 600 轮，不在 200/400 轮重启；使用实验目录下 last.pth，测试随机种子 1，测试 ×4 和 ×8。第一组切换时从第 277 轮恢复，因此此前存在 200 轮续训边界；此后不再分阶段。
- 使用现有拟合光谱响应；不使用测试指标选 checkpoint。禁止用 200 epoch 消融与原 600 epoch 基线比较。
- 默认独立重训 full。历史 e600 指标可作参考；最终表以同一队列的 full 为对照。
- Test_V3 自动从 checkpoint 恢复消融和物理配置，显式冲突会报错；训练恢复也检查配置。

## 使用

在 CAVE 目录执行（默认仅打印命令）：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python run_v3_ablation.py
/home/zsn/miniconda3/envs/sfno/bin/python run_v3_ablation.py --execute
```

单卡顺序执行，所有输出在 Checkpoint/ablation_v3/，拒绝覆盖已存在的实验。中断后可通过 --resume-existing 从实验目录的 last.pth 恢复；已完成实验需要从 --variants 中排除。新的多种子完整队列用新目录：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python run_v3_ablation.py --execute --seeds 42 43 44 --output Checkpoint/ablation_v3_3seeds
/home/zsn/miniconda3/envs/sfno/bin/python run_v3_ablation.py --summarize
```

汇总生成 comparison.csv（每个实验、尺度一行）；多种子的均值和标准差需基于完整配对结果另行统计。缺失实验不会被当作零分。

验证：test_v3_ablation.py 覆盖默认权重一致性、混合公式、×4/×8 前反向、禁用分支梯度和物理引导模式；原 test_prgno_v3.py 验证默认模型采样几何与序列化。

## 连续训练切换记录

按用户要求，三组队列已切换为连续 600 轮：no_deformable 从第 277 轮 checkpoint 恢复，合并 loss 记录到实验根目录；e200/e400 历史目录保留。其余两组从头训练。新总日志为 CAVE/v3_ablation_three_continuous_execution.log，训练日志位于 Checkpoint/ablation_v3/*_train_continuous.log。三组采用相同的绝对 epoch 学习率；不再在 400 轮重置随机种子。

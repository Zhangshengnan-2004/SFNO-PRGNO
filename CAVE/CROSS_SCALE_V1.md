# V1 ×4 → ×8 跨尺度泛化

不训练、不微调；加载 Checkpoint/train_x4_prgno/last.pth（seed42，200 epoch）的所有可学习参数及固定光谱响应，只将 physics.spatial.psf 替换为 ×8 模型构建的匹配数据退化核。模型的 scale_factor、采样相位按 ×8 构建。其余状态严格加载；checkpoint 原文件不改动。

运行（在 CAVE 目录）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/zwu/miniconda3/envs/sfno/bin/python -u Test.py --model sfno_prgno --checkpoint Checkpoint/train_x4_prgno/last.pth --sf 8 --cross-scale --data_path Data/Test --report physics/v1_x4_to_x8_seed42.json > test_v1_x4_to_x8_seed42.log 2>&1
```

有 CUDA 时自动使用 GPU；可用 CUDA_VISIBLE_DEVICES 指定物理卡。当前首次执行使用 CPU。测试 seed=1，12 个 CAVE 测试场景，HR 尺寸512。ERGAS 的倍率系数由原来固定100/4改为100/sf；其余沿用既有实现。与 ×4 的 ERGAS 数字比较时应注意倍率系数不同。

这是已知目标退化算子的跨倍率测试，不等于未知退化条件下的盲泛化，也不能用一次 ×8 测试宣称任意尺度均有效。尚未包含 ×8 专门训练或 baseline 的跨尺度对照。

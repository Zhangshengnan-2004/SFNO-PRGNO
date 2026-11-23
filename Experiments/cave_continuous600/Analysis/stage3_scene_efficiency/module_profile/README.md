# V3-stage3 模块瓶颈测量

独立脚本：../profile_modules.py，不修改模型或checkpoint。

条件：RTX3060Ti，seed42连续600轮V3-stage3权重，batch1，HR256×256，FP32/TF32关闭，eval+inference_mode；×4/×8各预热10次、测30次。随机GPU输入，不含数据加载和传输。不是训练前反向profile。

方法：

1. 无hook测量整网基准耗时。
2. 在三个VectorAttention、三个Regional、Stage3 Deformable、三个GI和共享physics模块注册pre/post hooks，用CUDA event记录开始与结束；仅在整网前向结束同步。每个forward先按类别加总，再对30次取中位数。验证调用次数为3/3/1/3/3，各测量模块互不嵌套，避免重复累加子模块时间。
3. 单独进行3次显存探测；每个模块入口同步，记录memory_allocated并reset_peak_memory_stats；出口同步，记录max_memory_allocated减去入口值。该数字是单次执行产生的额外峰值，包括输出及临时张量，不是模块参数大小，也不是整网峰值。
4. 移除hooks后另测整网峰值。各模块的峰值不能相加。
5. --trace额外保存一次×4前向的Chrome trace和算子级汇总；trace记录有额外开销，不作为benchmark耗时。

Physics只包含model.physics（空间/光谱残差、伴随、lift），不包含共享中间重建头。整网时间与五类之和的差值还包括上采样、初始/最终投影、重建头、门控修正的外层加减及hook开销。

×4无hook整网中位数115.83ms，加hook116.33ms；×8分别115.54和115.59ms。本次额外开销较小，仍需按设备/分辨率重新测量。

主要发现：VectorAttention约34%且单次额外峰值570MiB；Regional约26%；GI约22%；Deformable约10%；Physics约4%。建议优先实验前端减层/邻域中间张量优化，再评估Regional的FFN或宽度；暂不优先移除计算占比较小的物理引导。占比说明时间上限，不保证某结构变更带来等比例提速，也不说明精度影响。

复现（GPU空闲时执行）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Experiments/cave_continuous600/Analysis/stage3_scene_efficiency/profile_modules.py --trace
```

文件：summary.md（表格），profile.json（每层、每尺度完整数据），operators_x4.txt（算子表），trace_x4.json（时间线，可用兼容Chrome trace的查看器打开）。脚本支持--size、--iterations和--output；默认重复运行会覆盖本目录profile结果，旧模型权重不变。

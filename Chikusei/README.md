> 2026-11-13补充：protocol_v2与protocol_v3_joint_loss两个旧协议目录（含代码、响应副本及验收文件）已按用户要求整体删除。当前实验在Experiments/sfno_restart_20261111与Experiments/v3_stage3_restart_20261111，使用各自冻结的响应和代码；本目录BSQ/HDR原始数据保留。

> 2026-11-13：本目录旧版SFNO/V3-stage3实验权重、训练日志及测试结果已按用户要求删除；原始数据及代码保留。下方协议和命令仅作历史说明，不能据此恢复已删除权重。当前训练与结果见[实验索引](../KEPT_RESULTS_20261110.md)。

> 本文档仅保留旧v1协议的历史说明，不用于当前训练。

# Chikusei SFNO / V3-stage3 对照

本地合成 MSI 协议 `chikusei_synthetic_v1`，并非论文精确复现。
直接读取上传的 ENVI BSQ：2517×2335×128、little-endian int16，固定除以文件头标注的10000。
保留全部128波段；含负数、超过10000（包括15000无效标记）的训练块拒绝采样，测试块启动时验证。
空间划分（Python零起始、右端不包含）：训练行[0,1792)、全部列；隔离行[1792,1800)；测试四个192×192块，左上角(1992,0)、(1992,576)、(2184,0)、(2184,576)。其余区域不用。
合成3通道MSI：128个波段索引上的归一化高斯响应，中心25/63/100，标准差15；不是标定相机响应。
沿用Pavia退化：s×s、sigma=2高斯，周期边界、半像素中心、采样相位s/2−1；V3使用对应FFT算子及伴随。

两个模型均从头训练，无预训练权重；seed42，×4训练600轮，batch1，patch64，每轮1000随机块、旋转翻转、FP32、Adam+L1、初始lr1e-4，在完成200/300/350/500/550轮后减半。
V3-stage3复用CAVE模型：所有阶段保留区域聚合/GI，仅第三阶段启用可变形分支；已知合成响应用于谱残差。
两个模型共用同一数据和配置，最终last.pth分别评估×4及跨尺度×8，不按测试集选择检查点。
PSNR为峰值1的逐波段平均，SAM为度，ERGAS按尺度归一化；不裁剪预测。
每轮保存last.pth，每200轮保留独立权重；config记录BSQ+HDR联合SHA256及版本，response.npy记录响应。

一次启动，依次执行baseline训练→×4/×8测试→stage3训练→×4/×8测试：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Chikusei/run_comparison.py --detach
```

不要重复启动。总状态/日志：`Runs/comparison_seed42/{status.json,execution.log}`。
各模型日志：`Runs/{sfno,v3_deform_stage3}_continuous600_x4_seed42/`。
权重/指标：`Checkpoint/{sfno,v3_deform_stage3}_continuous600_x4_seed42/`，含`last.pth`、`train.csv`、`test.log`、`test_x8.log`、`last_test_x4.json`、`last_test_x8.json`。
若中断先检查状态；单模型可用run.py train及同配置的--resume路径恢复，自动队列拒绝重复覆盖。

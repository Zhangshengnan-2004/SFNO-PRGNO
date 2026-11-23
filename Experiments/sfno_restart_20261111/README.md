# SFNO六组重启：正式训练持久保存，按更新次数监控

2026-11-11按用户要求重新开始Pavia、Chikusei的seed42、43、44，共六组。所有正式模型从头训练，不加载先前权重；独立于此前失败的临时预检和batch1稳定版。旧记录保留，CAVE及截图四组历史结果不动。

## 固定训练协议

| 项目 | 设置 |
|---|---|
| 预算 | 每种子600轮×1000随机块；有效batch50，每轮20次Adam更新，总12000次 |
| 模型与初始化 | 原SFNO结构，保留公开模型构造器默认初始化；取消全层He重置 |
| batch | microbatch5×10梯度累积；同一有效batch共享PR随机置换 |
| Adam | 初始lr2e-4，beta=(0.9,0.999)，eps=1e-8 |
| 学习率 | gamma0.95，每6轮衰减；第600轮约1.25e-6 |
| 数据 | 64×64，旋转/翻转；保留已核验的MSI响应、归一化、空间划分和s×s sigma2退化 |
| 损失 | GT非零有效像素上的masked L1；不加入联合光谱/相对损失 |
| 精度 | FP32，AMP/TF32关闭，确定性算法，workers=0 |
| 测试 | ×4训练，第600轮固定权重测试×4和跨尺度×8；测试seed1 |

论文明确了64块、增强、高斯核、相机、Adam参数、batch50、lr2e-4和gamma0.95。初始化逐层规则、衰减间隔、总更新数、完整loss、归一化和作者精确输入未确认。本版的默认初始化是明确例外，每6轮的间隔是实现约定；不能称为严格论文复现。来源和差异见paper_alignment.json，连同源码和数据SHA256冻结在protocol.json。

## 已排除的流程问题

- 不再用临时目录跑长收敛预检；不再用“40轮内L1必须比Bicubic低10%”作为启动门槛。
- 启动前仅做技术验收：真实SFNO梯度累积、PR随机状态、两数据集全尺寸×4/×8确定性推理、真实模型和Adam断点恢复、监控前后随机状态/模式保持。
- 用故意输出错误尺度的轻量模型触发生产代码停止分支，验证停止前检查点确实落盘。
- Preflight/内的小预算检查点和测试日志持久保存，明确属于技术检查，不是正式结果。
- 正式六组直接按固定600轮预算训练，每轮原子保存模型、Adam、全部随机状态及连续质量失败计数。每120轮另存快照。

## 训练期间的停止条件

数值异常立即停止：loss、梯度或更新后的参数非有限。初始probe L1>2拒绝启动；第1轮平均L1>1，或达到500次Adam更新后平均L1>0.1时停止，这些都是拦截严重输出尺度问题的工程阈值，并非论文规定。

每20轮记录固定训练区probe的L1、PSNR、SAM、ERGAS和对应Bicubic值。2000次Adam更新前只记录，不因暂时落后Bicubic而停止；2000次更新后，连续3个检查点的L1和PSNR未同时优于Bicubic才停止。一次改善即清零连续失败计数。当前配置最早在第140轮（2800次更新）因连续收敛失败停止。

上述有限数值质量停止都在保存当前完整轮次检查点和stop_reason.json后执行；NaN等非法数值保留此前最后有效检查点。恢复时严格校验协议、源码和配置，连续失败计数随检查点保存，不会丢失或重置。

所有probe来自训练区，只用于运行监控。最终测试集不参与调参、早停或检查点选择。最终测试报告附Bicubic均值及三项指标差值，不隐藏退化结果。这些措施针对已出现的流程错误，不保证所有最终指标。

## 数据与指标

Pavia：1096×715×102，/8000；训练[0,704)行，四个192测试块起点(712,0)、(712,192)、(904,0)、(904,192)。使用已核验的102×4响应合成MSI，物理传感器及与作者响应的一致性未确认。

Chikusei：2517×2335×128，/10000；Canon EOS5D MarkII响应按HDR波长插值；左上1500×2000的12个500块训练，下方八块测试；×8取左上496。作者精确坐标和论文六块/八块冲突未确认。测试坏观测填充供输入使用，指标排除原无效像素及GT零背景。

统一common_hsi_eval_v1：预测裁剪[0,1]，GT非零和sensor-valid共同掩码；逐波段PSNR均值、SAM度数、ERGAS实际100/s。不能把旧协议分数直接当作当前论文复现分数。

## 队列和结果

Pavia42 → Chikusei42 → Pavia43 → Chikusei43 → Pavia44 → Chikusei44。每组完成600轮和×4/×8测试后进入下一组。

- Runs/status.json：队列状态；Runs/execution_*.log：执行日志。
- Checkpoint/{dataset}_sfno_x4_seed{seed}/：config.json、progress.json、train.csv、last.pth、每120轮快照、health_initial.json、health_epoch_*.json、health_latest.json及最终测试日志/JSON。
- train.csv记录epoch、lr、L1、optimizer_steps、sampled_patches。
- validation.json/log：技术验收证据，不表示正式模型收敛或已完成训练。
- Runs/per_seed.csv、summary.md/json：各种子测试；三种子齐全后生成mean_std.csv，样本标准差ddof=1。

中断后先确认原队列退出，同协议可恢复：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Experiments/sfno_restart_20261111/run_queue.py --resume --detach
```

若因质量条件停止，先阅读stop_reason.json和health_latest.json，不自动绕过失败条件或改变冻结参数。

## 加入V3-stage3后的联合调度

2026-11-11按用户要求，在Chikusei SFNO seed42训练和测试结束后先运行两数据集的V3-stage3 seed42，之后交错执行seed43/44。当前GPU训练子进程保持原PID和启动时间，旧独立派发器由外层调度接管。总计12组、24份×4/×8报告，状态及恢复入口见[联合调度说明](../sfno_v3_queue_20261111/README.md)。本目录冻结源码和协议未改变。

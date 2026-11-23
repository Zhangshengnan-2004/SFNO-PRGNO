# Pavia / Chikusei：SFNO恢复收敛后的三种子实验

2026-11-11按用户要求删除本目录失败的Pavia seed42、Chikusei seed42及中断的Pavia seed43权重、训练/测试日志和旧汇总；六组全部从头训练。截图保留的四组历史结果、CAVE和其他项目不变。删除记录见restart_native_20261111.json，先前推理诊断保留在diagnosis_20261111.json。

本版保持修正后的数据输入与评价，恢复原SFNO运行代码的初始化及优化设置；属于受控复现实验，不能称为完整对齐论文的训练协议。只训练SFNO，V3-stage3未加入队列。

## 本次修复

| 项目 | 当前设置 |
|---|---|
| 初始化 | 保留SFNO及PyTorch各层构造器默认值；取消对所有Conv/Linear统一He重置 |
| 预算 | 600轮×1000随机块，batch1，无梯度累积，共600000次Adam更新/种子 |
| 优化器 | Adam，beta=(0.9,0.999)，eps=1e-8，初始lr=1e-4 |
| 学习率 | 完成200、300、350、500、550轮后各乘0.5；第600轮3.125e-6 |
| 损失 | 每张图GT非零有效像素上的masked L1；两数据集统一，不加入联合光谱/相对损失 |
| 训练与测试 | ×4训练，固定第600轮权重测试×4及跨尺度×8；测试seed固定1 |
| 精度 | FP32，AMP和TF32关闭；确定性算法；workers=0 |
| 保存 | 每轮last.pth，保存模型、Adam和Python/NumPy/Torch/CUDA随机状态；每120轮额外快照 |

旧失败版本的有效batch50使总更新数降到12000次；本版恢复到历史batch1的600000次。这明确偏离论文中记录的batch50、初始lr2e-4、衰减系数0.95；作者完整训练预算及逐层初始化仍待确认。保留构造器默认初始化，也不等于作者确认的He实现。

## 启动前收敛验收

validate.py检查600000次更新预算和全部学习率边界、真实batch1前后向、初始L1<2、两数据集全尺寸×4/×8确定性推理、真实入口的模型及Adam断点恢复逐张量一致。

额外对两个数据集分别运行seed42短程训练。四个固定probe由seed1729从各自训练区域生成，只用于训练收敛检查，不用正式测试集挑配置或权重。在预先固定的第3、5、10轮依次验收：probe L1低于初始值的10%且低于Bicubic的90%，probe PSNR高于Bicubic至少1dB。首次通过即结束该临时检查；10轮仍不通过则不启动正式队列。检查权重在临时目录自动删除，正式六组均从头开始。

正式训练还有两项自动停止条件：第1轮平均L1>1，或第10轮平均L1>0.05。队列遇到失败退出，不会继续消耗后续五组预算。所有这些阈值用于拦截明显异常，不保证最终论文指标。

正式队列只有在validation.json为passed、training_quality_gate为true且协议SHA256一致时启动。

## 保持的数据与指标协议

Pavia：原始1096×715×102，固定/8000；训练行[0,704)和全部715列；隔离行[704,712)；四个192测试块起点(712,0)、(712,192)、(904,0)、(904,192)。使用SHOTUN处理文件中的102×4响应合成四通道MSI。其REF与原图起点(246,235)的256块/8000一致，REF@srf与MSI一致；该核验块处于训练区域，不是测试图。矩阵物理传感器及其与SFNO作者响应的一致性未确认。

Chikusei：原始2517×2335×128，固定/10000；Canon EOS 5D Mark II响应按HDR波长插值，380–780nm外补零、行归一化；左上1500×2000的12个500块训练。八个测试块起点为行1500/2000与列0/500/1000/1500组合；×4尺寸500，×8取左上496。作者精确测试坐标和论文六块/八块差异仍未解决。

两个数据集均使用64×64随机块和旋转/翻转；拒绝负数、超量程和全零训练块。测试无效观测按波段最近有效像素填充供输入使用，指标排除原无效像素及GT全零背景。

LR-HSI使用s×s、sigma2高斯，半像素中心、周期边界、采样起点s/2−1，无附加噪声。边界及相位属于本实现约定。

统一指标common_hsi_eval_v1：GT非零及sensor-valid共用掩码，预测裁剪[0,1]；PSNR逐波段均值，SAM角度制、有效GT上的零预测计90度，ERGAS使用实际100/s。每份测试同时报告Bicubic，避免只看模型自身指标。最终固定预算权重，不以测试分数早停或选模型。

## 六组队列及输出

Pavia42 → Chikusei42 → Pavia43 → Chikusei43 → Pavia44 → Chikusei44。每组600轮后完成×4/×8测试，再进入下一组。GPU有其他计算进程时等待。

- 队列状态：Runs/status.json；执行日志：Runs/execution_*.log。
- 单组日志：Runs/{pavia,chikusei}_seed{42,43,44}/。
- 单组输出：Checkpoint/{pavia,chikusei}_sfno_x4_seed{42,43,44}/，含config.json、response.npy、progress.json、train.csv、last.pth、epoch_120/240/360/480/600.pth及最终测试日志和JSON。
- train.csv：epoch,lr,L1,optimizer_steps,sampled_patches。
- 汇总：Runs/per_seed.csv、summary.json、summary.md；全部12份测试完成后生成mean_std.csv。先对各测试块取均值，再对seed42/43/44算均值和样本标准差ddof=1；未齐三种子不生成完整三种子标准差。

不要重复启动。需要恢复时，在确认原进程退出后执行：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Experiments/sfno_corrected_20261110/run_queue.py --resume --detach
```

protocol.json冻结原始数据、响应、源码SHA256与当前配置。改变源码或配置须重新冻结与验收，不得将旧协议权重混入本次实验。

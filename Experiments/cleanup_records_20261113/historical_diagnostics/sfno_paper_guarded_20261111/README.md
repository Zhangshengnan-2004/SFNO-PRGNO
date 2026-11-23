# SFNO：论文已知设置与训练质量监控

范围：Pavia、Chikusei，seed42/43/44六组；600轮×1000块/轮，×4训练，固定最后一轮权重评估×4和跨尺度×8。CAVE及截图保留的四组历史结果不变。

本目录独立于先前恢复稳定的sfno_corrected_20261110，避免在长任务中途改变初始化、batch或学习率。先前稳定版权重保留；本版仅在精确冻结协议验收通过后启动。

## 论文核对

逐项依据项目内论文PDF第7页4.1节及作者公开仓库核对；paper_alignment.json保存论文SHA256、页码、已知设置和未确认细节，protocol.json冻结其散列。

| 项目 | 本版 | 核对状态 |
|---|---|---|
| 训练块与增强 | 64×64，旋转/翻转 | 论文明确 |
| LR-HSI | s×s高斯核，sigma2，s倍降采样 | 核大小/标准差明确；边界与相位未明确 |
| Chikusei MSI | Canon EOS5D MarkII响应，按实际波长插值 | 相机明确；矩阵是否与作者逐值相同未确认 |
| 优化器 | Adam，beta0.9/0.999，eps1e-8 | 论文明确 |
| 初始学习率 | 2e-4 | 论文明确 |
| 衰减系数 | 0.95 | 论文明确；完整可执行调度未公开 |
| 衰减间隔 | 每6轮，即6000块 | 本实现约定，不能宣称论文确认；最后一轮约1.25e-6 |
| 有效batch | 50，microbatch5×10 | 匹配论文batch数；因8GB GPU使用累积，PR随机置换在同一有效batch内共享 |
| 初始化 | 保留公开模型各层构造器默认值 | 明确例外：论文引用He，但逐层规则缺失；不再使用已实测导致严重放大的全层He重置 |
| 总预算 | 600轮×1000块，共12000次Adam更新/种子 | 用户固定预算，论文没有给出唯一可复现的总更新数 |
| 损失 | GT非零有效像素masked L1 | 本实现约定；作者完整loss未确认 |

batch50使每轮20次Adam更新；不能与batch1下的每轮1000次更新混淆。所有日志记录optimizer_steps和sampled_patches。衰减间隔、初始化、归一化、Pavia MSI来源和精确测试坐标仍有差异，本版不能称为严格论文复现。

## 防止异常重复消耗整轮预算

1. 冻结源代码、数据、SRF、配置及论文核对报告SHA256；恢复必须与冻结配置完全一致。
2. GPU验证两个数据集的真实microbatch5×10、全尺寸×4/×8确定性推理与真实模型/Adam断点恢复。
3. 用真实SFNO比较batch4与microbatch2×2的梯度和PR随机状态，梯度相对L2误差必须小于1e-4；正式同样共享PR置换。此检查不宣称跨batch GPU算术逐位相等。
4. 启动前分别在两个数据集上做短程收敛检查，只用训练区固定probe。预定第10、20轮检查，之后每5轮检查至40轮，L1必须低于初始值10%和Bicubic90%，PSNR高于Bicubic至少1dB；失败不启动正式队列。
5. 正式每组开始前检查初始probe L1<=2；每次检查loss、梯度、更新后参数有限；第1轮L1<=1、第10轮<=0.05。每20轮记录固定训练区probe；达到2000次Adam更新后，probe必须在L1和PSNR上优于Bicubic，否则停止整个队列并保留已保存权重和失败记录。数值异常检查不等待预热。
6. probe检查前后恢复全部随机状态和model模式，不改变训练采样轨迹。probe不使用最终测试集；不依据最终测试分数挑超参数或挑检查点。

这些措施可拦截此前的输出尺度异常、明显不收敛和配置混用，不能保证最终分数或消除所有数据协议差异。

## 数据及评价

保持此前已核验的数据协议：Pavia为102通道HSI/8000、SHOTUN提供的102×4响应、上方704行训练，下方四个192块测试；Chikusei为128通道/10000、上方1500×2000的12个500块训练，下方八块测试，×8取496。训练和测试空间不重叠。

论文Pavia真实MSI来源不明；当前四通道由GT通过已核验响应合成。Chikusei论文4.1八块与4.2六块存在冲突，当前采用八块，但作者精确坐标未确认。不能把这些设定冒充作者原始输入。

common_hsi_eval_v1：共用GT非零及sensor-valid掩码，预测裁剪[0,1]；逐波段PSNR均值，SAM度数、有效GT上的零预测计90度，ERGAS实际100/s。正式测试同时报告Bicubic。

## 队列和结果

Pavia42 → Chikusei42 → Pavia43 → Chikusei43 → Pavia44 → Chikusei44。每组完成600轮及×4/×8测试后才进入下一组；不终止其他项目任务。

- Runs/status.json：队列状态；Runs/execution_*.log：执行日志。
- Checkpoint/{dataset}_sfno_x4_seed{seed}/：config.json、train.csv、progress.json、last.pth、每120轮快照、health_initial.json、health_epoch_*.json及最终测试日志/JSON。
- validation.json/log：冻结协议验收及训练区probe结果；它们不是正式测试成绩。
- Runs/per_seed.csv、summary.md/json：按训练种子汇总；三个种子齐全后计算样本标准差ddof=1并生成mean_std.csv。

需恢复时先确认旧进程退出，再执行：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Experiments/sfno_paper_guarded_20261111/run_queue.py --resume --detach
```

预检监控修正记录：v7在第20轮被正式质量门槛提前中止，与预先允许至40轮的收敛检查冲突；v8按更新次数区分数值异常与收敛预热，启动前优于Bicubic的判定标准不变。记录见preflight_v7_record.json及validation_v7_interrupted_at20.log。

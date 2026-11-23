# Pavia / Chikusei：V3-stage3三种子，与当前SFNO匹配

本目录独立保存V3-stage3源码快照、协议、检查点和日志。SFNO基线位于../sfno_restart_20261111/，其冻结源码和正在运行的权重不修改。六组V3从头训练，不从SFNO或旧V3检查点初始化。

模型固定为SFNO_PRGNO_V3(ablation="deform_stage3")，physics引导，空间及光谱残差均开启；只有第三阶段执行可变形分支。保留模型构造器的零初始化头/偏移及小门控，不做全层He重置。

数据、SRF、masked L1、600轮×1000块、有效batch50、初始lr2e-4、gamma0.95/每6轮、FP32、最后第600轮×4/×8测试及统一指标均与当前SFNO相同。每轮保存模型、Adam、随机状态、监控计数；收敛预热2000次更新、连续3次失败才停止，数值异常立即停止，有限数值质量停止前先保存。

物理引导采用matched_spatial.py中的退化及伴随，与当前数据生成的s×s、sigma2、周期边界、采样相位一致。Pavia为102HSI/4MSI，Chikusei为128HSI/3MSI；响应直接使用对应冻结矩阵，禁止沿用旧Pavia人工三通道MSI。

seed固定不等于所有CUDA算子逐位确定性。V3的grid_sample和抗锯齿resize反向采用warn-only确定性设置，关闭TF32，SDPA使用math后端。断点恢复模型和Adam以atol2e-5、rtol2e-4检查，记录实测最大差异；不宣称和SFNO一样逐位完全一致。

CPU检查已覆盖两个数据集前反向、仅Stage3偏移有梯度、×4/×8空间算子及伴随。小规模CUDA检查不影响当前SFNO子进程。完整GPU技术验收（真实micro5×10、全尺寸推理、梯度累积、断点恢复和停止前保存）由联合调度在当前Chikusei SFNO seed42训练及×4/×8测试结束后执行，通过才开始V3 seed42。

输出：Checkpoint/{pavia,chikusei}_v3_stage3_x4_seed{42,43,44}/；日志：Runs/{pavia,chikusei}_seed{42,43,44}/。汇总与调度入口见../sfno_v3_queue_20261111/README.md。论文未明确的输入、初始化及衰减间隔仍按对比实验说明披露，不声称完全复现论文。

## 2026-11-12缓存修复

正式训练前的完整验收发现：健康推理先生成的FFT核缓存是inference tensor，随后反向传播报错。现固定核在inference_mode(False)+no_grad中创建，数值及空间算子不变。增加了先推理再训练反向的CPU/GPU回归检查，完整GPU验收通过，模型断点恢复最大权重差约4.73e-7。旧协议、失败报告及日志均保留；已完成SFNO权重和报告未修改。

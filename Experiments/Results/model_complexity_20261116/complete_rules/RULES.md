# 参数量与算法 FLOPs 计数规则

统计范围：当前已完成实验的 SFNO、SPGFNet=V3-stage3（deform_stage3）、SSR-NET；CAVE/Pavia/Chikusei；×4/×8，合计18组。CPU FP32，batch=1，HR=64×64；LR分别16×16、8×8。不是整张测试图计算量，不对应原论文尚未公开确认的统计口径。不同数据集通道分别31/3、102/4、128/3。

## 总量定义与限制

总数为卷积/矩阵乘核心计数 + 本文列出的其余算子规则。一次乘加=2。普通实数加减乘除各1，sqrt/rsqrt/exp/log/tanh/erf各按1个算法操作；ReLU比较按1，leaky-ReLU比较+乘法按2。特殊函数的1次不意味着硬件只用1条指令。结果是明确约定下可复现的算法FLOPs，不是CPU/GPU机器指令的精确计数。FFT和插值算法的实现细节不同，理论FLOPs也会不同；无权声称是独立于计数约定的唯一精确值。

`core_GFLOPs`使用PyTorch 2.6.0 FlopCounterMode内建卷积、mm/bmm/addmm等规则；`additional_GFLOPs`为剩余浮点操作及显式规则；`algorithmic_GFLOPs`为两者之和。details.json按算子分解；unaccounted_ops必须为空，否则脚本报错，不输出通过结果。

## Params

sum(p.numel() for p in model.parameters())，单位M=10^6。包括已注册但未在当前前向使用的参数，不计buffer、优化器、激活。通过strict=True加载对应seed42的last.pth。SPGFNet前两阶段各有65563个可变形参数和1个门控未启用，合计131128；各数据集前向启用参数分别482908、697442、816103。SSR-NET有1个未使用的weight参数，仍计入注册总量。参数计数是整数精确值。

## 算术与归一化

- 卷积/GEMM：PyTorch默认2 FLOPs/MAC，bias或addmm加项另加输出元素数；卷积按含padding的完整核计数。
- 实数逐元素add/sub/mul/div：每输出元素1。复数加减2；复数×复数6；复数×实数2。若出现复数除法按11。
- scatter_add：每个源元素1次加法。
- 每组k个元素：sum=k-1；mean=k；std=4k+1（显式均值、去中心、平方、求和、方差除法及开方）。
- LayerNorm/GroupNorm：每组5k+2，再加每元素weight乘法和bias加法（各存在时1）。定义为mean=k；去中心=k；平方=k；方差求和=k-1、除法=1；eps加法=1；rsqrt=1；归一化乘法=k。使用这一表达式，不模拟内核Welford指令路径。
- softmax：每组5k-2，含k-1个最大值比较、k次减法、k次exp、k-1次求和、k次除法。
- sigmoid：4/元素；GELU（erf形式）：5/元素；log/tanh等特殊函数1/元素。
- 自适应平均池化：按每个输出窗口实际覆盖输入元素数计“求和+除法”，窗口允许重叠。

## 注意力覆盖与防止漏计

关闭torch.backends.mha快速融合路径，仅用于把MultiheadAttention投影分解为可统计算子，数学结构不变。verify_complete.py检查优化路径与计数路径输出接近。

CPU仍可使用融合SDPA：对Q[B,H,L,D]、K[B,H,S,D]、V[B,H,S,Dv]，计2BHLS(D+Dv) + BHLS（缩放）+ BHL(5S-2)（softmax）。QKV与输出投影由其各自linear/GEMM算子计数，不重复计入SDPA。

## FFT约定

复数FFT采用经典radix-2算法成本5*N*log2(N)；按批次/通道累计，2-D N=H*W。归一化另加每复数输出2次实数缩放。实数FFT预留半成本规则（本次主要使用c2c）。这是理论算法约定，不测量cuFFT/MKL实际指令，也不声称对非2幂尺寸是精确执行计数。本次H=W=64。

复数频域乘法另由mul计数，不包含在FFT公式里。固定核、OTF、采样索引的缓存构建不计入warm前向；先在no_grad下warm一次。

## 重采样约定

- 普通双线性插值：每输出通道位置4乘3加=7。
- 双三次插值：16点直接加权和，每输出通道位置16乘15加=31。
- 抗混叠双线性插值：按可分离水平、垂直滤波计数，过滤长度为实际边界内tap数t，各输出2t-1。水平中间尺寸为输入高×输出宽，随后垂直得到最终尺寸。tap边界按ATen half-pixel支持域计算；零权重tap也计入。
- grid_sample（bilinear/border/align_corners=False）：每空间位置14次坐标及权重计算，每通道7次插值算术；模型显式生成grid和采样后权重混合由其各自逐元素算子另计。

## 明确省略/计零项

以下省略不能隐去：普通interpolate/抗混叠interpolate内部坐标与插值系数生成（将系数视为确定几何的预处理），grid_sample的整数索引/floor/border比较、地址计算，warm缓存构建，Python标量/shape几何算术，随机数生成/随机置换/排序比较，显存/内存分配与搬运，拼接/切片/索引/reshape/transpose/unfold复制，纯dtype转换和lazy共轭视图。整型与布尔操作不计浮点操作。GPU/CPU缓存、FFT计划、算子调度不计。没有反向传播、损失函数或优化器步骤。

SSR-NET原始前向即使适配器只返回重建结果，仍会计算三路边缘差分；这是当前实际实现的一部分，计入本次总量，未擅自裁剪前向。

本次18组没有未分类算子；“无未分类”表示所有观测到算子均有计数或计零规则，不意味着上述省略项不存在。

## 完整性与复跑

profile_complete.py在运行前后对9个last.pth进行SHA256核验并断言不变，details.json记录对应路径与摘要。只读权重，报告写入独立complete_rules目录。verify_complete.py独立闭式核对SSR-NET三个通道配置、分组卷积+bias计数，并检查MHA不同执行路径的数值一致性。

命令（项目根目录）：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Experiments/Results/model_complexity_20261116/profile_complete.py
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Experiments/Results/model_complexity_20261116/verify_complete.py
```

推荐表注：复杂度在batch=1、HR=64×64、×4下统计，乘加计2 FLOPs；包含FFT、注意力、归一化和采样，FFT采用radix-2理论计数，重采样按本文加权求和约定计数，几何预处理与访存不计。不得与原论文不同/未知输入与工具口径的FLOPs直接混合比较。

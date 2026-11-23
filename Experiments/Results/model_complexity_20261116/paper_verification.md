# 原论文复杂度核验（2026-11-16）

作者仓库：https://github.com/weili419/SFNO
核验 commit：156fecb52c230e554d46e9c3be8533263fa9eb75。

原论文 Table 1：CAVE SFNO 0.32 M / 61.20 G；Chikusei 1.57 M / 468.62 G；Pavia Centre 1.34 M / 331.85 G。这些是论文引用值，不能替换当前复现模型的实测值。

作者 CAVE/Checkpoint/BestModel_ts.pt 实测注册参数 318245（0.318245 M，四舍五入0.32 M）。其 spe1、spa1、conv00、conv01、fc1、fc2、convw 合计73600参数，未在顶层前向调用。公开 CAVE/Model/SFNO.py 及当前CAVE复现模型参数均244645，差值正好73600。官方TorchScript的VectorAttention前向也与当前源码残差写法有差异，不能将两者认为完全相同的实现。

作者 Test.py 默认 batch=1，HR=512×512，sf=4，但这只是测试默认值，未发现它被明确用于论文复杂度统计的证据。仓库未提供全模型统计入口；仅Attention.flops(N)有局部公式，不足以重建整表。该局部公式采用乘加为1的矩阵乘法计数，但不能据此推断整表工具或口径。

当前模型THOP 64×64检查：1.219297280 G MACs，THOP参数计数0.243673 M（漏计部分自定义模块参数），故Params仍应直接用sum(p.numel())。按空间面积将MACs粗略换算到512得到78.03502592 G，不能复现论文61.20 G，不能作为精确512实测值。

verified_x4.csv 保留精确注册参数；计算量改列为PyTorch FlopCounterMode支持算子的GFLOPs（主要为卷积与矩阵乘，乘加=2），不再将手工FFT/插值估算混入标准FLOPs列。batch=1，HR=64×64，LR=16×16。该列仍不是完整FLOPs，不能与原论文直接比较。此前major_estimate列仅作为估算参考。

结论：CAVE参数量差异已解释；原论文完整FLOPs统计协议未公开确认，无法诚实给出“与论文完全相同口径”的SPGFNet/SSR-NET FLOPs。Pavia和Chikusei的论文参数量也不应替换当前实际权重对应的参数量。

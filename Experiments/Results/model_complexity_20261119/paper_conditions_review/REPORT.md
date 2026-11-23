# 论文复杂度条件复核与测试尺寸实测（2026-11-19）

**尚不能声称按论文Table1完全同口径复测。** 本次已检查论文全文、作者仓库全部20个提交及预训练模型；作者没有公开全模型统计脚本、Table1输入几何、计数工具/MAC口径及遥感模型配置，用户也没有额外作者说明。

已完成可核实的工作：作者CAVE参数可重现0.32M；8组CPU实际前向计数、无未分类算子；按作者默认CAVE测试几何和论文遥感测试图像尺寸得到如下结果。**这些测试图像尺寸不等于已确认的Table1 FLOPs输入。**

| 数据集 | 模型来源 | HR / LR（×4） | Params(M) | 核心FLOPs(G) | 约定下算法FLOPs(G) |
|---|---|---|---:|---:|---:|
| CAVE | SFNO 作者TorchScript | 512 / 128 | 0.318245 | 162.0721 | 171.9458 |
| CAVE | SFNO 本地复现 | 512 / 128 | 0.244645 | 158.8509 | 168.2480 |
| CAVE | V3-stage3 本地 | 512 / 128 | 0.614036 | 226.3755 | 250.9626 |
| Chikusei | SFNO 本地复现 | 500 / 125 | 0.565327 | 695.1770 | 718.0825 |
| Chikusei | V3-stage3 本地 | 500 / 125 | 0.947231 | 779.0709 | 843.9585 |
| Pavia Centre | SFNO 本地复现 | 192 / 48 | 0.450020 | 73.0680 | 75.9143 |
| Pavia Centre | V3-stage3 本地 | 192 / 48 | 0.828570 | 84.7199 | 92.0475 |

HR/LR均为正方形，batch1；core采用乘加2，algorithmic添加归一化、激活、FFT、插值等明确规则。FFT及插值部分是算法估算约定。

## 与论文数值的边界

| 数据集 | 论文Params(M) | 论文FLOPs(G) | 本次确认程度 |
|---|---:|---:|---|
| CAVE | 0.32 | 61.20 | 作者模型注册参数0.318245M可重现；FLOPs完整统计口径无法确认 |
| Chikusei | 1.57 | 468.62 | 作者该数据集模型/权重未公开，Params/FLOPs均不能声称已对齐 |
| Pavia Centre | 1.34 | 331.85 | 作者该数据集模型/权重未公开，Params/FLOPs均不能声称已对齐 |

作者CAVE模型512输入下core162.072100864G，即只对卷积/矩阵乘换算为1 MAC也为81.036050432 GMACs，仍不是61.20。因此不能只把输入改512或只除2就宣称复现论文。其他算术操作也不应统一除2。

## 统一64输入的实际对照

| 模型 | Params(M) | 核心FLOPs(G) | 约定下算法FLOPs(G) |
|---|---:|---:|---:|
| SFNO 作者TorchScript | 0.318245 | 2.5324 | 2.6867 |
| SFNO 本地复现 | 0.244645 | 2.4820 | 2.6289 |
| V3-stage3 本地 | 0.614036 | 3.5449 | 3.8867 |

作者模型与当前源码存在PR窗口、排列和ADE残差差别，不能将它们视为完全同一实现。未调用参数只影响注册参数统计，不产生对应前向FLOPs。

## 条件、证据与限制

- **严格同口径结论**：无法确认论文Table1 FLOPs的完整统计条件。论文全文和作者全部20个公开提交均未找到全模型profile入口；用户确认没有额外作者说明。因此本次绝不标作已复现论文61.20/468.62/331.85G。
- **本次完成**：8组实际CPU前向计数：作者CAVE模型64/512；本地SFNO/V3分别CAVE512、Chikusei500、Pavia192；没有按面积外推。
- **作者CAVE来源**：官方BestModel_ts.pt，318245个注册参数；作者Test.py默认及导出trace示例均为batch1、HR512、LR128、31/3通道、×4。默认测试/trace输入不是Table1 FLOPs输入的证明。
- **其他两数据集**：图像尺寸取自论文§4.1测试设置；模型采用当前本地实验：Chikusei128/3，Pavia102/4。作者没有公开这两组网络/权重，不能确认与论文配置相同。
- **计数口径**：batch1、×4、乘加=2；core为PyTorch支持的卷积/矩阵乘，algorithmic为core加明确的其他算法规则。FFT/插值等是理论成本约定，不是实际机器指令。
- **FFT限制**：沿用5*N*log2(N)理论成本；192和500不是2幂，不能声称该公式精确等于FFT库内部执行计数。
- **大图内存优化**：本地模型仅对eval VectorAttention使用既有32行分块实现，保持数学计算，避免高波段大图激活峰值；本次统计包含实际执行的分块路径。作者TorchScript保持原实现。
- **计数器补充**：TorchScript使用aten._convolution，已补bias加法；equal布尔比较计0；floor_divide仅允许整数shape几何并计0；常数padding计0。8组未分类算子均为空。
- **模型区别**：作者TS PR窗口8/4/2且无随机排列；当前公开/本地SFNO PR窗口4/4/2且有随机排列；作者ADE没有当前源码额外的+x。
- **未启用参数**：作者TS注册但未在顶层前向调用的7个模块共73600参数，解释其0.318245M与源码0.244645M之差；不构成这些未调用模块的前向FLOPs。
- **论文对齐程度**：CAVE作者参数0.318245M可四舍五入得到论文0.32M；论文Chikusei1.57M/Pavia1.34M因作者配置未公开不能独立重现。作者512实测core162.0721G、algorithmic171.9458G均不等于61.20G。
- **工作区处理**：只读权重并核对前后SHA256；不修改模型/训练配置/已有性能报告，也不启动或暂停训练。此前HR64统一比较表保持不变。

## 来源

- [原论文](../../../../Li%20等%20-%202026%20-%20Arbitrary-scale%20spatial-spectral%20fusion%20using%20kernel%20integral%20and%20progressive%20resampling.pdf)：§4.1测试尺寸、Table1复杂度及Table5消融。
- [作者测试入口固定版本](https://github.com/weili419/SFNO/blob/156fecb52c230e554d46e9c3be8533263fa9eb75/CAVE/Test.py#L15)：默认512、batch1、×4；不是复杂度脚本。
- [官方历史核查证据](sfno_author_history_20261119_evidence.json)：20提交、91历史路径、23文本版本，未发现完整profile入口或遥感模型。
- [作者模型独立64核验](sfno_author_ts64_review.json)：含实际forward代码、PR窗口、参数及算子分项。
- [既有明确计数规则](../../model_complexity_20261116/complete_rules/RULES.md)；本次新增TorchScript packet处理见[profile_geometry.py](profile_geometry.py)。

## 文件

- [test_geometry_profiles.xlsx](test_geometry_profiles.xlsx)：三个工作表，测试尺寸实测、64控制组、条件与限制。
- [test_geometry_profiles.csv](test_geometry_profiles.csv)：本次实际大图数值。
- 各`*_hr*.json`：每次前向的形状、权重哈希、完整算子分项和未分类清单。
- [profile_geometry.py](profile_geometry.py)：可复跑实际前向；[build_report.py](build_report.py)：本报告与工作簿生成。

此前的HR64统一算法计数表仍可用于明确条件下的本地模型比较；本次实测不是对该表的覆盖，也不应替换论文引用值后声称全部为论文同口径。

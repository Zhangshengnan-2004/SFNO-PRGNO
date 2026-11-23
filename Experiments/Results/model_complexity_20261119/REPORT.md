# 比较表复杂度补齐与核对（2026-11-19）

主表采用当前实现统一测量的×4 FLOPs：batch=1、HR64×64、LR16×16、乘加=2。×8为HR64×64、LR8×8。

| Dataset | Methods | Params(M) | FLOPs ×4(G) | FLOPs ×8(G) |
|---|---|---:|---:|---:|
| CAVE | SSR-Net | 0.026041 | 0.2148 | 0.2148 |
| CAVE | MHF-Net | 1.211047 | 7.9952 | 8.0161 |
| CAVE | SFNO | 0.244645 | 2.6289 | 2.6289 |
| CAVE | V3-stage3 | 0.614036 | 3.8867 | 3.8573 |
| Chikusei | SSR-Net | 0.442753 | 3.6333 | 3.6333 |
| Chikusei | MHF-Net | 16.604171 | 98.6603 | 99.0185 |
| Chikusei | SFNO | 0.565327 | 11.7651 | 11.7651 |
| Chikusei | V3-stage3 | 0.947231 | 13.6519 | 13.6228 |
| Pavia Centre | SSR-Net | 0.281215 | 2.3087 | 2.3087 |
| Pavia Centre | MHF-Net | 10.791840 | 64.9083 | 65.1357 |
| Pavia Centre | SFNO | 0.450020 | 8.4349 | 8.4349 |
| Pavia Centre | V3-stage3 | 0.828570 | 10.1581 | 10.1290 |

## 统计说明与截图更正

- **统计日期**：2026-11-19
- **主表FLOPs**：按×4统计；batch=1，HR=64×64，LR=16×16；单位G=10^9；一次乘加=2 FLOPs。×8另列于复杂度明细。
- **通道配置**：CAVE HSI/MSI=31/3；Chikusei=128/3；Pavia Centre=102/4。
- **计数含义**：明确算法约定下的FLOPs：卷积、矩阵乘、注意力、归一化、激活、FFT、采样。FFT采用radix-2理论公式，特殊函数按单个算法操作计算，不是硬件指令精确数。
- **省略项**：整数索引/几何、访存、随机置换、固定几何与OTF warm缓存预处理、普通插值内部坐标和系数生成；详见原RULES.md。
- **原规则**：/home/zsn/projects/SFNO-PRGNO/Experiments/Results/model_complexity_20261116/complete_rules/RULES.md
- **MHF新增规则**：var_mean correction0每组k元素计4k；完整显式MomentNorm为8n+2C；常数padding计零；转置卷积按输入空间和完整实际计算计数，后续crop不扣减既有计算。
- **MHF模型**：当前comparison中PyTorch作者CMHF翻译，rank12/stages20，forward推理路径；×8共享权重适配。按结构统计，无需训练或读取checkpoint。
- **Chikusei MHF**：截至生成表格时未发现该组训练配置/测试报告。复杂度按128/3输入及相同ComparisonMHF结构得到，不代表该组已训练完成。
- **参数量**：精确注册参数总量，包括注册但未启用的参数；V3-stage3前两阶段131128个可变形/门控参数未参与前向。
- **截图更正1**：截图SSR-Net与V3-stage3的Params(M)放反；CAVE分别应为0.026041与0.614036。
- **截图更正2**：截图中标Chikusei的数据块对应Pavia；标Pavia Center的数据块对应Chikusei。新主表按真实数据集归位。
- **截图更正3**：Chikusei SFNO seed42 ×4 PSNR：原报告43.7923866839，四位小数43.7924；截图43.7942疑为录入错误，新表按报告填入。
- **性能来源**：主表性能从训练seed42权重对应的原测试JSON读取，test seed=1；CAVE为历史连续600轮协议，Pavia/Chikusei为当前comparison协议。缺失的MHF遥感性能保持空白。
- **论文SFNO值**：论文报告CAVE61.20G、Chikusei468.62G、Pavia331.85G；统计尺寸/工具/倍率/乘加口径未确认，且作者序列化模型和本地源码存在差异。仅供引用，不可与本表本地测量混用。
- **文件处理**：输入只有截图，生成新Excel/CSV，未覆盖用户原表、既有复杂度文件、训练代码、权重或测试报告。

## 文件

- [comparison_completed.xlsx](comparison_completed.xlsx)：三个工作表，包含与截图同结构的已核对比较表、双倍率复杂度和说明。
- [comparison_completed.csv](comparison_completed.csv)：含性能的主表；[complexity_all.csv](complexity_all.csv)：双倍率复杂度精确值。
- [profile_mhf.py](profile_mhf.py)、[mhf_details.json](mhf_details.json)：MHF测量与算子分项，所有观测算子均有计数或计零规则。
- [table_evidence.json](table_evidence.json)：原统计源文件与测试报告哈希；已有9个模型checkpoint哈希也已与原测量记录核对。
- [build_tables.py](build_tables.py)：可复核表格生成脚本。

主表FLOPs为明确规则下的算法计数，不能声称与未知统计协议的论文数值完全同口径。MHF只做CPU前向，未启动、暂停或修改任何训练。

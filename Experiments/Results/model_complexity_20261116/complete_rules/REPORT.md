# Params 与算法 FLOPs（补全规则后）

条件：batch=1、HR=64×64、FP32、乘加计2。参数为精确注册数量；FLOPs为RULES.md所列算法约定下的计数，FFT理论计数、插值系数预处理等限制见规则。不是原论文口径的复刻。

| 数据集 | 模型 | Params (M) | FLOPs ×4 (G) | FLOPs ×8 (G) |
|---|---|---:|---:|---:|
| cave | SFNO | 0.244645 | 2.6289 | 2.6289 |
| cave | SPGFNet (V3-stage3) | 0.614036 | 3.8867 | 3.8573 |
| cave | SSR-NET | 0.026041 | 0.2148 | 0.2148 |
| pavia | SFNO | 0.450020 | 8.4349 | 8.4349 |
| pavia | SPGFNet (V3-stage3) | 0.828570 | 10.1581 | 10.1290 |
| pavia | SSR-NET | 0.281215 | 2.3087 | 2.3087 |
| chikusei | SFNO | 0.565327 | 11.7651 | 11.7651 |
| chikusei | SPGFNet (V3-stage3) | 0.947231 | 13.6519 | 13.6228 |
| chikusei | SSR-NET | 0.442753 | 3.6333 | 3.6333 |

已覆盖主要算子：卷积、矩阵乘、bias、MHA/SDPA、逐元素算术、归一化、激活、softmax、池化、FFT、插值、grid_sample、scatter_add与边缘差分。18组无未分类算子，独立闭式核对及数值一致性检查通过；9个checkpoint前后SHA256不变。

计零或省略：访存/索引/整数几何，随机置换/排序，固定几何/OTF warm缓存预处理，普通插值内部坐标与系数生成。特殊函数按每次1个算法操作；并非实际机器指令计数。详见RULES.md。

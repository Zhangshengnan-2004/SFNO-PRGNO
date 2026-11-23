CAVE 数据与共享训练代码

截图对应的连续600轮SFNO / V3-stage3（SPGFNet）以及seed42/43/44的六组结果，已统一移动至[Experiments/cave_continuous600](../Experiments/cave_continuous600/README.md)。截图对应seed42，三种子统计见新目录Summary/mean_std.md。

原始数据Data、共享Train.py / Test.py / Train_V3.py / Test_V3.py、Model和physics中的拟合响应保留在此；新实验目录另有迁移时的代码快照和统一入口。原CAVE三种子队列启动脚本已随历史运行记录移动至新目录History/queue_scripts，仅作历史记录。

Checkpoint/v3_balanced_scales内现存Stage1/2、Checkpoint/ablation_v3及其余SFNO/V1历史权重不属于本次迁移。旧V2/完整V3实验和200/400轮测试输出已按先前要求清理，见../Experiments/cleanup_records_20261113/。

新目录中的原始metrics.json、权重和日志保留原文与校验值，因此其中checkpoint/config路径可能仍指向迁移前位置；新位置见各组location.json及根目录path_map.json。旧文档中的路径只作历史说明。

实验目录索引

三数据集训练seed42/43/44的最终“均值 ± 样本标准差”见[随机种子汇总](Results/three_seed_summary_20261115/summary.md)，可直接打开[Excel表格](Results/three_seed_summary_20261115/summary.xlsx)，逐种子数值见[per_seed.csv](Results/three_seed_summary_20261115/per_seed.csv)。共18组600轮实验、36份×4/×8测试报告。

| 数据集与实验 | 目录 | 主要内容 |
|---|---|---|
| CAVE，连续600轮SFNO与V3-stage3 | [cave_continuous600](cave_continuous600/README.md) | seed42/43/44六组已完成实验、×4/×8测试、三种子汇总及代码快照 |
| Pavia / Chikusei，SFNO | [sfno_restart_20261111](sfno_restart_20261111/README.md) | 当前修正协议的SFNO实验 |
| Pavia / Chikusei，V3-stage3 | [v3_stage3_restart_20261111](v3_stage3_restart_20261111/README.md) | 与上述SFNO匹配的V3-stage3实验 |
| Pavia / Chikusei联合队列 | [sfno_v3_queue_20261111](sfno_v3_queue_20261111/README.md) | 调度状态、逐种子指标和联合汇总 |
| 历史清理记录 | [cleanup_records_20261113](cleanup_records_20261113/README.md) | 删除清单和诊断记录；不是新的训练实验 |

Pavia/Chikusei活动任务见[active_sfno.json](active_sfno.json)。各实验的Checkpoint保存权重、训练曲线和测试报告，Runs保存运行日志。CAVE新整理目录的具体布局见其README。

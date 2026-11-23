# 实验结果位置与旧版清理说明

2026-11-13：按用户最新要求，已删除此前指定保留的四组旧版实验：Pavia SFNO、Pavia V3-stage3、Chikusei protocol_v3_joint_loss SFNO及V3-stage3（均为seed42）。对应权重、训练日志、×4/×8测试报告和旧指标CSV已删除。先前的保留决定由本次要求替代。

旧实验的数据与评价协议未完全对齐论文，不能作为严格复现结果。原始数据保留；Chikusei的protocol_v2和protocol_v3_joint_loss历史协议目录亦已按用户要求整体删除。其余未指定的历史代码保留。

当前实验保存在以下位置：

- [SFNO](Experiments/sfno_restart_20261111/README.md)：两个数据集、seed42/43/44，权重与测试报告位于其Checkpoint目录。
- [V3-stage3](Experiments/v3_stage3_restart_20261111/README.md)：对应的V3-stage3实验，权重与测试报告位于其Checkpoint目录。
- [联合队列](Experiments/sfno_v3_queue_20261111/README.md)：运行状态见status.json，已完成结果见summary.md和per_seed.csv。
- 当前活动状态：[active_sfno.json](Experiments/active_sfno.json)。

当前实验仍有已明确记录的作者协议未知项，不宣称严格复现论文。CAVE、当前实验和原始数据不在本次删除范围内。

删除文件清单与校验记录见[清理记录](Experiments/cleanup_records_20261113/historical_experiments_cleanup.json)。更早清理清单中的“保留”仅表示当时状态。

CAVE连续600轮的SFNO/V3-stage3三种子实验已于2026-11-13迁移至[Experiments/cave_continuous600](Experiments/cave_continuous600/README.md)，包含权重、训练日志、×4/×8报告和汇总。

> 迁移说明：这是原始历史队列说明。结果已按模型/种子整理到../../Checkpoint和../../Runs，汇总在../../Summary；原命令仅作历史记录，新入口见../../README.md。

# 连续训练 baseline

原始 SFNO（Train.py --model sfno），从头连续训练600轮；seed=42、训练×4、batch=1、patch=64、每轮1000 patches、workers=2、Adam、L1、AMP关闭。与 v3_balanced_scales 内完整V3及Stage3相同数据划分、训练预算、绝对epoch学习率和测试协议。

学习率：1–200轮1e-4；201–300轮5e-5；301–350轮2.5e-5；351–500轮1.25e-5；501–550轮6.25e-6；551–600轮3.125e-6。不在200/400轮重启。

等待 P2Fusion-main/train_after_v3_status.json 为 p2fusion_complete 后启动；若上游失败则报错，不并行抢占GPU。后台入口：CAVE/run_continuous_baseline.py。

输出：CAVE/Checkpoint/v3_balanced_scales/baseline_seed42；使用last.pth完成×4/×8测试（seed=1，×8显式cross-scale）。随后更新同目录comparison.csv。原始分段训练baseline保持不变。

当前运行状态见status.json；训练、测试和队列日志保存在本目录。CPU已检查训练脚本语法及学习率边界与V3一致性。单种子为初步对照，未训练完成前不填入指标。

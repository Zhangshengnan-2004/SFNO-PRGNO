# 四模型统一续训：epoch200 → epoch400

SFNO、V1、V2、V3均从各自原目录last.pth（epoch200）恢复模型、Adam动量和步数。
只训练×4；每个模型最终总计400轮（新增200轮）。单GPU依次执行，每个模型完成后用last.pth测试×4和×8。

固定设置：batch1、HR patch64、每轮1000随机patch、seed42、Adam、无AMP；测试seed1、12场景。
学习率：201–300轮5e-5，301–350轮2.5e-5，351–400轮1.25e-5。
各模型使用相同绝对轮次学习率策略。优化器恢复后显式设置学习率，避免旧优化器覆盖新设置。
原断点没有保存完整随机数生成器状态，因此此次重设seed的续训不等价于原任务不间断地运行400轮。
未新增验证集；统一400轮是预先固定预算，不代表已证实所有模型充分收敛。

输出：
- SFNO: Checkpoint/train_x4_e400/
- V1: Checkpoint/train_x4_prgno_e400/
- V2: Checkpoint/train_x4_prgno_v2_e400/
- V3: Checkpoint/train_x4_prgno_v3_e400/

新目录train_log.csv仅记录201–400轮。训练实时日志位于本目录的sfno_train.log、v1_train.log、v2_train.log、v3_train.log。
完成训练后拷贝train.log到对应新目录，再生成test.log、metrics.json（×4）、test_x8.log、metrics_x8.json（×8）。
旧200轮目录不写入，启动时original_sha256.json记录所有旧实验文件校验和；队列结束后integrity.json检查是否改变。
status.json记录各模型queued/training/testing/complete/failed状态；某一项失败后继续执行其他模型，失败项不被误报为完成。
执行器run_continue_400.py拒绝复用现有运行目录与新实验目录，避免覆盖。

> 迁移说明：这是原始历史队列说明。结果已按模型/种子整理到../../Checkpoint和../../Runs，汇总在../../Summary；原命令仅作历史记录，新入口见../../README.md。

# CAVE baseline / V3-stage3 三种子比较

范围：表中的连续600轮SFNO baseline与V3-stage3；训练seed=42,43,44。保留现有seed42，新增seed43/44共4次从头训练。顺序：baseline43、stage3_43、baseline44、stage3_44。等待当前Pavia完整V3训练和x4/x8测试成功结束、GPU无其他计算进程后启动。

协议：×4训练600轮、不在200/400轮重启；batch1，patch64，每轮1000patches，workers2，Adam+L1，AMP关闭。初始lr1e-4，完成200/300/350/500/550轮后各乘0.5，与seed42一致。最终last.pth测试×4和×8（cross-scale），测试seed固定1，12个场景。没有加载seed42训练权重。

输出：Checkpoint/v3_balanced_scales/{baseline,deform_stage3}_seed{43,44}。训练测试日志在本目录，每完成一组后更新per_seed.csv及summary.json。全部12份测试报告齐备后生成mean_std.csv、mean_std.md及配对差值统计。

统计方法：每个训练种子的指标先对12个测试场景求平均，再对3个训练种子取算术平均及样本标准差s=sqrt(sum((x-mean)^2)/(3-1))，即ddof=1。不是场景间标准差，也不是标准误。另报告同训练seed的Stage3减baseline差值的均值与样本标准差。

缺失结果不填0、不计算所谓三种子结果。现有seed42权重和报告保存SHA256，完成后再次校验。当前只有1种子，没有最终均值±标准差。

后台入口：CAVE/run_three_seed_comparison.py。状态：status.json；总日志：execution.log。

手动重新汇总：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python run_three_seed_comparison.py --summarize
```

已验证：seed42训练/测试协议；三种子样本标准差、配对差值及缺失结果处理（临时合成数据，与真实实验目录隔离）。此队列拒绝重复启动或覆盖目标目录；如中断请先检查状态与checkpoint再决定续训，不能直接重复执行。


2026-11-06 队列调整：按用户要求，当前 CAVE Stage3 seed43 完成训练和测试后，先运行 Pavia V3-stage3 seed42，再运行 CAVE baseline seed44、Stage3 seed44。当前训练子进程未中断或重启。新调度器为 CAVE/run_reordered_comparison.py，总日志为 CAVE/reproduction/baseline_stage3_three_seeds/reordered_execution.log；原有各任务 status.json 持续更新。旧队列入口不要重复启动。


测试日志约定（2026-11-06）：自动训练队列完成训练后执行×4/×8测试，测试入口同时在权重目录保存test.log和test_x8.log，包含checkpoint路径、轮数、逐图/块指标及最终均值。保留已有日志，避免截断正在重定向的stdout。历史缺失日志依据已保存JSON报告补齐并标明来源，不重新推理；原Runs/reproduction日志仍保留。

# SFNO + V3-stage3 联合队列

按用户要求，在当前Chikusei SFNO seed42训练及×4/×8测试结束后，优先运行Pavia和Chikusei的V3-stage3 seed42，再继续seed43/44。总计两个数据集×两个模型×三个种子=12组，每组600轮后测试×4及×8，共24份报告。

| 次序 | 模型 | 数据集 | seed |
|---:|---|---|---:|
| 1 | SFNO | Pavia | 42（已完成并保留） |
| 2 | SFNO | Chikusei | 42（接管现有训练子进程，继续运行） |
| 3 | V3-stage3 | Pavia | 42 |
| 4 | V3-stage3 | Chikusei | 42 |
| 5 | SFNO | Pavia | 43 |
| 6 | SFNO | Chikusei | 43 |
| 7 | V3-stage3 | Pavia | 43 |
| 8 | V3-stage3 | Chikusei | 43 |
| 9 | SFNO | Pavia | 44 |
| 10 | SFNO | Chikusei | 44 |
| 11 | V3-stage3 | Pavia | 44 |
| 12 | V3-stage3 | Chikusei | 44 |

两模型使用相同数据协议、SRF、训练预算、损失、Adam/学习率和评价流程；V3单独记录CUDA可变形采样反向的数值非确定性。完整V3 GPU技术验收排在当前SFNO seed42结束后、首个V3训练前，失败则保留日志并停止联合队列。

外层调度不修改SFNO冻结源码及协议。交接只停止旧的任务派发父进程，不给正在训练的GPU子进程发送信号，也不重启它。handoff.json记录原派发PID、子进程PID/启动时间和原状态；外层队列持有两套原派发锁，防止重复启动独立队列。

- status.json：当前任务、阶段、子进程PID和日志。
- queue_plan.json：冻结的12组顺序、两套协议SHA256及本调度脚本SHA256。
- execution_*.log：联合调度日志；handoff.json：接管记录。
- per_seed.csv、summary.md/json：按模型/数据集/尺度分组汇总；三种子齐全后才计算样本标准差ddof=1。
- 实际SFNO输出仍在../sfno_restart_20261111/Checkpoint/。
- 实际V3输出在../v3_stage3_restart_20261111/Checkpoint/。

中断后确认原联合调度已退出，使用以下命令恢复，不再启动原SFNO独立派发器：

```bash
/home/zsn/miniconda3/envs/sfno/bin/python -u SFNO-PRGNO/Experiments/sfno_v3_queue_20261111/queue.py --resume --detach
```

恢复先等已存在的训练/测试子进程结束，重新读取最新完整检查点，已完成的最终测试跳过。stop_reason.json表示曾触发训练质量停止，须检查后处理，不自动绕过。

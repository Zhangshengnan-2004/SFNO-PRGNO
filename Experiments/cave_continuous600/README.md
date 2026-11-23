CAVE连续600轮：SFNO / V3-stage3（图中SPGFNet）

截图中的两行均对应训练seed42，SPGFNet行与原`deform_stage3`结果一致。这里同时收齐训练seed42、43、44共六组实验。测试随机种子均为1；×4训练，完成600轮后测试×4及跨尺度×8。截图不是三种子均值。

| 模型（seed42） | ×4 PSNR | SAM | ERGAS | ×8 PSNR | SAM | ERGAS |
|---|---:|---:|---:|---:|---:|---:|
| SFNO | 51.8970 | 2.0270 | 0.7789 | 49.5333 | 2.5104 | 0.5371 |
| V3-stage3 / SPGFNet | 52.0181 | 1.9487 | 0.7661 | 49.7121 | 2.4015 | 0.5298 |

目录按用途和模型／种子整理：

```text
cave_continuous600/
  Checkpoint/
    cave_sfno_x4_seed42/          # 另有seed43、44
    cave_v3_stage3_x4_seed42/     # 另有seed43、44
  Runs/
    cave_sfno_x4_seed42/          # train.log、test_x4.log、test_x8.log
    cave_v3_stage3_x4_seed42/     # 六组均齐全
  Summary/                      # per_seed.csv、mean_std.md/.csv、summary.json
  Analysis/stage3_scene_efficiency/
  Source/                       # 训练、测试和模型代码的迁移时快照
  History/                      # 原队列日志、状态及历史启动脚本
  run.py                        # 新统一入口
  status.json                   # 六组均已完成
  path_map.json                 # 原位置→新位置
  migration_manifest.json       # 迁移前后SHA256及修改说明
```

每个`Checkpoint`实验组中，`last.pth`是600轮最终权重，`train_log.csv`为连续600轮训练曲线；`test.log`、`test_x8.log`是×4／×8结果，`metrics.json`、`metrics_x8.json`包含12个测试场景的逐场景指标。`config.json`从原最终权重提取，`location.json`说明当前路径和训练／测试种子。

完整训练stdout日志在对应的`Runs`目录。三种子均值和样本标准差（ddof=1）见[Summary/mean_std.md](Summary/mean_std.md)，逐种子数值见[Summary/per_seed.csv](Summary/per_seed.csv)。其中baseline表示SFNO，deform_stage3表示V3-stage3。

共同训练配置：从头连续600轮、不在200／400轮重启；每轮1000块、batch1、patch64，Adam+L1，FP32，初始学习率1e-4，完成200／300／350／500／550轮后减半。各组原配置保留在权重内部及提取的config.json中。原始数据仍在`../../CAVE/Data`，未复制或改写。

从项目工作区`/home/zsn/projects`使用：

```bash
# 校验迁移文件、六组训练记录和12份测试报告；严格加载权重并在CPU做×4/×8小尺寸前向
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Experiments/cave_continuous600/run.py verify --load-weights

# 查看三种子均值和标准差，不覆盖原汇总
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Experiments/cave_continuous600/run.py summary

# 打印重新测试命令；去掉--dry-run才执行，输出必须是新文件
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Experiments/cave_continuous600/run.py test --model v3_stage3 --seed 42 --sf 8 --output /tmp/cave_stage3_seed42_x8_new.json --dry-run

# 打印同配置从头训练命令；输出必须是新目录
/home/zsn/miniconda3/envs/sfno/bin/python SFNO-PRGNO/Experiments/cave_continuous600/run.py train --model sfno --seed 42 --output /tmp/cave_sfno_seed42_new --dry-run
```

`--seed`标识训练种子，重新测试时统一使用原测试seed1。重新训练、测试不会自动覆盖迁移后的历史输出。`History/queue_scripts`仅保存原调度程序，不是新目录下的执行入口。

迁移保留了原权重、指标、训练CSV和日志的字节内容，文件内部可能仍记载原绝对路径，以`path_map.json`和各组`location.json`为当前位置依据。Source是迁移时快照，不声称它是训练启动时冻结的源代码。分析脚本仅调整了路径；完整V3对照权重此前已按要求删除，效率脚本重跑时会跳过该对照，已有历史效率报告保留。

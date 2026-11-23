# V1 消融实验

实验顺序：baseline（原 SFNO）、spatial（仅空间残差）、spectral（仅光谱残差）、dual（V1 双残差）、capacity（无物理残差的同参数量对照）。四个扩展模型均为 273668 参数。capacity 使用共享中间估计和 lift 生成调制，不调用空间/光谱退化算子；它用于区分增加容量与物理条件的贡献。

所有实验从头训练：200 epoch，64×64 patch，batch=1，Adam，学习率 1e-4，每 epoch 1000 patch，关闭 AMP。默认种子为 42、43、44。输出保存在 Checkpoint/ablation，原有权重不会被覆盖。执行器拒绝写入已有的非空实验目录；中断后的训练可通过 Train.py --resume 单独续训。

先补齐种子 42 的缺失对照：

```bash
/home/zwu/miniconda3/envs/sfno/bin/python run_ablation.py --variants spatial spectral capacity --seeds 42 --execute
```

完整重复实验：

```bash
/home/zwu/miniconda3/envs/sfno/bin/python run_ablation.py --execute
```

不加 --execute 仅生成计划。每项训练结束后自动运行测试，保存 train.log、test.log 和逐图 metrics.json。测试固定推理种子为 1；随机重排引起的波动需要另用多个 Test.py --seed 独立报告。

目前仍使用 20 个训练场景、最后一轮 checkpoint 做固定预算对比，不根据测试集选模型。尚未实现训练验证集划分和验证集选优；响应拟合脚本的验证场景不等于模型训练的独立验证集。未来加入验证集时，应统一重训全部对照，响应矩阵也只用该实验的训练场景拟合。

当前环境 CUDA 不可用，正式消融尚未启动；physics/ablation_smoke_report.json 仅证明前向/反向传播和参数量检查通过，不是重建性能结果。原 V1 验证结果见 physics/prgno_test_report.json。

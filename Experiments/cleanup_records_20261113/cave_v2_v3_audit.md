> 2026-11-13更新：清单中的candidate项已按用户要求删除；optional项及V3-stage3依赖继续保留。删除明细见cave_old_experiments_cleanup.json。以下为清理前核查记录。

CAVE中V2、完整V3与V3-stage3的文件核查（2026-11-13）

只考虑今后主要保留SFNO baseline和V3-stage3；未被当前任务调用不等于没有历史或消融证据价值。

本次只检查和列清单，没有删除或改写CAVE文件。当前运行进程为Experiments中的联合队列和Pavia V3-stage3 seed43，无CAVE V2/完整V3训练进程。

旧V2/完整V3候选合计1116.83 MiB；额外可选消融合计435.97 MiB。

**旧V2及完整V3的独立文件/目录**

| 相对CAVE路径 | 占用MiB | 说明 |
|---|---:|---|
| `Checkpoint/train_x4_prgno_v2` | 129.88 | 旧版独立200/400/600轮实验；不含deform_stage3。 |
| `Checkpoint/train_x4_prgno_v2_e400` | 115.14 | 旧版独立200/400/600轮实验；不含deform_stage3。 |
| `Checkpoint/train_x4_prgno_v2_e600` | 130.05 | 旧版独立200/400/600轮实验；不含deform_stage3。 |
| `Checkpoint/train_x4_prgno_v3` | 182.33 | 旧版独立200/400/600轮实验；不含deform_stage3。 |
| `Checkpoint/train_x4_prgno_v3_e400` | 182.52 | 旧版独立200/400/600轮实验；不含deform_stage3。 |
| `Checkpoint/train_x4_prgno_v3_e600` | 182.52 | 旧版独立200/400/600轮实验；不含deform_stage3。 |
| `Train_V2.py` | 0.01 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `Test_V2.py` | 0.01 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `test_prgno_v2.py` | 0.00 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `diagnose_v2_cross_scale.py` | 0.01 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `probe_v2_branches.py` | 0.01 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `probe_v2_followup.py` | 0.01 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `probe_v2_local_footprint.py` | 0.01 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `probe_v2_spatial_guidance.py` | 0.01 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `scripts/run_v2_x4.sh` | 0.00 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `V2.md` | 0.00 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `v2_training_execution.log` | 0.00 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `test_v2_x4_to_x8_seed42.log` | 0.00 | 只服务旧V2训练、测试或诊断；删除后不能重跑该V2流程。 |
| `physics/v2_branch_sensitivity` | 0.11 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_branch_sensitivity.log` | 0.00 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_diagnostics` | 0.14 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_diagnostics.log` | 0.00 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_followup_sensitivity` | 0.05 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_followup_sensitivity.log` | 0.00 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_gpu_smoke.json` | 0.00 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_local_footprint_sensitivity` | 0.04 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_local_footprint_sensitivity.log` | 0.00 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_spatial_sensitivity` | 0.09 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_spatial_sensitivity.log` | 0.00 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `physics/v2_x4_to_x8_seed42.json` | 0.00 | V2专用报告/敏感性结果；其中stage3字样指V2第三阶段，非V3-stage3实验。 |
| `reproduction/continue_200_to_400/v2_train.log` | 14.94 | 混合续训目录内的V2专用部分；不包括SFNO/V1。 |
| `reproduction/continue_200_to_400/v2_resume_from_346.log` | 5.52 | 混合续训目录内的V2专用部分；不包括SFNO/V1。 |
| `reproduction/continue_400_to_600/v2_train.log` | 20.42 | 混合续训目录内的V2专用部分；不包括SFNO/V1。 |
| `reproduction/continue_200_to_400_startup_failed/v2_train.log` | 0.09 | 混合续训目录内的V2专用部分；不包括SFNO/V1。 |
| `reproduction/continue_200_to_400_startup_failed/train_x4_prgno_v2_e400` | 0.00 | 混合续训目录内的V2专用部分；不包括SFNO/V1。 |
| `Checkpoint/v3_balanced_scales/full_seed42` | 21.89 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `Checkpoint/v3_balanced_scales/full_seed42_train_continuous.log` | 65.52 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `Checkpoint/v3_balanced_scales/full_seed42_test.log` | 0.00 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `Checkpoint/v3_balanced_scales/full_seed42_test_x8.log` | 0.00 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `scripts/run_v3_x4.sh` | 0.00 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `V3.md` | 0.00 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `v3_training_execution.log` | 21.75 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `reproduction/continue_200_to_400/v3_resume_from_200.log` | 21.88 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |
| `reproduction/continue_400_to_600/v3_train.log` | 21.88 | 完整V3，非V3-stage3；full_seed42仍供逐场景效率benchmark.py的完整V3对照读取。 |

**需要单独判断的消融，不能仅因名称含V3认定无用**

| 相对CAVE路径 | 占用MiB | 说明 |
|---|---:|---|
| `Checkpoint/ablation_v3` | 272.54 | no_deformable/no_regional/fixed_gate等旧消融；可能仍有结构对照价值，不认定已无科研用途。 |
| `v3_ablation_three_execution.log` | 0.00 | no_deformable/no_regional/fixed_gate等旧消融；可能仍有结构对照价值，不认定已无科研用途。 |
| `v3_ablation_three_continuous_execution.log` | 0.00 | no_deformable/no_regional/fixed_gate等旧消融；可能仍有结构对照价值，不认定已无科研用途。 |
| `physics/benchmark_v3_efficiency.py` | 0.00 | no_deformable/no_regional/fixed_gate等旧消融；可能仍有结构对照价值，不认定已无科研用途。 |
| `physics/v3_efficiency_benchmark.json` | 0.00 | no_deformable/no_regional/fixed_gate等旧消融；可能仍有结构对照价值，不认定已无科研用途。 |
| `Checkpoint/v3_balanced_scales/deform_stage1_seed42` | 18.80 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |
| `Checkpoint/v3_balanced_scales/deform_stage1_seed42_train_continuous.log` | 62.91 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |
| `Checkpoint/v3_balanced_scales/deform_stage1_seed42_test.log` | 0.00 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |
| `Checkpoint/v3_balanced_scales/deform_stage1_seed42_test_x8.log` | 0.00 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |
| `Checkpoint/v3_balanced_scales/deform_stage2_seed42` | 18.80 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |
| `Checkpoint/v3_balanced_scales/deform_stage2_seed42_train_continuous.log` | 62.91 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |
| `Checkpoint/v3_balanced_scales/deform_stage2_seed42_test.log` | 0.00 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |
| `Checkpoint/v3_balanced_scales/deform_stage2_seed42_test_x8.log` | 0.00 | 不是完整V3，也不是V3-stage3；为可变形分支位置消融证据，应单独决定保留。 |

**V3-stage3共享模型依赖，必须保留**

- `Model/DynamicSpatial.py`
- `Model/FusionBackbone.py`
- `Model/MultiScaleDynamicGating.py`
- `Model/MultiScaleDynamicGatingV2.py`
- `Model/ObservationGrid.py`
- `Model/PhysicsGalerkin.py`
- `Model/SFNO_PRGNO_V2.py`
- `Model/SFNO_PRGNO_V3.py`
- `Model/SpacingDeformableSampling.py`
- `Model/VectorAttention.py`

**保留的Stage3、baseline和混合用途文件**

- `Train_V3.py`
- `Test_V3.py`
- `test_prgno_v3.py`
- `test_v3_ablation.py`
- `run_v3_ablation.py`
- `run_three_seed_comparison.py`
- `run_reordered_comparison.py`
- `run_continuous_baseline.py`
- `run_continue_400.py`
- `run_continue_600.py`
- `recover_continue_400.py`
- `continuation_schedule.py`
- `physics/cave_effective_response.npz`
- `physics/cave_effective_response_report.json`
- `physics/v3_balanced_scales_plan.md`
- `physics/v3_ablation_plan.md`
- `v3_balanced_scales_execution.log`
- `Checkpoint/v3_balanced_scales/comparison.csv`
- `reproduction/baseline_stage3_three_seeds`
- `analysis/stage3_scene_efficiency`
- `Checkpoint/v3_balanced_scales/baseline_seed42`
- `Checkpoint/v3_balanced_scales/baseline_seed43`
- `Checkpoint/v3_balanced_scales/baseline_seed44`
- `Checkpoint/v3_balanced_scales/deform_stage3_seed42`
- `Checkpoint/v3_balanced_scales/deform_stage3_seed43`
- `Checkpoint/v3_balanced_scales/deform_stage3_seed44`
- `Checkpoint/v3_balanced_scales/deform_stage3_seed42_train_continuous.log`
- `Checkpoint/v3_balanced_scales/deform_stage3_seed42_test.log`
- `Checkpoint/v3_balanced_scales/deform_stage3_seed42_test_x8.log`

**引用关系及限制**

- 完整V3的full_seed42仍是analysis/stage3_scene_efficiency/benchmark.py读取的一个对照，删除后原三模型benchmark不能原样重跑；Stage3权重和已保存分析仍应保留。
- V3-stage3的Train_V3/Test_V3/model类与完整V3共用；Model/SFNO_PRGNO_V2.py与MultiScaleDynamicGatingV2.py属于实际继承/导入链。
- 续训目录里的comparison.csv、status.json、original_sha256.json、integrity.json、README和总日志同时记录SFNO/V1/V2/V3，不能把整个目录当作V2/V3专用。
- 历史/文档中绝对路径仅作证据，不是当前队列外部代码依赖。

精确逐文件列表见同目录cave_v2_v3_audit.json。该JSON仅为核查清单，不是删除脚本。

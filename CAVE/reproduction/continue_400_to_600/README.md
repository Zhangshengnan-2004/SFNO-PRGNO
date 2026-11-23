# 四模型400→600轮队列

先等待400轮恢复服务结束，再核查四模型epoch400权重及每个模型×4、×8完整12场景报告；全部成功后才启动600轮阶段。
从各自_e400/last.pth恢复模型和Adam状态，每个再训练200轮，总轮次600。仍然只训练×4，统一seed42、batch1、patch64、每轮1000patch、无AMP。
学习率承接400轮末值：401–500为1.25e-5，501–550为6.25e-6，551–600为3.125e-6。
每个模型训练完成后以last.pth测试×4和跨尺度×8（测试seed1、12场景）。

新目录分别为Checkpoint/train_x4_e600、train_x4_prgno_e600、train_x4_prgno_v2_e600、train_x4_prgno_v3_e600。
各目录保存last.pth、定期权重、train_log.csv（401–600）、train.log、test.log、metrics.json、test_x8.log、metrics_x8.json。
本目录status.json记录等待/运行/完成/失败状态，各模型_train.log为实时日志；完成后comparison.csv汇总400/600轮结果。
200/400轮旧实验均不覆盖，并校验SHA256。原断点未保存完整随机状态，续训不等价于不间断训练。
等待使用独立systemd用户服务，不依赖远程终端；服务器关机或异常退出仍需恢复。当前未配置自动重启。

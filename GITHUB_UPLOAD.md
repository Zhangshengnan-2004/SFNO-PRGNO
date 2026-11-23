# 上传不含权重的源码版本

本地训练文件继续保留。`.gitignore` 排除权重、原始 `.mat/.bsq` 等数据、训练日志、Python 缓存；模型配置、CSV/JSON 指标汇总、训练划分清单与小型光谱响应文件继续保留。

## 为什么不能只增加忽略规则

当前原仓库的首次提交已经包含权重和原始数据。`.gitignore` 只控制未跟踪文件；新增一次“删除权重”提交，也不会清除更早提交中的大文件。

本项目提供独立导出脚本：按当前工作区生成没有旧 `.git` 历史的发布副本。它不会删除本地权重、改写原提交或推送远程。

```bash
cd /home/zsn/projects/SFNO-PRGNO
python scripts/export_source.py --output /home/zsn/projects/SFNO-PRGNO-github
```

目标目录必须为空或不存在；重复导出时使用新的目录名。副本中的 `SOURCE_EXPORT.json` 记录包含与排除的文件。

## 从干净副本上传

确认目标 GitHub 仓库为空后，在发布副本中执行：

```bash
cd /home/zsn/projects/SFNO-PRGNO-github
git init -b main
git add .
git commit -m "Publish source without weights and datasets"
git remote add origin https://github.com/Zhangshengnan-2004/SFNO-PRGNO.git
git push -u origin main
```

若远程仓库已有提交，普通 push 可能因历史不同被拒绝。此时先克隆远程仓库到另一个新目录，把发布副本的内容复制进去，再正常提交和推送；不要直接强推覆盖已有历史。如果远程历史本身也含需移除的大文件，还需要另行清理远程历史。

不要在原目录直接执行上述初始化流程，也不要删除原目录的 `.git`。原目录继续作为本地训练工作区，干净副本用于此次源码发布。

GitHub 的普通 Git 仓库会阻止超过100 MiB的文件；具体历史清理规则见[官方大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

# Insert Hole 实验归档

本目录保存 50 条专家示范、ACT 4000-step 训练以及固定 100 Episode 测评的
可复现记录。

- `task_report.md`：完整实验报告与综合分析；
- `experiment_manifest.yaml`：版本、环境、配置、结果与文件哈希；
- `reproduce.md`：数据转换、训练、评测和统计命令；
- `raw_results/`：三组正式评测的 metadata、精简日志与场景系统描述；
- `figures/`：三组训练曲线；
- `*_comparison.md`：组间统计说明；
- `*_per_episode.csv`：逐 seed 配对数据；
- `generate_freeze_vs_finetune_stats.py`：Freeze/Finetune 统计生成脚本；
- `artifacts/dataset_stats.pkl`：三组共同使用的归一化统计。

大型 HDF5、模型 checkpoint 和完整视频没有进入 Git 历史。模型与视频通过
`insert-hole-act-ablation-2026-07-24` Release 保存，HDF5 数据需存放在独立
数据存储中。

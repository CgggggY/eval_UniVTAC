# UniVTAC Insert Hole 复现与视触觉消融实验

本 Fork 基于原始 [univtac/UniVTAC](https://github.com/univtac/UniVTAC) 仓库，
完成了 `insert_hole` 任务的数据转换、ACT 策略训练、固定 100 Episode 测评以及
Vision/触觉消融分析。原项目主体、任务定义和第三方依赖均保持不变；本文件主要
说明本 Fork 相比上游仓库增加或修改了什么，以及如何找到本次实验产物。

## 1. 我们做了什么

实验流程如下：

1. 在 FreeGPU 容器中配置 Isaac Sim、Isaac Lab、TacEx/UIPC 与 cuRobo 环境；
2. 使用 `insert_hole/demo` 的 50 条成功专家示范；
3. 将原始 HDF5 转换为 ACT 使用的状态—动作和视触觉图像格式；
4. 使用相同数据、训练 seed `0`、ACT 主干和 4000 个训练 step 训练三组策略；
5. 使用固定 seed `1000000–1000099`，分别完成 100 个闭环 Episode 测评；
6. 对逐 seed 成功关系、失败条件、插入深度、姿态、相对滑动和动作阶段进行统计；
7. 将代码改动、配置、结果、模型、视频与复现命令归档到 GitHub。

三组实验设置与结果：

| 组别 | 输入 | 触觉编码器 | 成功率 |
|---|---|---|---:|
| Vision | 第三视角相机 | 不使用 | 28/100 |
| Freeze | 视觉 + 左右触觉 | 加载预训练权重并冻结 | 28/100 |
| Finetune | 视觉 + 左右触觉 | 加载预训练权重并以 `1e-5` 微调 | 27/100 |

三组总体成功率接近，但成功 seed 并不完全相同。主要失败条件集中在插入深度不足
与物体相对夹爪滑动；详细原因、逐 seed 配对结果和进一步改进方向见
[完整实验报告](./analysis/task_report.md)。

## 2. 相比原始仓库修改了什么

### 2.1 ACT 部署兼容修复

修改文件：

```text
policy/ACT/deploy_policy.py
```

原实现会无条件读取旧命名的 `left_gsmini/right_gsmini` 触觉观测，导致
Vision-only 策略出现 `KeyError: left_gsmini`。本 Fork 的修改包括：

- Vision-only 模型不再读取不需要的触觉输入；
- 触觉模型支持当前的 `left_tactile/right_tactile` 名称；
- 同时保留对旧 `left_gsmini/right_gsmini` 名称的兼容。

### 2.2 训练配置补全

修改文件：

```text
policy/ACT/train_config.yml
policy/ACT/train_config_freeze.yml
policy/ACT/train_config_vision.yml
```

补充了当前 DETR/ACT 参数解析器要求的 `num_epochs` 和 `state_dim`，避免启动训练时
出现缺少 `--num_epochs` 或 `--state_dim` 的参数错误。此前出现的
`--policy_class` 缺失则来自未正确读取训练配置文件，而不是这两个字段本身。

### 2.3 新增实验配置

新增文件：

```text
policy/ACT/train_config_freeze_verified.yml
policy/ACT/deploy_vision.yml
policy/ACT/deploy_freeze.yml
policy/ACT/deploy_finetune.yml
policy/ACT/SIM_TASK_CONFIGS.json
```

这些文件分别固定了 Vision、Freeze、Finetune 的训练/部署名称，以及转换后的
50-Episode ACT 数据集目录、相机名称和轨迹长度设置。

## 3. 新增的实验归档

所有轻量实验记录保存在 [`analysis/`](./analysis/)：

| 路径 | 内容 |
|---|---|
| [`task_report.md`](./analysis/task_report.md) | 完整实验设置、结果、原因分析与改进建议 |
| [`experiment_manifest.yaml`](./analysis/experiment_manifest.yaml) | 代码版本、环境、配置、结果和模型 SHA-256 |
| [`reproduce.md`](./analysis/reproduce.md) | 数据转换、三组训练与固定评测命令 |
| [`raw_results/`](./analysis/raw_results/) | 三组正式测评的 metadata、精简日志和场景描述 |
| [`figures/`](./analysis/figures/) | 三组训练 loss、L1 和 KL 曲线 |
| `*_comparison.md` | Vision/Freeze 与 Freeze/Finetune 的统计比较 |
| `*_per_episode.csv` | 固定 100 seeds 的逐条配对结果 |
| [`generate_freeze_vs_finetune_stats.py`](./analysis/generate_freeze_vs_finetune_stats.py) | 可重新生成统计结果的脚本 |
| [`dataset_stats.pkl`](./analysis/artifacts/dataset_stats.pkl) | 三组共同使用的 ACT 归一化统计 |

## 4. 模型与视频下载

模型权重和完整评测视频体积较大，没有写入普通 Git 历史，而是保存在
[GitHub Release：insert-hole-act-ablation-2026-07-24](https://github.com/CgggggY/eval_UniVTAC/releases/tag/insert-hole-act-ablation-2026-07-24)。

Release 包含：

```text
vision_policy_last.ckpt
freeze_policy_last.ckpt
finetune_policy_last.ckpt
tactile_encoder_best.pth
final_eval_videos.tar.gz
SHA256SUMS
```

三份 `policy_last.ckpt` 是正式 100-Episode 测评实际加载的模型。视频压缩包包含
Vision、Freeze、Finetune 三个目录，共 300 条视频。

下载后可验证文件完整性：

```bash
sha256sum -c SHA256SUMS
```

## 5. 数据说明

本次实验涉及两类 HDF5：

```text
data/insert_hole/demo/hdf5/*.hdf5
policy/ACT/data/sim-insert_hole/demo-50/episode_*.hdf5
```

第一类是信息完整的原始专家示范；第二类是由
`policy/ACT/process_data.py` 转换得到的 ACT 训练数据。转换后每个时间步使用：

```text
qpos[t]   = 原始 embodiment/joint[t, 0:8]
action[t] = 原始 embodiment/joint[t+1, 0:8]
```

原始与转换数据合计约 15 GB，因此没有上传到普通 Git 仓库或本次 Release。
如需完全从头复现，应另行取得专家 HDF5 数据，并按照
[复现说明](./analysis/reproduce.md) 进行转换。

## 6. 快速查看顺序

首次查看本 Fork 时，建议依次阅读：

1. 本文件：了解相对上游的变化；
2. [`analysis/task_report.md`](./analysis/task_report.md)：了解实验结论；
3. [`analysis/experiment_manifest.yaml`](./analysis/experiment_manifest.yaml)：确认版本和哈希；
4. [`analysis/reproduce.md`](./analysis/reproduce.md)：重新运行实验；
5. GitHub Release：下载实际评测模型与视频。

## 7. 主要结论

当前结果没有观察到触觉输入带来稳定的总体成功率提升，也没有观察到微调触觉编码器
优于冻结编码器。不过，这不代表触觉完全无效：三组完成的 seed 集合存在明显差异，
触觉策略也改变了部分旋转条件下的行为表现。现阶段更可能的瓶颈包括：

- 50 条示范不足以覆盖接触后的纠偏行为；
- 专家数据中的夹爪动作几乎不变，缺少主动抗滑监督；
- ACT 长动作块与时间聚合可能削弱即时触觉反馈；
- 全局触觉特征可能丢失插入所需的局部接触和剪切信息；
- 单次训练 seed 和 100 个 Episode 仍不足以证明 1–5 个百分点的小差异。

后续可优先探索多训练 seed、触觉遮挡/噪声测试、接触阶段短 chunk、触觉时序特征、
主动夹持纠偏示范以及更大规模的数据实验。

## 8. 上游项目

本 Fork 的基础平台、论文、安装方法与任务定义来自 UniVTAC。通用平台说明仍以
[上游仓库](https://github.com/univtac/UniVTAC) 和本仓库原始
[`README.md`](./README.md) 为准。

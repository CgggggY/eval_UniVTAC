# Insert Hole ACT 实验复现说明

本说明对应 `experiment_manifest.yaml` 中记录的 Vision、Freeze 和 Finetune
三组实验。原始 HDF5 数据和模型权重不进入普通 Git 历史；原始数据保存在
[ModelScope 数据集](https://modelscope.cn/datasets/CgggggY/Univtac_insert_hole_50)，
权重与测评视频见同名 GitHub Release。

## 1. 环境

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/univtac/environments/UniVTAC
```

训练前：

```bash
export TMPDIR=/root/gpufree-data/univtac/tmp
export PIP_CACHE_DIR=/root/gpufree-data/univtac/cache/pip
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
```

评测还需要加载 Isaac Sim 环境：

```bash
source /root/isaacsim/setup_conda_env.sh
```

这些绝对路径是本次 FreeGPU 容器中的实际路径；在其他机器上应替换为对应安装位置。

## 2. 转换专家数据

下载 50 条原始专家示范：

```bash
modelscope download \
  --dataset CgggggY/Univtac_insert_hole_50 \
  --local_dir /tmp/Univtac_insert_hole_50
```

将下载目录中的 `raw/insert_hole/demo` 放入仓库的
`data/insert_hole/demo`。可以在源数据目录运行以下命令校验全部 114 个文件：

```bash
sha256sum -c SHA256SUMS
```

原始数据目录：

```text
data/insert_hole/demo/hdf5
```

转换为 ACT 格式：

```bash
cd policy/ACT
python process_data.py insert_hole demo 50
```

输出目录：

```text
policy/ACT/data/sim-insert_hole/demo-50
```

转换会把长度为 `T` 的原始轨迹整理成 `T-1` 个状态—动作训练对：

```text
qpos[t]   = embodiment/joint[t, 0:8]
action[t] = embodiment/joint[t+1, 0:8]
```

## 3. 训练

在 `policy/ACT` 目录中分别执行：

```bash
bash train.sh insert_hole demo 50 0 0 train_config_vision
bash train.sh insert_hole demo 50 0 0 train_config_freeze_verified
bash train.sh insert_hole demo 50 0 0 train_config
```

三条命令依次对应 Vision、Freeze、Finetune。正式测评加载每个输出目录中的
`policy_last.ckpt`。

## 4. 固定 100 Episode 测评

在仓库根目录执行。Vision：

```bash
export TRAIN_CONFIG=train_config_vision
export EP_NUM=50
export CUDA_VISIBLE_DEVICES=0
python scripts/eval_policy.py insert_hole demo ACT/deploy_vision --total_num 100
```

Freeze：

```bash
export TRAIN_CONFIG=train_config_freeze_verified
export EP_NUM=50
export CUDA_VISIBLE_DEVICES=0
python scripts/eval_policy.py insert_hole demo ACT/deploy_freeze --total_num 100
```

Finetune：

```bash
export TRAIN_CONFIG=train_config
export EP_NUM=50
export CUDA_VISIBLE_DEVICES=0
python scripts/eval_policy.py insert_hole demo ACT/deploy_finetune --total_num 100
```

本次固定种子范围为 `1000000–1000099`，结果依次为 `28/100`、`28/100`
和 `27/100`。

## 5. 重新生成统计

仓库已经包含三组正式评测的精简日志与 metadata：

```bash
python analysis/generate_freeze_vs_finetune_stats.py
```

脚本会读取 `analysis/raw_results`，不依赖被 `.gitignore` 排除的原始
`eval_result` 目录。

## 6. 校验 Release 文件

下载 Release 资产后，在其所在目录执行：

```bash
sha256sum -c SHA256SUMS
```

文件校验值、源代码 commit、运行环境和最终结果也记录在
`analysis/experiment_manifest.yaml` 中。

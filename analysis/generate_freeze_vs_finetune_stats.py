#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "analysis/raw_results/freeze"
FINETUNE_DIR = ROOT / "analysis/raw_results/finetune"
VISION_META = ROOT / "analysis/raw_results/vision/metadata.json"
CSV_PATH = ROOT / "analysis/freeze_vs_finetune_100_per_episode.csv"
REPORT_PATH = ROOT / "analysis/freeze_vs_finetune_100_comparison.md"


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def load_actions(path):
    pattern = re.compile(
        r"Seed\s+(\d+)\s+(success|failed).*?\n"
        r"steps:\s*(\d+)\s*,\s*actions:\s*(\d+)"
    )
    rows = {}
    for seed, result, steps, actions in pattern.findall(path.read_text(errors="replace")):
        rows[seed] = {
            "result": result,
            "steps": int(steps),
            "actions": int(actions),
        }
    return rows


def alignment(record):
    # rel_pose stores the quaternion as [qw, qx, qy, qz].
    _, _, _, _, qx, qy, _ = record["rel_pose"]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def failed_conditions(record):
    if record["result"] == "success":
        return []
    x, y, z = record["rel_pose"][:3]
    conditions = []
    if abs(x) >= 0.01:
        conditions.append("x")
    if abs(y) >= 0.01:
        conditions.append("y")
    if z >= -0.04:
        conditions.append("z_depth")
    if alignment(record) <= 0.99:
        conditions.append("orientation")
    if record.get("inhand_bias", -math.inf) >= 0.04:
        conditions.append("inhand_bias")
    return conditions


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def exact_mcnemar(left_only, right_only):
    n = left_only + right_only
    if n == 0:
        return 1.0
    low = min(left_only, right_only)
    tail = sum(math.comb(n, k) for k in range(low + 1)) / (2**n)
    return min(1.0, 2 * tail)


def fmt_seed_list(values):
    return ", ".join(f"`{value}`" for value in values)


def median_range(values, digits=2):
    return (
        f"{statistics.median(values):.{digits}f}"
        f"（{min(values):.{digits}f}–{max(values):.{digits}f}）"
    )


freeze = load_json(FREEZE_DIR / "metadata.json")
finetune = load_json(FINETUNE_DIR / "metadata.json")
vision = load_json(VISION_META)
freeze_actions = load_actions(FREEZE_DIR / "log.log")
finetune_actions = load_actions(FINETUNE_DIR / "log.log")

seeds = sorted(freeze)
assert seeds == sorted(finetune) == sorted(vision)
assert len(seeds) == 100
assert len(freeze_actions) == len(finetune_actions) == 100
assert all(freeze[s]["rotate"] == finetune[s]["rotate"] for s in seeds)


def succeeded(records, seed):
    return records[seed]["result"] == "success"


groups = {
    "both_success": [
        s for s in seeds if succeeded(freeze, s) and succeeded(finetune, s)
    ],
    "freeze_only": [
        s for s in seeds if succeeded(freeze, s) and not succeeded(finetune, s)
    ],
    "finetune_only": [
        s for s in seeds if not succeeded(freeze, s) and succeeded(finetune, s)
    ],
    "both_failed": [
        s for s in seeds if not succeeded(freeze, s) and not succeeded(finetune, s)
    ],
}

group_for_seed = {
    seed: group for group, group_seeds in groups.items() for seed in group_seeds
}


def row_fields(prefix, record, execution):
    conditions = failed_conditions(record)
    if record["result"] == "success":
        inhand_status = "pass_implied"
        stop_reason = "success"
    elif "inhand_bias" in record:
        inhand_status = "fail" if record["inhand_bias"] >= 0.04 else "pass"
        stop_reason = "inhand_early_stop" if record.get("early_stop") else "failed"
    else:
        inhand_status = "unknown_not_recorded"
        stop_reason = "action_limit" if execution["actions"] >= 300 else "failed"
    x, y, z = record["rel_pose"][:3]
    return {
        f"{prefix}_result": record["result"],
        f"{prefix}_failed_conditions": "+".join(conditions),
        f"{prefix}_rel_x_m": f"{x:.9f}",
        f"{prefix}_rel_y_m": f"{y:.9f}",
        f"{prefix}_rel_z_m": f"{z:.9f}",
        f"{prefix}_orientation_alignment": f"{alignment(record):.9f}",
        f"{prefix}_inhand_status": inhand_status,
        f"{prefix}_inhand_bias_m": (
            f"{record['inhand_bias']:.9f}" if "inhand_bias" in record else ""
        ),
        f"{prefix}_stop_reason": stop_reason,
        f"{prefix}_actions": execution["actions"],
        f"{prefix}_cost_step": record["cost_step"],
        f"{prefix}_cost_time_s": f"{record['cost_time']:.6f}",
    }


rows = []
for seed in seeds:
    rotate = "0" if abs(freeze[seed]["rotate"]) < 1e-9 else "pi"
    row = {"seed": seed, "rotate": rotate, "pair_group": group_for_seed[seed]}
    row.update(row_fields("freeze", freeze[seed], freeze_actions[seed]))
    row.update(row_fields("finetune", finetune[seed], finetune_actions[seed]))
    rows.append(row)

with CSV_PATH.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def condition_counts(records):
    failed = [record for record in records.values() if record["result"] == "failed"]
    counts = Counter()
    combinations = Counter()
    for record in failed:
        conditions = failed_conditions(record)
        counts.update(conditions)
        combinations["+".join(conditions)] += 1
    return failed, counts, combinations


freeze_failed, freeze_counts, freeze_combos = condition_counts(freeze)
finetune_failed, finetune_counts, finetune_combos = condition_counts(finetune)


def action_limit_count(records, actions):
    return sum(
        record["result"] == "failed" and actions[seed]["actions"] >= 300
        for seed, record in records.items()
    )


def result_metrics(records, actions, result):
    selected = [seed for seed in seeds if records[seed]["result"] == result]
    return {
        "actions": [actions[s]["actions"] for s in selected],
        "steps": [records[s]["cost_step"] for s in selected],
        "times": [records[s]["cost_time"] for s in selected],
    }


def inhand_metrics(records):
    values = [
        record["inhand_bias"]
        for record in records.values()
        if record["result"] == "failed"
        and record.get("early_stop")
        and "inhand_bias" in record
    ]
    return values


def rotate_stats(rotate_value):
    selected = [s for s in seeds if abs(freeze[s]["rotate"] - rotate_value) < 1e-9]
    return {
        "n": len(selected),
        "freeze": sum(succeeded(freeze, s) for s in selected),
        "finetune": sum(succeeded(finetune, s) for s in selected),
        "both": sum(
            succeeded(freeze, s) and succeeded(finetune, s) for s in selected
        ),
        "freeze_only": sum(
            succeeded(freeze, s) and not succeeded(finetune, s) for s in selected
        ),
        "finetune_only": sum(
            not succeeded(freeze, s) and succeeded(finetune, s) for s in selected
        ),
        "failed": sum(
            not succeeded(freeze, s) and not succeeded(finetune, s)
            for s in selected
        ),
    }


freeze_successes = sum(record["result"] == "success" for record in freeze.values())
finetune_successes = sum(
    record["result"] == "success" for record in finetune.values()
)
vision_successes = sum(record["result"] == "success" for record in vision.values())
freeze_ci = wilson(freeze_successes, 100)
finetune_ci = wilson(finetune_successes, 100)
mcnemar_p = exact_mcnemar(len(groups["freeze_only"]), len(groups["finetune_only"]))


def combo_lines(counter):
    lines = []
    for combination, count in counter.most_common():
        label = f"`{combination}`"
        if "inhand_bias" not in combination:
            label += "，inhand 未记录，达到动作上限"
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines)


metrics = {}
for name, records, actions in [
    ("Freeze", freeze, freeze_actions),
    ("Finetune", finetune, finetune_actions),
]:
    for result in ("success", "failed"):
        metrics[(name, result)] = result_metrics(records, actions, result)

rot0 = rotate_stats(0.0)
rotpi = rotate_stats(math.pi)
freeze_inhand = inhand_metrics(freeze)
finetune_inhand = inhand_metrics(finetune)

report = f"""# Insert Hole：Freeze 与 Finetune 的 100 条固定 Seed 评测对比

生成日期：2026-07-24

## 1. 数据来源与可比性

- Freeze：
  - 归档目录：`analysis/raw_results/freeze`
  - 原始结果目录：`eval_result/ACT/insert_hole/deploy_freeze/2026-07-23_19:57:07`
  - 模型：`train_config_freeze_verified/policy_last.ckpt`
  - `lr_tactile_backbone = 0`
  - 最终结果：{freeze_successes}/100（{freeze_successes:.2f}%）
- Finetune：
  - 归档目录：`analysis/raw_results/finetune`
  - 原始结果目录：`eval_result/ACT/insert_hole/deploy_finetune/2026-07-24_12:05:06`
  - 模型：`train_config/policy_last.ckpt`
  - `lr_tactile_backbone = 1e-5`
  - 最终结果：{finetune_successes}/100（{finetune_successes:.2f}%）
- 两组均使用 seed `1000000–1000099`，无缺失、无重复。
- 两组相同 seed 的 `rotate` 完全一致，checkpoint 均显示权重完整匹配。
- 作为参照，Vision 为 {vision_successes}/100（{vision_successes:.2f}%）。

## 2. 成功结果的配对关系

| 配对结果 | 条数 |
|---|---:|
| 两组共同成功 | {len(groups["both_success"])} |
| 仅 Freeze 成功 | {len(groups["freeze_only"])} |
| 仅 Finetune 成功 | {len(groups["finetune_only"])} |
| 两组共同失败 | {len(groups["both_failed"])} |
| 合计 | 100 |

Freeze 与 Finetune 的成功率差为 {freeze_successes - finetune_successes} 个百分点。
Freeze 的 Wilson 95% 置信区间为 {freeze_ci[0]*100:.2f}%–{freeze_ci[1]*100:.2f}%，
Finetune 为 {finetune_ci[0]*100:.2f}%–{finetune_ci[1]*100:.2f}%。
针对 {len(groups["freeze_only"]) + len(groups["finetune_only"])} 个结果不一致的配对，
精确 McNemar 检验 `p = {mcnemar_p:.3f}`。当前结果没有显示两组总体成功率存在显著差异。

### 2.1 两组共同成功（{len(groups["both_success"])}）

{fmt_seed_list(groups["both_success"])}

### 2.2 仅 Freeze 成功（{len(groups["freeze_only"])}）

{fmt_seed_list(groups["freeze_only"])}

这些 seed 中 Finetune 的失败条件组合：

{chr(10).join(f"- `{combo}`：{count} 条" for combo, count in Counter("+".join(failed_conditions(finetune[s])) for s in groups["freeze_only"]).most_common())}

### 2.3 仅 Finetune 成功（{len(groups["finetune_only"])}）

{fmt_seed_list(groups["finetune_only"])}

这些 seed 中 Freeze 的失败条件组合：

{chr(10).join(f"- `{combo}`：{count} 条" for combo, count in Counter("+".join(failed_conditions(freeze[s])) for s in groups["finetune_only"]).most_common())}

### 2.4 两组共同失败（{len(groups["both_failed"])}）

{fmt_seed_list(groups["both_failed"])}

## 3. 失败条件统计

判定阈值：

- `x`：`abs(rel_x) < 0.01 m`
- `y`：`abs(rel_y) < 0.01 m`
- `z_depth`：`rel_z < -0.04 m`
- `orientation`：`orientation_alignment > 0.99`
- `inhand_bias`：`inhand_bias < 0.04 m`

以下条件可以重叠，因此各列不能直接相加。

| 模型 | 失败数 | x 未通过 | y 未通过 | z 深度未通过 | 姿态未通过 | inhand early-stop | 300 actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Freeze | {len(freeze_failed)} | {freeze_counts["x"]} | {freeze_counts["y"]} | {freeze_counts["z_depth"]} | {freeze_counts["orientation"]} | {freeze_counts["inhand_bias"]} | {action_limit_count(freeze, freeze_actions)} |
| Finetune | {len(finetune_failed)} | {finetune_counts["x"]} | {finetune_counts["y"]} | {finetune_counts["z_depth"]} | {finetune_counts["orientation"]} | {finetune_counts["inhand_bias"]} | {action_limit_count(finetune, finetune_actions)} |

关键观察：

- Finetune 的 73 条失败中有 72 条插入深度未通过；唯一例外是 seed
  `1000017`，其 x/y、深度和姿态均通过，但 `inhand_bias = 0.040266 m`
  超过阈值，因此仍判定失败。
- Finetune 有 72/73 条失败由 inhand early-stop 直接终止，Freeze 为 70/72。
- Finetune 的姿态未通过略少（36 对 38），但 x 横向误差未通过更多（11 对 3）；
  这些是终止帧统计，不能单独证明解冻导致某一控制能力退化。

### 3.1 Freeze 的失败条件组合

| 条件组合 | 条数 |
|---|---:|
{combo_lines(freeze_combos)}

### 3.2 Finetune 的失败条件组合

| 条件组合 | 条数 |
|---|---:|
{combo_lines(finetune_combos)}

## 4. 按孔槽旋转方向分层

| rotate | Episode 数 | Freeze 成功 | Finetune 成功 | 共同成功 | 仅 Freeze | 仅 Finetune | 共同失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | {rot0["n"]} | {rot0["freeze"]}（{rot0["freeze"]/rot0["n"]*100:.2f}%） | {rot0["finetune"]}（{rot0["finetune"]/rot0["n"]*100:.2f}%） | {rot0["both"]} | {rot0["freeze_only"]} | {rot0["finetune_only"]} | {rot0["failed"]} |
| `π` | {rotpi["n"]} | {rotpi["freeze"]}（{rotpi["freeze"]/rotpi["n"]*100:.2f}%） | {rotpi["finetune"]}（{rotpi["finetune"]/rotpi["n"]*100:.2f}%） | {rotpi["both"]} | {rotpi["freeze_only"]} | {rotpi["finetune_only"]} | {rotpi["failed"]} |

## 5. 动作、仿真步数与耗时

表中为中位数，括号内为最小值–最大值。

| 模型/结果 | actions | cost_step | cost_time |
|---|---:|---:|---:|
| Freeze 成功 | {median_range(metrics[("Freeze", "success")]["actions"], 1)} | {median_range(metrics[("Freeze", "success")]["steps"], 1)} | {median_range(metrics[("Freeze", "success")]["times"], 2)} s |
| Freeze 失败 | {median_range(metrics[("Freeze", "failed")]["actions"], 1)} | {median_range(metrics[("Freeze", "failed")]["steps"], 1)} | {median_range(metrics[("Freeze", "failed")]["times"], 2)} s |
| Finetune 成功 | {median_range(metrics[("Finetune", "success")]["actions"], 1)} | {median_range(metrics[("Finetune", "success")]["steps"], 1)} | {median_range(metrics[("Finetune", "success")]["times"], 2)} s |
| Finetune 失败 | {median_range(metrics[("Finetune", "failed")]["actions"], 1)} | {median_range(metrics[("Finetune", "failed")]["steps"], 1)} | {median_range(metrics[("Finetune", "failed")]["times"], 2)} s |

`cost_step` 包含 reset、预移动和策略执行；比较策略执行长度时应优先使用日志中的
`actions`。

early-stop 失败的 `inhand_bias`：

| 模型 | 数量 | 最小值 | 中位数 | 均值 | 最大值 |
|---|---:|---:|---:|---:|---:|
| Freeze | {len(freeze_inhand)} | {min(freeze_inhand):.6f} | {statistics.median(freeze_inhand):.6f} | {statistics.mean(freeze_inhand):.6f} | {max(freeze_inhand):.6f} |
| Finetune | {len(finetune_inhand)} | {min(finetune_inhand):.6f} | {statistics.median(finetune_inhand):.6f} | {statistics.mean(finetune_inhand):.6f} | {max(finetune_inhand):.6f} |

## 6. 逐条记录

逐 seed 的完整记录保存在：

`analysis/freeze_vs_finetune_100_per_episode.csv`

CSV 包含配对结果、两组失败条件、最终相对位姿、姿态对齐值、inhand bias、
终止原因、actions、cost_step 和 cost_time。

## 7. 简要结论

- Freeze 为 {freeze_successes}%，Finetune 为 {finetune_successes}%，只相差
  {freeze_successes - finetune_successes} 个百分点；配对检验不支持显著差异。
- 两组共同成功 {len(groups["both_success"])} 条，但各有
  {len(groups["freeze_only"])} 条和 {len(groups["finetune_only"])} 条独有成功，
  说明总体成功率相近并不代表逐 seed 行为完全一致。
- 两组失败仍主要集中在插入深度不足、夹持内滑动，以及部分姿态未对齐。
- 在 50 条示范和 4000 step 的当前设置下，直接以 `1e-5` 解冻触觉编码器没有显示
  优于冻结编码器；这可能与小数据微调不稳定、学习率偏大或最后 checkpoint
  相对最佳验证 checkpoint 轻微退化有关。
- 当前评测均使用 `policy_last.ckpt`，因此比较口径公平；若研究 best checkpoint，
  应对所有模型统一重测。
"""

REPORT_PATH.write_text(report)
print(REPORT_PATH)
print(CSV_PATH)

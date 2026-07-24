# Insert Hole ACT Vision–Tactile Ablation

This release archives the evaluated model weights and complete evaluation videos for the
50-demonstration, 4000-step ACT experiment documented in `analysis/task_report.md`.

## Results

| Variant | Input | Tactile encoder | Success |
|---|---|---|---:|
| Vision | Third-view camera | Not used | 28/100 |
| Freeze | Vision + bilateral tactile | Pretrained, frozen | 28/100 |
| Finetune | Vision + bilateral tactile | Pretrained, learning rate `1e-5` | 27/100 |

All variants used training seed `0` and the same fixed evaluation seeds
`1000000–1000099`. The released files named `*_policy_last.ckpt` are the exact
checkpoints loaded by the formal evaluations.

## Assets

- `vision_policy_last.ckpt`: evaluated Vision checkpoint;
- `freeze_policy_last.ckpt`: evaluated pretrained-and-frozen tactile checkpoint;
- `finetune_policy_last.ckpt`: evaluated pretrained-and-finetuned tactile checkpoint;
- `tactile_encoder_best.pth`: common pretrained tactile encoder;
- `final_eval_videos.tar.gz`: all 300 formal evaluation videos, grouped by variant;
- `SHA256SUMS`: integrity checks for every asset above.

The repository contains the normalization statistics, compact raw metadata/logs, training
curves, exact configurations, paired per-seed CSV files, experiment manifest, and
reproduction commands.

Large raw and converted HDF5 datasets are intentionally excluded. They total approximately
15 GB and should be preserved in a dedicated dataset store with a separate checksum
manifest.

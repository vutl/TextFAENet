# FMISeg QaTa Legacy Dual Metrics

> INVALID FOR PAPER COMPARISON.
>
> This legacy wrapper output is inconsistent with the official FMISeg evaluation
> (`FMISeg/eval_output_qatacov19v2.log`: `test_dice=0.909438`,
> `test_MIoU=0.833917`). Do not use the values below in tables/figures.
> Use `scripts/evaluate_fmiseg_official_qata_per_image.py` instead.

- Checkpoint: `D:\Documents\LMIS\FMISeg\save_model\last.ckpt`
- Legacy model files: git commit `c22b6d7`
- Threshold: `0.5`
- Test images: `2113`

| Output | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---:|---:|---:|---:|
| First branch (original wrapper) | 0.506083 | 0.642366 | 0.383648 | 0.473151 |
| Two-branch ensemble | 0.511399 | 0.648764 | 0.388697 | 0.480126 |

Original FMISeg code returned only the first output branch to TorchMetrics. The ensemble row is added for reference.

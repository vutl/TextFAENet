# FMISeg QaTa Legacy Dual Metrics

- Checkpoint: `D:\Documents\LMIS\FMISeg\save_model\last.ckpt`
- Legacy model files: git commit `c22b6d7`
- Threshold: `0.5`
- Test images: `1`

| Output | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---:|---:|---:|---:|
| First branch (original wrapper) | 0.782365 | 0.782365 | 0.642528 | 0.642528 |
| Two-branch ensemble | 0.805522 | 0.805522 | 0.674372 | 0.674372 |

Original FMISeg code returned only the first output branch to TorchMetrics. The ensemble row is added for reference.

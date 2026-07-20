# QaTa Dual-Metric Evaluation

`Per-image` is the arithmetic mean of image-level scores. `Global` pools all test pixels before computing the score.

| Run | Checkpoint | Thr | Images | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `paper_missing0715_qata_resnet50_simple_native_keep_decoder_seed42` | epoch 9 | 0.35 | 2113 | 0.829276 | 0.902882 | 0.742293 | 0.822957 |

## Configuration

- `paper_missing0715_qata_resnet50_simple_native_keep_decoder_seed42`: `resnet50_tgfs_v2`, visual `imagenet`, text `simple`, fusion `decoder`, HH `keep`. Per-image rows: `runs\paper_missing0715_qata_resnet50_simple_native_keep_decoder_seed42\test_per_image_metrics.csv`.

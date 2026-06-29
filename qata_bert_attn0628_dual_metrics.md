# QaTa Dual-Metric Evaluation

`Per-image` is the arithmetic mean of image-level scores. `Global` pools all test pixels before computing the score.

| Run | Checkpoint | Thr | Images | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qata_bert_attn0628_qata_cxrbert_v3_both_crossattn_keep_seed42` | epoch 16 | 0.35 | 2113 | 0.804924 | 0.889348 | 0.711355 | 0.800744 |

## Configuration

- `qata_bert_attn0628_qata_cxrbert_v3_both_crossattn_keep_seed42`: `lfaenet_tgfs_v3`, visual `resnet50`, text `cxr_bert`, fusion `both`, HH `keep`. Per-image rows: `runs\qata_bert_attn0628_qata_cxrbert_v3_both_crossattn_keep_seed42\test_per_image_metrics.csv`.

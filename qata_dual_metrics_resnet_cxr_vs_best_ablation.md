# QaTa Dual-Metric Evaluation

`Per-image` is the arithmetic mean of image-level scores. `Global` pools all test pixels before computing the score.

| Run | Checkpoint | Thr | Images | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` | epoch 13 | 0.35 | 2113 | 0.803207 | 0.888346 | 0.709589 | 0.799121 |
| `qata_paper0516_qata_simple_native_zero_both_seed42` | epoch 18 | 0.35 | 2113 | 0.826091 | 0.900397 | 0.737495 | 0.818839 |

## Configuration

- `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42`: `resnet50_tgfs_v2`, visual `imagenet`, text `cxr_bert`, fusion `decoder`, HH `keep`. Per-image rows: `runs\qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42\test_per_image_metrics.csv`.
- `qata_paper0516_qata_simple_native_zero_both_seed42`: `lfaenet_tgfs_v2`, visual `from_scratch`, text `simple`, fusion `both`, HH `zero`. Per-image rows: `runs\qata_paper0516_qata_simple_native_zero_both_seed42\test_per_image_metrics.csv`.

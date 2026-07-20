# QaTa Dual-Metric Evaluation

`Per-image` is the arithmetic mean of image-level scores. `Global` pools all test pixels before computing the score.

| Run | Checkpoint | Thr | Images | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `paper_missing0715_qata_resnet50_simple_native_drop_ll_decoder_seed42` | epoch 19 | 0.35 | 2113 | 0.835619 | 0.906789 | 0.748626 | 0.829473 |
| `paper_missing0715_qata_resnet50_simple_native_drop_lh_decoder_seed42` | epoch 9 | 0.35 | 2113 | 0.835422 | 0.906908 | 0.749001 | 0.829672 |
| `paper_missing0715_qata_resnet50_simple_native_drop_hl_decoder_seed42` | epoch 24 | 0.35 | 2113 | 0.837643 | 0.909026 | 0.752095 | 0.833224 |
| `paper_missing0715_qata_resnet50_simple_native_drop_hh_decoder_seed42` | epoch 5 | 0.55 | 2113 | 0.832533 | 0.902707 | 0.745327 | 0.822668 |

## Configuration

- `paper_missing0715_qata_resnet50_simple_native_drop_ll_decoder_seed42`: `resnet50_tgfs_v2`, visual `imagenet`, text `simple`, fusion `decoder`, HH `keep`. Per-image rows: `runs\paper_missing0715_qata_resnet50_simple_native_drop_ll_decoder_seed42\test_per_image_metrics.csv`.
- `paper_missing0715_qata_resnet50_simple_native_drop_lh_decoder_seed42`: `resnet50_tgfs_v2`, visual `imagenet`, text `simple`, fusion `decoder`, HH `keep`. Per-image rows: `runs\paper_missing0715_qata_resnet50_simple_native_drop_lh_decoder_seed42\test_per_image_metrics.csv`.
- `paper_missing0715_qata_resnet50_simple_native_drop_hl_decoder_seed42`: `resnet50_tgfs_v2`, visual `imagenet`, text `simple`, fusion `decoder`, HH `keep`. Per-image rows: `runs\paper_missing0715_qata_resnet50_simple_native_drop_hl_decoder_seed42\test_per_image_metrics.csv`.
- `paper_missing0715_qata_resnet50_simple_native_drop_hh_decoder_seed42`: `resnet50_tgfs_v2`, visual `imagenet`, text `simple`, fusion `decoder`, HH `keep`. Per-image rows: `runs\paper_missing0715_qata_resnet50_simple_native_drop_hh_decoder_seed42\test_per_image_metrics.csv`.

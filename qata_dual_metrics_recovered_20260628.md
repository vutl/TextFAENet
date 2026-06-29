# QaTa Dual-Metric Evaluation

`Per-image` is the arithmetic mean of image-level scores. `Global` pools all test pixels before computing the score.

| Run | Checkpoint | Thr | Images | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qata_b4_e50_cxrbert_frozen_v2` | epoch 26 | 0.50 | 2113 | 0.816922 | 0.893188 | 0.725478 | 0.806991 |
| `qata_b4_e50_cxrbert_frozen_v2_rerun` | epoch 6 | 0.50 | 2113 | 0.802817 | 0.883089 | 0.707854 | 0.790653 |
| `qata_faenet_notext_valclean_e5` | epoch 2 | 0.50 | 2113 | 0.626855 | 0.737254 | 0.511161 | 0.583850 |
| `qata_faenet_notext_adamw_cosine_e30` | epoch 13 | 0.50 | 2113 | 0.163964 | 0.195463 | 0.093334 | 0.108318 |

## Configuration

- `qata_b4_e50_cxrbert_frozen_v2`: `lfaenet_tgfs_v2`, visual `from_scratch`, text `cxr_bert`, fusion `decoder`, HH `keep`. Per-image rows: `runs\qata_b4_e50_cxrbert_frozen_v2\test_per_image_metrics.csv`.
- `qata_b4_e50_cxrbert_frozen_v2_rerun`: `lfaenet_tgfs_v2`, visual `from_scratch`, text `cxr_bert`, fusion `decoder`, HH `keep`. Per-image rows: `runs\qata_b4_e50_cxrbert_frozen_v2_rerun\test_per_image_metrics.csv`.
- `qata_faenet_notext_valclean_e5`: `faenet`, visual `from_scratch`, text `cxr_bert`, fusion `decoder`, HH `keep`. Per-image rows: `runs\qata_faenet_notext_valclean_e5\test_per_image_metrics.csv`.
- `qata_faenet_notext_adamw_cosine_e30`: `faenet`, visual `from_scratch`, text `cxr_bert`, fusion `decoder`, HH `keep`. Per-image rows: `runs\qata_faenet_notext_adamw_cosine_e30\test_per_image_metrics.csv`.

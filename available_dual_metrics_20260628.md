# Available Per-Image and Global Metrics - 2026-06-28

This file aggregates runs referenced by:

- `brain_breast_prompt_protocol_results_20260626.md`
- `qata_dual_metrics_resnet_cxr_vs_best_ablation.md`
- `qata_results_summary_20260530.md`
- `qata_dual_metrics_resnet_cxr_vs_best_ablation.json`

Rules:

- If `test_dual_metrics.json` exists, use it.
- Else if `test_per_image_metrics.csv` exists, recompute per-image and global metrics from stored intersections/pixel counts.
- Else if `final_test.json` contains `global_*`, use it.
- Else use `final_test.json` Dice/IoU as per-image only and leave global as `-`.
- If a checkpoint remains, `can_rerun_from_checkpoint=yes` means global metrics can be recovered by rerunning inference.

Summary: `32` runs have both per-image and global metrics now; `5` have per-image only; `3` have no final test metrics; `0` can be rerun from a checkpoint to recover global metrics.

## Metrics Table

| Run | Status | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Best epoch | Thr | Images | Source | Checkpoint rerun? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `paper0623_brain_structured_v3resnet50cxr_both_seed42` | dual_available | 83.85 | 74.67 | 83.88 | 72.23 | 39 | 0.35 | 600 | `final_test.json` | no |
| `paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42` | dual_available | 84.15 | 75.51 | 84.54 | 73.21 | 40 | 0.55 | 600 | `final_test.json` | no |
| `paper0623_breast_structured_v3resnet50cxr_both_seed42` | dual_available | 85.18 | 76.54 | 87.82 | 78.29 | 14 | 0.55 | 113 | `final_test.json` | no |
| `paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42` | dual_available | 80.23 | 71.79 | 80.51 | 67.38 | 15 | 0.35 | 113 | `final_test.json` | no |
| `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` | dual_available | 80.32 | 70.96 | 88.83 | 79.91 | 13 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_paper0516_qata_simple_native_zero_both_seed42` | dual_available | 82.61 | 73.75 | 90.04 | 81.88 | 18 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_paper0516_qata_simple_native_learned_both_seed42` | dual_available | 81.90 | 72.91 | 89.79 | 81.47 | 21 | 0.5 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42` | dual_available | 81.99 | 72.97 | 89.90 | 81.65 | 21 | 0.4 | 2113 | `test_dual_metrics.json` | no |
| `qata_diag0516_qata_simple_native_keep_both_seed42` | dual_available | 82.13 | 73.07 | 89.45 | 80.91 | 11 | 0.55 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42` | per_image_only | 82.10 | 73.24 | - | - | 11 | 0.45 | - | `final_test.json_per_image_only` | no |
| `qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42` | dual_available | 82.09 | 73.12 | 90.03 | 81.87 | 24 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42` | per_image_only | 81.96 | 72.85 | - | - | 13 | 0.5 | - | `final_test.json_per_image_only` | no |
| `qata_b4_e50_cxrbert_frozen_v2` | dual_available | 81.69 | 72.55 | 89.32 | 80.70 | 26 | 0.5 | 2113 | `test_dual_metrics.json` | no |
| `qata_paper0516_qata_simple_native_keep_decoder_seed42` | dual_available | 81.64 | 72.44 | 89.48 | 80.97 | 17 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42` | per_image_only | 81.63 | 72.59 | - | - | 12 | 0.35 | - | `final_test.json_per_image_only` | no |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42` | dual_available | 81.60 | 72.53 | 89.58 | 81.12 | 12 | 0.5 | 2113 | `test_dual_metrics.json` | no |
| `screening0506_qata_cxr_frozen_keep_both_seed42` | dual_available | 81.54 | 72.42 | 89.37 | 80.79 | 25 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_diag0516_qata_cxr_frozen_learned_both_seed42` | dual_available | 81.19 | 71.69 | 88.65 | 79.61 | 7 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42` | per_image_only | 81.08 | 71.85 | - | - | 17 | 0.35 | - | `final_test.json_per_image_only` | no |
| `screening0506_qata_cxr_frozen_keep_decoder_seed42` | dual_available | 81.27 | 72.01 | 88.96 | 80.11 | 24 | 0.4 | 2113 | `test_dual_metrics.json` | no |
| `qata_b4_e50_cxrbert_frozen_v2_rerun` | dual_available | 80.28 | 70.79 | 88.31 | 79.07 | 6 | 0.5 | 2113 | `test_dual_metrics.json` | no |
| `qata_v3_remote_main_cxr_seed42` | dual_available | 79.93 | 70.57 | 88.29 | 79.03 | 5 | 0.45 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_qata_resnet50_simple_shuffle_keep_decoder_seed42` | dual_available | 79.37 | 69.83 | 87.66 | 78.04 | 34 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_qata_resnet50_simple_generic_keep_decoder_seed42` | dual_available | 79.25 | 69.65 | 87.50 | 77.78 | 20 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42` | dual_available | 78.61 | 68.84 | 87.08 | 77.11 | 37 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` | dual_available | 77.94 | 67.87 | 86.49 | 76.19 | 34 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_paper0516_qata_faenet_visual_clean_seed42` | dual_available | 77.57 | 67.66 | 86.12 | 75.62 | 30 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_paper0516_qata_simple_empty_keep_both_seed42` | dual_available | 76.59 | 66.68 | 85.43 | 74.57 | 15 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42` | dual_available | 76.42 | 66.24 | 84.81 | 73.63 | 17 | 0.4 | 2113 | `test_dual_metrics.json` | no |
| `qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42` | dual_available | 76.41 | 66.48 | 84.76 | 73.55 | 9 | 0.45 | 2113 | `test_dual_metrics.json` | no |
| `qata_diag0516_qata_simple_generic_keep_both_seed42` | dual_available | 75.99 | 65.75 | 84.16 | 72.65 | 7 | 0.45 | 2113 | `test_dual_metrics.json` | no |
| `qata_paper0516_qata_simple_shuffle_keep_both_seed42` | dual_available | 75.86 | 65.80 | 84.68 | 73.44 | 8 | 0.35 | 2113 | `test_dual_metrics.json` | no |
| `qata_faenet_notext_valclean_e5` | dual_available | 62.69 | 51.12 | 73.73 | 58.38 | 2 | 0.5 | 2113 | `test_dual_metrics.json` | no |
| `qata_faenet_notext_adamw_cosine_e30` | dual_available | 16.40 | 9.33 | 19.55 | 10.83 | 13 | 0.5 | 2113 | `test_dual_metrics.json` | no |
| `qata_diag0516_qata_cxr_lora8_keep_both_seed42` | dual_available | 80.22 | 70.59 | 87.80 | 78.25 | 4 | 0.55 | 2113 | `test_dual_metrics.json` | no |
| `qata_resnet0523_qata_resnet50_simple_empty_keep_decoder_seed42` | no_final | - | - | - | - | - | - | - | `-` | no |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hh_decoder_seed42` | no_final | - | - | - | - | - | - | - | `-` | no |
| `qata_resnet0523_qata_resnet50_simple_native_drop_ll_decoder_seed42` | no_final | - | - | - | - | - | - | - | `-` | no |
| `qata_resnet0523_qata_resnet50_simple_native_keep_decoder_seed42` | dual_available | 77.11 | 66.86 | 86.07 | 75.55 | 1 | 0.55 | 2113 | `test_dual_metrics.json` | no |
| `screening0506_mosmed_cxr_frozen_keep_both_seed42` | per_image_only | 67.37 | 52.90 | - | - | 7 | 0.5 | - | `final_test.json_per_image_only` | no |

## Runs Needing Inference Rerun For Global Metrics

These runs have old per-image-only `final_test.json` metrics but still have enough checkpoint/config files to recompute global Dice/IoU.

- None.

## Runs That Cannot Be Recovered Without Retraining Or A Checkpoint

- `qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42`: status `per_image_only`, source `final_test.json_per_image_only`
- `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42`: status `per_image_only`, source `final_test.json_per_image_only`
- `qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42`: status `per_image_only`, source `final_test.json_per_image_only`
- `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42`: status `per_image_only`, source `final_test.json_per_image_only`
- `qata_resnet0523_qata_resnet50_simple_empty_keep_decoder_seed42`: status `no_final`, source `-`
- `qata_resnet0523_qata_resnet50_simple_native_drop_hh_decoder_seed42`: status `no_final`, source `-`
- `qata_resnet0523_qata_resnet50_simple_native_drop_ll_decoder_seed42`: status `no_final`, source `-`
- `screening0506_mosmed_cxr_frozen_keep_both_seed42`: status `per_image_only`, source `final_test.json_per_image_only`

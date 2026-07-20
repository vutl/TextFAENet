# Tables 2-5 dual-metric ledger

All local values below are full-test results. `Reported` preserves the number copied from a source paper when prediction-level artifacts are unavailable; it is not automatically labelled per-image or global.

## Table 2

| Method | Dataset/group | Variant/prompt | Reported Dice | Reported IoU | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Provenance/artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| UNet | Brain MRI | source-reported baseline | 72.16 | 60.51 | - | - | - | - | source_reported_aggregation_not_recomputed |
| UNet | Breast US | source-reported baseline | 71.54 | 62.34 | - | - | - | - | source_reported_aggregation_not_recomputed |
| UNet | QaTa-COV19 | source-reported baseline | 78.45 | 68.76 | - | - | - | - | source_reported_aggregation_not_recomputed |
| UNet | MosMedData+ | source-reported baseline | 64.58 | 50.73 | - | - | - | - | source_reported_aggregation_not_recomputed |
| nnUNet | Brain MRI | source-reported baseline | 77.76 | 67.71 | - | - | - | - | source_reported_aggregation_not_recomputed |
| nnUNet | Breast US | source-reported baseline | 73.77 | 63.77 | - | - | - | - | source_reported_aggregation_not_recomputed |
| nnUNet | QaTa-COV19 | source-reported baseline | 80.42 | 70.81 | - | - | - | - | source_reported_aggregation_not_recomputed |
| nnUNet | MosMedData+ | source-reported baseline | 72.59 | 60.36 | - | - | - | - | source_reported_aggregation_not_recomputed |
| TransUNet | Brain MRI | source-reported baseline | 80.83 | 71.52 | - | - | - | - | source_reported_aggregation_not_recomputed |
| TransUNet | Breast US | source-reported baseline | 80.60 | 71.68 | - | - | - | - | source_reported_aggregation_not_recomputed |
| TransUNet | QaTa-COV19 | source-reported baseline | 78.63 | 69.13 | - | - | - | - | source_reported_aggregation_not_recomputed |
| TransUNet | MosMedData+ | source-reported baseline | 71.24 | 58.44 | - | - | - | - | source_reported_aggregation_not_recomputed |
| VT-MFLV | Brain MRI | source-reported baseline | 84.63 | 75.37 | - | - | - | - | source_reported_aggregation_not_recomputed |
| VT-MFLV | Breast US | source-reported baseline | 78.05 | 67.15 | - | - | - | - | source_reported_aggregation_not_recomputed |
| VT-MFLV | QaTa-COV19 | source-reported baseline | 83.34 | 72.09 | - | - | - | - | source_reported_aggregation_not_recomputed |
| VT-MFLV | MosMedData+ | source-reported baseline | 75.61 | 63.98 | - | - | - | - | source_reported_aggregation_not_recomputed |
| STPNet | Brain MRI | source-reported baseline | 79.66 | 69.62 | - | - | - | - | source_reported_aggregation_not_recomputed |
| STPNet | Breast US | source-reported baseline | 71.25 | 60.13 | - | - | - | - | source_reported_aggregation_not_recomputed |
| STPNet | QaTa-COV19 | source-reported baseline | 80.63 | 71.42 | - | - | - | - | source_reported_aggregation_not_recomputed |
| STPNet | MosMedData+ | source-reported baseline | 76.18 | 63.41 | - | - | - | - | source_reported_aggregation_not_recomputed |
| MedCLIP-SAMv2 | Brain MRI | our structured prompt | - | - | 0.01 | 0.01 | 0.04 | 0.02 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_brain_structured\summary.json` |
| MedCLIP-SAMv2 | Breast US | our structured prompt | - | - | 5.04 | 2.66 | 5.49 | 2.82 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_breast_structured\summary.json` |
| MedCLIP-SAMv2 | QaTa-COV19 | our structured prompt | - | - | 20.77 | 12.30 | 22.51 | 12.68 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_qata_structured\summary.json` |
| MedCLIP-SAMv2 | MosMedData+ | our structured prompt | - | - | 0.13 | 0.07 | 0.47 | 0.24 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_mosmed_structured\summary.json` |
| FMISeg | Brain MRI | our structured prompt | - | - | 84.42 | 75.34 | 84.75 | 73.54 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_runs\fmiseg_brain_structured\final_test_metrics.json` |
| FMISeg | Breast US | our structured prompt | - | - | 84.23 | 75.35 | 84.87 | 73.71 | local_full_test; `D:\Documents\LMIS\Text-FAENet\..\FMISeg\checkpoints\breast_tumors_text\final_test_metrics.json` |
| FMISeg | QaTa-COV19 | our structured prompt | - | - | 84.58 | 76.17 | 91.14 | 83.72 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\fmiseg_qata_official\summary.json` |
| FMISeg | MosMedData+ | our structured prompt | - | - | 68.07 | 53.82 | 71.66 | 55.84 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_runs\fmiseg_mosmed_structured\final_test_metrics.json` |
| LFAENet-TGFS (ours) | Brain MRI | selected local full run | - | - | 83.85 | 74.67 | 83.88 | 72.23 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper0623_brain_structured_v3resnet50cxr_both_seed42\final_test.json` |
| LFAENet-TGFS (ours) | Breast US | selected local full run | - | - | 85.18 | 76.54 | 87.82 | 78.29 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper0623_breast_structured_v3resnet50cxr_both_seed42\final_test.json` |
| LFAENet-TGFS (ours) | QaTa-COV19 | selected local full run | - | - | 81.99 | 72.97 | 89.90 | 81.65 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42\test_dual_metrics.json` |
| LFAENet-TGFS (ours) | MosMedData+ | selected local full run | - | - | 72.20 | 59.09 | 79.54 | 66.03 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\mosmed_v9e_448\final_test.json` |

## Table 3

| Method | Dataset/group | Variant/prompt | Reported Dice | Reported IoU | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Provenance/artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| MedCLIP-SAMv2 | Brain MRI | original prompt (paper-reported) | 80.03 | 70.71 | - | - | - | - | source_reported_aggregation_not_recomputed |
| MedCLIP-SAMv2 | Brain MRI | original prompt (local reproduction) | - | - | 59.41 | 48.94 | 51.35 | 34.54 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_brain_native\summary.json` |
| MedCLIP-SAMv2 | Brain MRI | our structured prompt | - | - | 0.01 | 0.01 | 0.04 | 0.02 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_brain_structured\summary.json` |
| MedCLIP-SAMv2 | Breast US | original prompt (paper-reported) | 78.87 | 69.08 | - | - | - | - | source_reported_aggregation_not_recomputed |
| MedCLIP-SAMv2 | Breast US | original prompt (local reproduction) | - | - | 83.10 | 73.36 | 81.71 | 69.07 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_breast_native\summary.json` |
| MedCLIP-SAMv2 | Breast US | our structured prompt | - | - | 5.04 | 2.66 | 5.49 | 2.82 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_metrics\medclipsamv2_breast_structured\summary.json` |
| FMISeg | Brain MRI | original MedCLIP-SAMv2-style prompt | - | - | 81.55 | 71.16 | 80.97 | 68.03 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_runs\fmiseg_brain_medclip_prompt\final_test_metrics.json` |
| FMISeg | Breast US | original MedCLIP-SAMv2-style prompt | - | - | 81.87 | 73.14 | 82.69 | 70.49 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_runs\fmiseg_breast_medclip_prompt\final_test_metrics.json` |
| FMISeg | Brain MRI | our structured prompt | - | - | 84.42 | 75.34 | 84.75 | 73.54 | local_full_test; `D:\Documents\LMIS\Text-FAENet\external_runs\fmiseg_brain_structured\final_test_metrics.json` |
| FMISeg | Breast US | our structured prompt | - | - | 84.23 | 75.35 | 84.87 | 73.71 | local_full_test; `D:\Documents\LMIS\Text-FAENet\..\FMISeg\checkpoints\breast_tumors_text\final_test_metrics.json` |
| LFAENet-TGFS (ours) | Brain MRI | original MedCLIP-SAMv2-style prompt | - | - | 84.15 | 75.51 | 84.54 | 73.21 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42\final_test.json` |
| LFAENet-TGFS (ours) | Breast US | original MedCLIP-SAMv2-style prompt | - | - | 80.23 | 71.79 | 80.51 | 67.38 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42\final_test.json` |
| LFAENet-TGFS (ours) | Brain MRI | our structured prompt | - | - | 83.85 | 74.67 | 83.88 | 72.23 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper0623_brain_structured_v3resnet50cxr_both_seed42\final_test.json` |
| LFAENet-TGFS (ours) | Breast US | our structured prompt | - | - | 85.18 | 76.54 | 87.82 | 78.29 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper0623_breast_structured_v3resnet50cxr_both_seed42\final_test.json` |

## Table 4

| Method | Dataset/group | Variant/prompt | Reported Dice | Reported IoU | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Provenance/artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| QaTa ablation | A | FAENet visual-only, scratch | - | - | 77.57 | 67.66 | 86.12 | 75.62 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_paper0516_qata_faenet_visual_clean_seed42\test_dual_metrics.json` |
| QaTa ablation | A | FAENet visual-only, ResNet50 | - | - | 77.94 | 67.87 | 86.49 | 76.19 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42\test_dual_metrics.json` |
| QaTa ablation | A | TGFS decoder, ResNet50 + CXR-BERT | - | - | 80.32 | 70.96 | 88.83 | 79.91 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42\test_dual_metrics.json` |
| QaTa ablation | A | TGFS decoder, ResNet50 + lightweight, learned HH | - | - | 81.99 | 72.97 | 89.90 | 81.65 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42\test_dual_metrics.json` |
| QaTa ablation | B | Native, scratch lightweight, both | - | - | 82.13 | 73.07 | 89.45 | 80.91 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_diag0516_qata_simple_native_keep_both_seed42\test_dual_metrics.json` |
| QaTa ablation | B | Empty, scratch lightweight, both | - | - | 76.59 | 66.68 | 85.43 | 74.57 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_paper0516_qata_simple_empty_keep_both_seed42\test_dual_metrics.json` |
| QaTa ablation | B | Shuffled, scratch lightweight, both | - | - | 75.86 | 65.80 | 84.68 | 73.44 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_paper0516_qata_simple_shuffle_keep_both_seed42\test_dual_metrics.json` |
| QaTa ablation | B | Generic, scratch lightweight, both | - | - | 75.99 | 65.75 | 84.16 | 72.65 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_diag0516_qata_simple_generic_keep_both_seed42\test_dual_metrics.json` |
| QaTa ablation | B | Native, scratch CXR-BERT, both | - | - | 81.54 | 72.42 | 89.37 | 80.79 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\screening0506_qata_cxr_frozen_keep_both_seed42\test_dual_metrics.json` |
| QaTa ablation | B | Empty, scratch CXR-BERT, both | - | - | 76.41 | 66.48 | 84.76 | 73.55 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42\test_dual_metrics.json` |
| QaTa ablation | B | Shuffled, scratch CXR-BERT, both | - | - | 76.42 | 66.24 | 84.81 | 73.63 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42\test_dual_metrics.json` |
| QaTa ablation | C | Decoder-only, scratch lightweight | - | - | 81.64 | 72.44 | 89.48 | 80.97 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_paper0516_qata_simple_native_keep_decoder_seed42\test_dual_metrics.json` |
| QaTa ablation | C | Encoder-decoder, scratch lightweight | - | - | 82.13 | 73.07 | 89.45 | 80.91 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_diag0516_qata_simple_native_keep_both_seed42\test_dual_metrics.json` |
| QaTa ablation | C | Decoder-only, scratch CXR-BERT | - | - | 81.27 | 72.01 | 88.96 | 80.11 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\screening0506_qata_cxr_frozen_keep_decoder_seed42\test_dual_metrics.json` |
| QaTa ablation | C | Encoder-decoder, scratch CXR-BERT | - | - | 81.54 | 72.42 | 89.37 | 80.79 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\screening0506_qata_cxr_frozen_keep_both_seed42\test_dual_metrics.json` |

## Table 5

| Method | Dataset/group | Variant/prompt | Reported Dice | Reported IoU | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Provenance/artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Frequency ablation | A | Keep HH | - | - | 82.93 | 74.23 | 90.29 | 82.30 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper_missing0715_qata_resnet50_simple_native_keep_decoder_seed42\test_dual_metrics.json` |
| Frequency ablation | A | Zero HH | - | - | 82.09 | 73.12 | 90.03 | 81.87 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42\test_dual_metrics.json` |
| Frequency ablation | A | Learned HH retention | - | - | 81.99 | 72.97 | 89.90 | 81.65 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42\test_dual_metrics.json` |
| Frequency ablation | B | Full LL/LH/HL/HH | - | - | 82.93 | 74.23 | 90.29 | 82.30 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper_missing0715_qata_resnet50_simple_native_keep_decoder_seed42\test_dual_metrics.json` |
| Frequency ablation | B | w/o LL | - | - | 83.56 | 74.86 | 90.68 | 82.95 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper_missing0715_qata_resnet50_simple_native_drop_ll_decoder_seed42\test_dual_metrics.json` |
| Frequency ablation | B | w/o LH | - | - | 83.54 | 74.90 | 90.69 | 82.97 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper_missing0715_qata_resnet50_simple_native_drop_lh_decoder_seed42\test_dual_metrics.json` |
| Frequency ablation | B | w/o HL | - | - | 83.76 | 75.21 | 90.90 | 83.32 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper_missing0715_qata_resnet50_simple_native_drop_hl_decoder_seed42\test_dual_metrics.json` |
| Frequency ablation | B | w/o HH | - | - | 83.25 | 74.53 | 90.27 | 82.27 | local_full_test; `D:\Documents\LMIS\Text-FAENet\runs\paper_missing0715_qata_resnet50_simple_native_drop_hh_decoder_seed42\test_dual_metrics.json` |

## Missing local dual metrics

| Table | Method | Dataset/group | Variant/prompt |
|---:|---|---|---|

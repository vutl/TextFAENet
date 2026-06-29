# Fair External Comparison Protocol - 2026-06-29

This note supersedes the earlier low-score FMISeg legacy evaluator output.

## What Counts As Fair

- QaTa paper comparison must use per-image Dice/IoU as the main metric.
- Global Dice/IoU can be reported as auxiliary numbers only.
- FMISeg is valid only if evaluated through the current official `FMISeg/evaluate.py` / `net.creratemodel.CreateModel` path.
- The old `fmiseg_qata_legacy_dual_metrics_20260628.*` values are invalid for paper comparison because they produce about `0.51` Dice, while official FMISeg evaluation reports about `0.91` global Dice and `0.83` global IoU on QaTa.
- MedCLIP-SAMv2 should be evaluated on Brain/Breast primarily with our structured prompts. MedCLIP-native prompts are allowed only as a secondary prompt-protocol ablation.

## Current Valid QaTa Anchors

| Model | Source | Per-image Dice | Global Dice | Per-image IoU | Global IoU |
|---|---|---:|---:|---:|---:|
| Ours best | `runs/qata_paper0516_qata_simple_native_zero_both_seed42` | 0.826091 | 0.900397 | 0.737495 | 0.818839 |
| Ours intended ResNet+CXR | `runs/qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` | 0.803207 | 0.888346 | 0.709589 | 0.799121 |
| FMISeg official | `D:\Documents\LMIS\FMISeg\eval_output_qatacov19v2.log` | pending official per-image export | 0.909438 | pending official per-image export | 0.833917 |

## Required Exports For Per-Sample Winner Figures

Run this using the FMISeg environment so the exported per-image CSV matches the official aggregate:

```powershell
cd D:\Documents\LMIS\Text-FAENet
D:\anaconda3\envs\fmiseg\python.exe -u scripts\evaluate_fmiseg_official_qata_per_image.py `
  --fmiseg-root D:\Documents\LMIS\FMISeg `
  --config config\train.yaml `
  --checkpoint save_model\last-v1.ckpt `
  --batch-size 8 `
  --num-workers 0 `
  --device cuda
```

For MedCLIP-SAMv2 mask folders, evaluate only folders generated with the target prompt protocol:

```powershell
D:\anaconda3\python.exe -u scripts\evaluate_mask_folder_per_image.py `
  --pred-dir <MEDCLIP_SAMV2_PRED_MASK_DIR> `
  --gt-dir <TEXT_FAENET_GT_MASK_DIR> `
  --output-csv external_metrics\<dataset>_<prompt_protocol>_medclipsamv2\test_per_image_metrics.csv
```

## Figure Selection Rule

For qualitative figures, select cases only from a comparison table containing:

- Ours best
- Ours intended/main model
- FMISeg official
- MedCLIP-SAMv2 with our prompt, if available
- MedCLIP-SAMv2 native prompt, optional supplementary only

Do not use internal prompt-ablation winner tables as the main cross-model comparison.

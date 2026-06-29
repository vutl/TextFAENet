# Brain/Breast Prompt-Protocol Results - 2026-06-26

All four `paper0623_*` Text-FAENet v3 runs are completed.

Protocol:

- Model preset: `v3_resnet50_cxr`
- Visual encoder: ImageNet-pretrained ResNet50
- Text encoder: frozen CXR-BERT
- TGFS version: v3
- Fusion: encoder-decoder (`fusion_mode=both`)
- Frequency prior: learned HH
- Checkpoint selection: best validation Dice
- Test threshold: selected on validation split
- Paper metric: per-image mean Dice/IoU

## Main Results

| Run | Dataset | Prompt protocol | Best epoch | Threshold | Per-image Dice | Per-image IoU | Global Dice | Global IoU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `paper0623_brain_structured_v3resnet50cxr_both_seed42` | Brain MRI | Structured prompt | 39 | 0.35 | 83.85 | 74.67 | 83.88 | 72.23 |
| `paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42` | Brain MRI | MedCLIP-style prompt | 40 | 0.55 | 84.15 | 75.51 | 84.54 | 73.21 |
| `paper0623_breast_structured_v3resnet50cxr_both_seed42` | Breast US | Structured prompt | 14 | 0.55 | 85.18 | 76.54 | 87.82 | 78.29 |
| `paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42` | Breast US | MedCLIP-style prompt | 15 | 0.35 | 80.23 | 71.79 | 80.51 | 67.38 |

## JSON Values

| Run | Loss | Pred pos ratio | GT pos ratio | Test images |
|---|---:|---:|---:|---:|
| `paper0623_brain_structured_v3resnet50cxr_both_seed42` | 0.181266 | 0.015867 | 0.014781 | 600 |
| `paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42` | 0.178838 | 0.014946 | 0.014781 | 600 |
| `paper0623_breast_structured_v3resnet50cxr_both_seed42` | 0.174350 | 0.038956 | 0.039304 | 113 |
| `paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42` | 0.225069 | 0.043824 | 0.039304 | 113 |

## Interpretation

- Brain MRI: MedCLIP-style prompts are slightly better than our structured prompts.
  - Delta: `+0.30` Dice and `+0.84` IoU in per-image metrics.
- Breast US: our structured prompts are much better than MedCLIP-style prompts.
  - Delta: `+4.95` Dice and `+4.75` IoU in per-image metrics.
- The Breast structured run is currently the strongest Brain/Breast run:
  - `85.18` per-image Dice
  - `76.54` per-image IoU
- The old draft value for Brain structured (`85.87 / 77.32`) is not supported by
  these completed local runs. The traceable full-run value is now
  `83.85 / 74.67`.

## Recommended Paper Table Updates

### Main Four-Dataset Table

Use these values for `\method{} (ours)` if reporting the `v3_resnet50_cxr_both`
Brain/Breast results:

| Dataset | Dice | IoU |
|---|---:|---:|
| Brain MRI | 83.85 | 74.67 |
| Breast US | 85.18 | 76.54 |

### Prompt-Template Transfer Table

Use:

| Model | Prompt format | Brain Dice | Brain IoU | Breast Dice | Breast IoU |
|---|---|---:|---:|---:|---:|
| `\method{}` | MedCLIP-style prompt | 84.15 | 75.51 | 80.23 | 71.79 |
| `\method{}` | Structured prompt | 83.85 | 74.67 | 85.18 | 76.54 |

## Source Files

- `runs/paper0623_brain_structured_v3resnet50cxr_both_seed42/final_test.json`
- `runs/paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42/final_test.json`
- `runs/paper0623_breast_structured_v3resnet50cxr_both_seed42/final_test.json`
- `runs/paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42/final_test.json`


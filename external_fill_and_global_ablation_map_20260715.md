# External table fills and global QaTa ablation map

This note separates results that can be inserted into the paper from diagnostic
artifacts whose prompt or training provenance is not sufficient for a fair
paper row.

## 1. External methods: values recoverable now

### MedCLIP-SAMv2 original/native prompt protocol

The archived zero-shot outputs contain all 600 Brain test masks and all 113
Breast test masks. Their generating scripts explicitly use:

- Brain: `saliency_maps/text_prompts/brain_tumors_testing.json`.
- Breast: `saliency_maps/text_prompts/breast_tumors_testing.json`.

These are MedCLIP-SAMv2's own descriptive prompts, not our structured CSV
prompts. Re-evaluation against the current, filename-matched GT folders gives:

| Dataset | Images | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Paper use |
|---|---:|---:|---:|---:|---:|---|
| Brain MRI | 600 | 59.41 | 48.94 | 51.35 | 34.54 | Local reproduction of the MedCLIP-native protocol; it does not reproduce the published `80.03/70.71` row |
| Breast US | 113 | 83.10 | 73.36 | 81.71 | 69.07 | Valid local MedCLIP-native result |

Exact artifacts:

- `external_metrics/medclipsamv2_brain_native/summary.json`
- `external_metrics/medclipsamv2_brain_native/test_per_image_metrics.csv`
- `external_metrics/medclipsamv2_breast_native/summary.json`
- `external_metrics/medclipsamv2_breast_native/test_per_image_metrics.csv`

The manuscript's MedCLIP-SAMv2 original-prompt values `80.03/70.71` (Brain)
and `78.87/69.08` (Breast) are source-reported values. Do not silently replace
them with the local reproductions; label one set as source-reported and the
other as local reproduction if both are shown.

### MedCLIP-SAMv2 MosMed archive

There are 1,800 filename-matched masks in `sam_outputs/lung_CT/test_masks`.
Their measured values are:

| Images | Per-image Dice | Per-image IoU | Global Dice | Global IoU |
|---:|---:|---:|---:|---:|
| 1,800 | 30.09 | 21.02 | 28.69 | 16.74 |

Exact artifacts:

- `external_metrics/medclipsamv2_mosmed_generic/summary.json`
- `external_metrics/medclipsamv2_mosmed_generic/test_per_image_metrics.csv`

This result is **diagnostic only**. The old lung-CT zero-shot script requests
one interactive text prompt and does not record that prompt. Therefore the
mask folder cannot be proven to use either our structured prompts or a fixed
MedCLIP `lung_CT_P2` prompt. It must not fill the main-table MosMed cell under a
named prompt protocol. A new explicit-prompt inference run is required.

### MedCLIP-SAMv2 QaTa

No full-test MedCLIP-SAMv2 prediction folder/result is present. The existing
`external_metrics/medclipsamv2_qata_our_prompt_highgap` artifact contains only
five selected cases and cannot fill a test-set table cell.

### FMISeg

The only valid locally trained Brain/Breast FMISeg result is Breast with our
structured CSV prompts:

| Prompt protocol | Dataset | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Paper use |
|---|---|---:|---:|---:|---:|---|
| Our structured prompts | Breast US | 84.23 | 75.35 | 84.87 | 73.71 | Can fill Table 3 |

The run uses the Text-FAENet Breast train/val/test roots and prepared CSVs,
selects on `val_dice_image`, and evaluates the Breast test split. Exact source:
`D:/Documents/LMIS/FMISeg/checkpoints/breast_tumors_text/final_test_metrics.json`.

The file `brain_tumors_eval_from_breast_text/final_test_metrics.json` is not a
valid Brain row: it evaluates the Breast-trained model on Brain. No
Brain-trained FMISeg result is present. No FMISeg run using the original
MedCLIP-style Brain/Breast prompts is present.

## 2. Which currently blank external cells can be filled?

| Table/row | Brain | Breast | QaTa | MosMed |
|---|---|---|---|---|
| Table 2, MedCLIP-SAMv2 | Existing source-reported value; local native reproduction also available | Existing source-reported value; local native reproduction also available | Missing: full inference required | Missing: old full masks have unknown prompt provenance; rerun required |
| Table 2, FMISeg | Missing: Brain-trained run required | Already reported | Already reported; official local dual metrics also exist | Already reported |
| Table 3, MedCLIP with our structured prompts | Missing: structured-prompt inference required | Missing: structured-prompt inference required | n/a | n/a |
| Table 3, FMISeg with MedCLIP-style prompts | Missing: train/eval required | Missing: train/eval required | n/a | n/a |
| Table 3, FMISeg with our structured prompts | Missing: Brain train/eval required | **84.23/75.35 per-image** | n/a | n/a |

Thus the only previously blank external cell pair that can be filled now with
clean provenance is FMISeg + structured prompt + Breast: `84.23/75.35`
(per-image Dice/IoU).

## 3. Converting QaTa ablations to global Dice/IoU

Global metrics cannot be derived from mean per-image Dice/IoU alone. They are
available only where a checkpoint was re-evaluated or pixel-count/prediction
artifacts were retained.

### Table 4: every row is convertible

| Group | Variant | Global Dice | Global IoU |
|---|---|---:|---:|
| A | FAENet visual-only, scratch | 86.12 | 75.62 |
| A | FAENet visual-only, ResNet50 | 86.49 | 76.19 |
| A | TGFS decoder, ResNet50 + CXR-BERT | 88.83 | 79.91 |
| A | TGFS decoder, ResNet50 + lightweight text, learned HH | 89.90 | 81.65 |
| B | Native, scratch lightweight, both | 89.45 | 80.91 |
| B | Empty, scratch lightweight, both | 85.43 | 74.57 |
| B | Shuffled, scratch lightweight, both | 84.68 | 73.44 |
| B | Generic, scratch lightweight, both | 84.16 | 72.65 |
| B | Native, scratch CXR-BERT, both | 89.37 | 80.79 |
| B | Empty, scratch CXR-BERT, both | 84.76 | 73.55 |
| B | Shuffled, scratch CXR-BERT, both | 84.81 | 73.63 |
| C | Decoder-only, scratch lightweight | 89.48 | 80.97 |
| C | Encoder-decoder, scratch lightweight | 89.45 | 80.91 |
| C | Decoder-only, scratch CXR-BERT | 88.96 | 80.11 |
| C | Encoder-decoder, scratch CXR-BERT | 89.37 | 80.79 |

These values come from `test_dual_metrics.json` or recovered full-test
inference indexed by `available_dual_metrics_20260628.json`. Notice that global
aggregation changes some rankings: for example, lightweight decoder-only is
slightly above encoder-decoder globally even though the latter is higher under
mean per-image metrics. Bold formatting must be recomputed after conversion.

### Table 5: partially convertible

| Block | Variant | Global Dice | Global IoU | Status |
|---|---|---:|---:|---|
| A | Keep HH | 89.58 | 81.12 | Available |
| A | Zero HH | 90.03 | 81.87 | Available |
| A | Learned HH retention | 89.90 | 81.65 | Available |
| B | Full LL/LH/HL/HH | 89.58 | 81.12 | Available |
| B | w/o LL | - | - | Checkpoint and predictions absent |
| B | w/o LH | - | - | Checkpoint and predictions absent |
| B | w/o HL | - | - | Checkpoint and predictions absent |
| B | w/o HH | - | - | Checkpoint and predictions absent |

The four drop-one-band global rows require retraining (or recovery of their
original checkpoints/prediction masks). Their existing per-image results cannot
be algebraically converted to global metrics.

## 4. Recommended consistent presentation

- If Table 2 keeps source-reported/global pulmonary values, converting Table 4
  to global is possible, but Table 5 cannot be fully converted without four
  reruns.
- The currently complete, reproducible choice is to keep Tables 4 and 5 as
  mean per-image ablations and state that aggregation explicitly.
- Do not mix MedCLIP source-reported Brain/Breast values, local native-prompt
  reproductions, and structured-prompt reruns without labeling each protocol.

# Paper Experiment Gap Analysis - 2026-06-23

This note maps the current `experiments_balanced.tex` story to local code,
checkpoints, prompt files, and missing runs.

## Current Paper Narrative

The experiments section is organized around three claims:

1. Text-guided frequency decoding improves a FAENet-style visual frequency
   backbone.
2. The model uses prompt semantics, not just the presence of text tokens.
3. The structured prompt protocol transfers beyond pulmonary infection to
   Brain MRI and Breast US.

The strongest supported part today is the QaTa controlled ablation suite. The
weakest part today is the Brain/Breast prompt-template transfer table, because
several cells are still placeholders or are not backed by local full training
runs.

## Local Data/Prompt Status

### Text-FAENet structured prompts

Structured prompt datasets exist with train/val/test CSVs:

- `datasets/brain_tumors`
- `datasets/breast_tumors`

These CSVs use our structured laterality/count/location style prompts.

### MedCLIP-SAMv2 prompt datasets

MedCLIP-style prompt datasets exist at:

- `D:\Documents\LMIS\MedCLIP-SAMv2\data\brain_tumors`
- `D:\Documents\LMIS\MedCLIP-SAMv2\data\breast_tumors`

The split filenames match the Text-FAENet Brain/Breast splits. The main
difference is the `Description` text:

- Text-FAENet: structured location/count/laterality prompts.
- MedCLIP-SAMv2: generic or class/descriptive CLIP-style prompts.

This makes Brain/Breast suitable for prompt-template transfer experiments.

## Results That Are Locally Backed

### QaTa-COV19

The QaTa controlled ablations are locally backed by completed runs. The most
important rows are:

| Role | Run | Dice | IoU | Notes |
|---|---:|---:|---:|---|
| Best scratch/simple text model | `runs/qata_paper0516_qata_simple_native_learned_both_seed42` | 82.72 | 73.91 | Best QaTa Dice among current scratch/simple TGFS runs. |
| Main ResNet50+CXR-BERT diagnostic | `runs/qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` | 80.30 | 70.93 | Architecture-aligned ResNet50 + CXR-BERT + decoder TGFS row. |
| ResNet50 lightweight best | `runs/qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42` | 82.24 | 73.43 | Best ResNet50 lightweight-text decoder row. |
| Scratch FAENet visual-only | `runs/qata_paper0516_qata_faenet_visual_clean_seed42` | 77.57 | 67.65 | Clean no-text baseline. |
| ResNet50 FAENet visual-only | `runs/qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` | 77.90 | 67.81 | Clean ResNet visual-only baseline. |

The frequency-drop ablation rows are also locally backed by the
`qata_resnet0523_*` and `qata_resnet0523_fp32fix_*` runs.

### FMISeg Breast Structured Prompt

FMISeg has a completed Breast structured-prompt run:

`D:\Documents\LMIS\FMISeg\checkpoints\breast_tumors_text\final_test_metrics.json`

Metrics:

| Metric type | Dice | IoU |
|---|---:|---:|
| Per-image | 84.23 | 75.35 |
| Global | 84.87 | 73.71 |

This can fill the FMISeg Breast row if the paper reports per-image metrics.

### MedCLIP-SAMv2 Local Output Caveat

Local MedCLIP-SAMv2 output masks exist, but they do not reproduce the paper-style
Brain number currently written in `experiments_balanced.tex`.

Measured from local `sam_outputs/*/test_masks`:

| Dataset | Per-image Dice | Per-image IoU | Global Dice | Global IoU |
|---|---:|---:|---:|---:|
| Brain | 59.41 | 48.94 | 51.35 | 34.54 |
| Breast | 83.10 | 73.36 | 81.71 | 69.07 |

Do not mix these local mask numbers with a published MedCLIP-SAMv2 table row
unless the evaluation protocol is explicitly explained.

## Results That Are Not Yet Safely Backed

These entries appear in the current paper draft but need verification or fresh
full runs before being treated as final:

| Paper entry | Current status | Required action |
|---|---|---|
| Ours Brain 85.87 / 77.32 | Not found in local `runs` as a completed full v3 result. | Train Text-FAENet v3 on Brain structured prompts. |
| Ours Breast 83.86 / 76.15 | Not found in local `runs` as a completed full v3 result. | Train Text-FAENet v3 on Breast structured prompts. |
| Ours Brain/Breast with MedCLIP-style prompts | Missing. | Train Text-FAENet v3 using MedCLIP-SAMv2 CSV prompt files. |
| FMISeg Brain structured prompt | Missing full brain-trained run. | Train/evaluate FMISeg on Brain structured prompts. |
| FMISeg original MedCLIP-style prompt rows | Missing. | Optional: train FMISeg using MedCLIP-style prompt CSVs. |
| MedCLIP-SAMv2 with our structured prompts | Missing. | Optional/hard: rerun MedCLIP-SAMv2 prompt generation with structured prompts. |
| Ours MosMed 80.10 / 66.68 | Not backed by the visible Text-FAENet MosMed run inventory. | Locate the run file or demote this row until rerun. |

## Code Changes Made

### `scripts/train_brain_tumors.py`

The trainer now supports external CSV prompt files:

- `--train-csv-path`
- `--val-csv-path`
- `--test-csv-path`

This is required to train Text-FAENet on the same Brain/Breast images while
switching between our structured prompts and MedCLIP-SAMv2-style prompts.

The foreground-statistics path was also wired to use `--train-csv-path`, so
external CSV runs no longer crash before training.

Smoke test passed with MedCLIP Brain CSVs:

- Train/eval on 2 samples.
- Outputs both per-image and global Dice/IoU.
- Confirms external CSV loading, foreground stats, forward pass, checkpointing,
  and final metric writing.

### `scripts/run_brain_breast_prompt_protocol_suite.py`

New sequential runner for the four highest-priority missing Text-FAENet runs:

1. Brain structured prompts.
2. Breast structured prompts.
3. Brain MedCLIP-style prompts.
4. Breast MedCLIP-style prompts.

The runner uses the current `v3_resnet50_cxr` preset, which means:

- ResNet50 image encoder.
- ImageNet pretrained encoder.
- CXR-BERT text encoder.
- Frozen CXR-BERT backbone.
- TGFS v3.
- `fusion_mode=both`.
- `hh_drop_mode=learned`.
- Validation-threshold sweep.
- Final output includes per-image and global Dice/IoU.

The run names intentionally include `v3resnet50cxr_both` to avoid confusion with
the QaTa decoder-only ResNet/CXR ablation.

## Command To Run The Missing Text-FAENet Brain/Breast Suite

Run from `D:\Documents\LMIS\Text-FAENet`:

```powershell
$env:HF_HOME=(Resolve-Path .hf_cache).Path
$env:TORCH_HOME=(New-Item -ItemType Directory -Force .torch_cache).FullName
D:\anaconda3\python.exe -u scripts\run_brain_breast_prompt_protocol_suite.py --run-prefix paper0623 --continue-on-fail --resume-existing --skip-completed --num-workers 2
```

Expected run folders:

- `runs/paper0623_brain_structured_v3resnet50cxr_both_seed42`
- `runs/paper0623_breast_structured_v3resnet50cxr_both_seed42`
- `runs/paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42`
- `runs/paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42`

If you only want the structured-prompt main benchmark first:

```powershell
D:\anaconda3\python.exe -u scripts\run_brain_breast_prompt_protocol_suite.py --run-prefix paper0623 --prompt-protocols structured --continue-on-fail --resume-existing --skip-completed --num-workers 2
```

If you only want the prompt-template control runs:

```powershell
D:\anaconda3\python.exe -u scripts\run_brain_breast_prompt_protocol_suite.py --run-prefix paper0623 --prompt-protocols medclip --continue-on-fail --resume-existing --skip-completed --num-workers 2
```

## Figure Readiness

### Ready now

1. QaTa ablation bar/table figure.
2. TGFS schematic.
3. QaTa prompt intervention qualitative figure, using existing QaTa
   checkpoints.

### Partially ready

1. Cross-dataset qualitative segmentation figure.
   - QaTa is ready.
   - Breast can use existing FMISeg masks and local MedCLIP masks.
   - Brain should wait for a real Text-FAENet v3 checkpoint and ideally a
     brain-trained FMISeg checkpoint.

2. Brain/Breast prompt-template qualitative figure.
   - Needs Text-FAENet structured vs MedCLIP-prompt runs.
   - MedCLIP local masks exist but Brain quality does not match the paper row,
     so use with a protocol caveat.

### Not ready

1. FMISeg Brain structured-prompt figure/table row.
2. FMISeg MedCLIP-style prompt rows.
3. MedCLIP-SAMv2 with our structured prompts.

## Recommended Next Experiment Order

Priority 1:

1. Run Text-FAENet Brain/Breast structured prompts.
2. Run Text-FAENet Brain/Breast MedCLIP-style prompts.

These four runs directly support the main Brain/Breast table and the
prompt-template transfer table.

Priority 2:

3. Train FMISeg Brain on our structured prompts.
4. Optionally train FMISeg Brain/Breast on MedCLIP-style prompts.

Priority 3:

5. Rerun/evaluate MedCLIP-SAMv2 with our structured prompts, if the pipeline can
   be adapted cleanly.
6. Locate or rerun the MosMed result behind the current 80.10 / 66.68 paper row.

## FMISeg Brain Structured Baseline

FMISeg currently has:

- Completed Breast structured-prompt run: `FMISeg/checkpoints/breast_tumors_text`.
- A Brain-from-Breast evaluation config: `FMISeg/config/eval_brain_tumors_from_breast_text.yaml`.
- No full Brain-trained FMISeg config/checkpoint yet.

To fill the FMISeg Brain row fairly, create a FMISeg config analogous to
`train_breast_tumors_text.yaml`, but with Brain paths:

```yaml
TRAIN:
  dataset_name: brain_tumors
  wavelet_type: haar
  train_batch_size: 8
  lr: 0.0003
  valid_batch_size: 4
  image_size: [224,224]
  min_epochs: 20
  max_epochs: 120
  patience: 20
  device: 0
  model_save_path: ./checkpoints/brain_tumors_text
  model_save_filename: fmis-brain-text-{epoch:02d}-{val_dice_image:.4f}
  checkpoint_monitor: val_dice_image
  checkpoint_mode: max
  experiment_name: brain_tumors_text
  log_root: ./run_logs
  seed: 42
  matmul_precision: medium
  log_every_n_steps: 10
  accumulate_grad_batches: 4
  auto_pos_weight: true
  min_bce_pos_weight: 1.0
  max_bce_pos_weight: 256.0
  run_test_after_fit: true
  auto_prompt_from_mask: false
  test_metrics_filename: final_test_metrics.json

MODEL:
  bert_type: ./lib/BiomedVLP-CXR-BERT-specialized
  vision_type: ./lib/convnext-tiny-224
  project_dim: 768

DATA:
  train_csv_path: "D:\\Documents\\LMIS\\FMISeg\\data\\brain_tumors_prepared\\train.csv"
  train_root_path: "D:\\Documents\\LMIS\\Text-FAENet\\datasets\\brain_tumors"
  val_csv_path: "D:\\Documents\\LMIS\\FMISeg\\data\\brain_tumors_prepared\\val.csv"
  val_root_path: "D:\\Documents\\LMIS\\Text-FAENet\\datasets\\brain_tumors"
  test_csv_path: "D:\\Documents\\LMIS\\FMISeg\\data\\brain_tumors_prepared\\test.csv"
  test_root_path: "D:\\Documents\\LMIS\\Text-FAENet\\datasets\\brain_tumors"
```

Then run from `D:\Documents\LMIS\FMISeg`:

```powershell
D:\anaconda3\python.exe -u train.py --config config\train_brain_tumors_text.yaml
```

The important metric fields are `test_dice_image` and `test_iou_image` in
`final_test_metrics.json`, because the paper protocol reports per-image mean
Dice/IoU rather than global Dice/IoU.

## Main Risk

The current draft already contains strong Brain/Breast and MosMed numbers, but
not all of them are traceable to local completed run folders. Before final
submission, every table cell should either:

1. Point to a completed local run/result file.
2. Be marked as a published baseline under a clearly different protocol.
3. Stay as `--`.

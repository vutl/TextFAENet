# Text-FAENet Current Architecture and QaTa Ablation Report

Date: 2026-05-16

This document summarizes the current code state, the architecture we are actually running, the ablation space that has been opened, and the detailed results for the QaTa runs listed by the user.

## 1. Sources Inspected

Code inspected:

- `src/models/faenet.py`
- `src/models/lfaenet_tgfs.py`
- `src/models/lfaenet_tgfs_v2.py`
- `src/modules/blocks.py`
- `src/modules/wavelet.py`
- `src/data/qata_cov19.py`
- `src/data/mosmed_text_csv.py`
- `scripts/train_qata.py`
- `scripts/train_mosmed_text.py`
- `scripts/run_qata_mosmed_ablation_plan.py`
- `scripts/run_qata_diag_ablation.py`

Design and research notes inspected:

- `deep-research-report.md`
- `lfaenet_tgfs_detailed_design.md`
- `FAENet_chi_tiet.md`

Run artifacts inspected:

- `config.json`
- `history.json`
- `epoch_log.txt`
- `final_test.json`
- `final_test.txt`

The existing Vietnamese design notes are partly mojibake on disk, but the technical content is still recoverable: FAENet is the visual frequency baseline; LFAENet-TGFS is the decoder-side text-guided frequency-selection extension; the deep research report argues for ablation first, then loosening hard frequency priors, then stronger fusion/text-encoder tests, and only after that a larger FMISeg-style raw HF/LF hybrid.

## 2. Current Architecture

### 2.1 FAENet Baseline

The visual-only baseline in `src/models/faenet.py` follows the FAENet idea:

```text
image
  -> stem conv
  -> encoder stage 1: ConvBlock + FreqA
  -> encoder stage 2: ConvBlock + FreqA
  -> encoder stage 3: ConvBlock + FreqA
  -> encoder stage 4: ConvBlock + FreqA
  -> bottleneck: ConvBlock + FreqA
  -> decoder stage 4: upsample + skip concat + ConvBlock + FreqA
  -> decoder stage 3: upsample + skip concat + ConvBlock + FreqA
  -> decoder stage 2: upsample + skip concat + ConvBlock + FreqA
  -> decoder stage 1: upsample + skip concat + ConvBlock + FreqA
  -> final ConvBlock
  -> 1x1 segmentation head
```

Default channels are `(64, 128, 256, 512)` and bottleneck is `768`. The current implementation uses a U-Net-like realization of FAENet rather than a literal ResNet-50 port.

The key frequency module is `FreqA` from `src/modules/blocks.py`:

```text
feature x
  -> Haar DWT
  -> LL, LH, HL, HH
  -> per-band ICCA channel attention
  -> CCCA cross-band channel interaction
  -> concatenate four bands
  -> FrequencyMixer
  -> split bands
  -> inverse Haar DWT
  -> residual add
```

This module is purely visual. It does not see text. It uses feature-level DWT, not raw-image LF/HF dual branches.

### 2.2 LFAENet-TGFS v1

`src/models/lfaenet_tgfs.py` is the first language-guided version.

The main design is:

- Visual encoder remains FAENet-like.
- Text encoder produces a pooled text vector.
- Decoder stages replace plain `FreqA` with `TGFSBlock`.
- Text gates the four DWT sub-bands `LL/LH/HL/HH`.

The v1 limitation is that text is only a global pooled vector. It can channel-gate frequency bands, but it has no token-level visual grounding. It also injects text only in decoder-style TGFS, not in a configurable encoder/decoder/both way.

### 2.3 LFAENet-TGFS v2

The current main model is `LFAENetTGFSv2` in `src/models/lfaenet_tgfs_v2.py`.

High-level flow for a `224 x 224` grayscale image:

```text
image: B x 1 x 224 x 224
  -> stem: B x 64 x 224 x 224
  -> enc1: B x 64 x 224 x 224
  -> pool
  -> enc2: B x 128 x 112 x 112
  -> pool
  -> enc3: B x 256 x 56 x 56
  -> pool
  -> enc4: B x 512 x 28 x 28
  -> pool
  -> bottleneck: B x 768 x 14 x 14
  -> dec4 with skip enc4: B x 512 x 28 x 28
  -> dec3 with skip enc3: B x 256 x 56 x 56
  -> dec2 with skip enc2: B x 128 x 112 x 112
  -> dec1 with skip enc1: B x 64 x 224 x 224
  -> final refine
  -> logits: B x 1 x 224 x 224
```

The text path is encoded first and then passed to all stages:

```text
token_ids, attention_mask
  -> text encoder
  -> token embeddings: B x L x 256
  -> pooled embedding: B x 256
  -> encoder/decoder TGFS blocks depending on fusion_mode
```

Supported text encoders:

- `simple`: embedding + GRU + linear projection; pooled by attention mask.
- `biomedvlp-cxr-bert`: local `BiomedVLP-CXR-BERT-specialized` loaded with `local_files_only=True`; token hidden states are projected to `text_dim=256` and mean-pooled.

CXR-BERT adaptation options:

- `--freeze-text-backbone`: freezes the BERT backbone by default.
- `--unfreeze-last-n`: unfreezes the last N BERT encoder blocks.
- `--lora-r`: injects LoRA into attention `query` and `value` linear layers.

Fusion options:

- `fusion_mode=decoder`: encoder is visual `ConvBlock + FreqA`; decoder uses `TGFSDecoderStageV2`.
- `fusion_mode=encoder`: encoder and bottleneck use `EncoderStageText`; decoder uses visual `PlainDecoderStageV2`.
- `fusion_mode=both`: encoder, bottleneck, and decoder all use TGFS-style text-conditioned stages.

Important current point: this is still not the FMISeg raw HF/LF dual-branch architecture. Current DWT is applied to internal feature maps inside `FreqA` or `TGFSBlockV2`, not to the input image as separate LF/HF image streams.

## 3. TGFSBlockV2 Mechanics

`TGFSBlockV2` is the core contribution currently implemented.

For a feature map `x` with channel count `C`:

```text
x
  -> local 3x3 conv stack
  -> Haar DWT
  -> LL, LH, HL, HH
  -> per-band ICCA
  -> pooled text -> MLP -> four channel gates
  -> LL *= a_LL
  -> LH *= a_LH * lh_hl_scale
  -> HL *= a_HL * lh_hl_scale
  -> HH *= a_HH, then zero/keep/learned-scale depending on hh_drop_mode
  -> CCCA cross-band interaction
  -> concatenate bands
  -> token-level grounding map from visual queries and text token keys/values
  -> spatial mask ** spatial_sharpen_power
  -> band aggregate *= spatial mask
  -> grouped mixer and optional spatial self-attention
  -> split bands
  -> inverse Haar DWT
  -> text-conditioned branch scale
  -> output
```

Frequency prior controls now exposed:

- `--hh-drop-mode zero`: hard-zero HH after text gating.
- `--hh-drop-mode keep`: keep HH after text gating.
- `--hh-drop-mode learned`: multiply HH by a learned sigmoid scale.
- `--low-level-hf-scale`: fixed scalar for LH/HL at shallow stages, default `0.6`.
- `--spatial-sharpen-power`: fixed exponent for token grounding mask, default `2.0`.

This partially implements the deep-research recommendation. HH can now be zero/keep/learned, but LH/HL scale and spatial-sharpen are still fixed hyperparameters rather than learned stage-wise parameters.

Debug outputs:

- `a_ll_mean`
- `a_lh_mean`
- `a_hl_mean`
- `a_hh_mean`
- `lh_hl_scale`
- `hh_scale`
- `spatial_mask`

These can be exported by `--save-debug-vis`. The export currently focuses on decoder stages `dec4` to `dec1`.

## 4. Data and Training Protocol

### 4.1 QaTa Dataset

`src/data/qata_cov19.py` reads QaTa-COV19-v2 as:

- `Train/Images`
- `Train/GTs`
- `Test/Images`
- `Test/GTs`
- `prompt/train.csv`
- `prompt/test.csv`

Each sample returns:

- grayscale image normalized to `[0, 1]`
- binary mask thresholded at `>127`
- prompt text from the CSV
- mask name

Current `train_qata.py` default protocol:

- Uses train split for training plus proxy validation.
- Uses test split only for final evaluation.
- Default proxy val is `20%` of train, shuffled with the run seed.
- `--use-test-as-val` exists, but default is `False`.

This is important because several older runs were produced before the current clean ablation protocol and do not record all of the new fields.

### 4.2 MosMed Dataset

`src/data/mosmed_text_csv.py` reads MosMed text CSV format as:

- frames from `datasets/MosMed/frames`
- masks from `datasets/MosMed/masks`
- train CSV: `Train_text_MosMedData+ 1(in).csv`
- val CSV: `Val_text_MosMedData+ 1(in).csv`
- test CSV: `Test_text_MosMedData+(in).csv`

Each row provides `Image` and `text`. The image and mask names are shared.

MosMed training has additional controls:

- `--pos-weight auto`
- foreground statistics logging
- `pred_pos_ratio` and `gt_pos_ratio` metrics
- default loss weighting `0.3 BCE + 0.7 Dice`
- current runner disables AMP on MosMed by default because previous keep/both MosMed runs produced non-finite losses.

This report focuses on QaTa runs because that is the requested run list.

### 4.3 Prompt Modes

Both `train_qata.py` and `train_mosmed_text.py` now support:

| Mode | Behavior |
|---|---|
| `native` | Use dataset prompt unchanged. |
| `canonical` | Lowercase, strip, normalize spaces, ensure trailing period. |
| `generic` | Replace every prompt with `segment the abnormal medical region.` |
| `lesion` | Replace every prompt with `segment the lesion region.` |
| `empty` | Replace prompt with empty string. |
| `shuffle` | Shuffle prompts within batches using a seeded RNG. |

For CXR-BERT, Hugging Face tokenizer is used. For simple encoder, prompts are tokenized by a stable local hash tokenizer over alphanumeric tokens.

### 4.4 Checkpoint and Threshold Selection

Current clean protocol:

- Best checkpoint is selected by validation Dice only.
- Threshold sweep is validation-only.
- QaTa/MosMed thresholds are `0.35, 0.40, 0.45, 0.50, 0.55`.
- Final test is evaluated once with `best.pt` and the selected validation threshold.

Legacy caveat:

- Older runs such as `qata_b4_e50_cxrbert_frozen_v2` and `qata_b4_e50_cxrbert_frozen_v2_rerun` do not record `prompt_mode`, `hh_drop_mode`, `fusion_mode`, or `metric_thresholds`.
- Their final JSON also does not record `best_threshold`.
- They are useful historical references, but cleaner comparisons should rely more on the `screening0506` and `qata_diag0516` runs.

## 5. Implemented Deep-Research Recommendations

Implemented:

- CLI ablation space with `BooleanOptionalAction` for `--use-cxr-bert`, `--freeze-text-backbone`, `--use-amp`, `--use-deep-supervision`, etc.
- `--prompt-mode {native, canonical, generic, lesion, empty, shuffle}`.
- `--hh-drop-mode {zero, keep, learned}`.
- `--fusion-mode {encoder, decoder, both}`.
- `--unfreeze-last-n`.
- `--lora-r`.
- `--grad-accum-steps`.
- `--save-debug-vis`.
- QATA/MosMed sequential runner for screening and confirmatory seeds.
- QATA diagnostic runner focused on prompt, simple encoder, learned HH, and LoRA.

Partially implemented:

- Hard HH prior is now controllable by zero/keep/learned.
- Low-level LH/HL scale is still a fixed scalar.
- Spatial grounding sharpen is still a fixed exponent.

Not yet implemented:

- Raw-image LF/HF dual-branch encoder.
- FFBI-lite bottleneck interaction.
- Hybrid FMISeg-style visual trunk plus TGFS decoder.
- Stage-wise learned LH/HL scale.
- Stage-wise learned or annealed spatial sharpen.
- Boundary metrics such as HD95/NSD.

## 6. Run Result Matrix

All metrics below are from the run folders on disk. `best val` is selected from `history.json`. `test` is from `final_test.json` when available.

| Rank | Run | Main setting | Epochs logged | Best val Dice | Best val thr | Final test Dice | Final test IoU | Status |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `qata_diag0516_qata_simple_native_keep_both_seed42` | simple text, native prompt, HH keep, fusion both | 19 | 0.811177 | 0.55 | 0.821199 | 0.730517 | complete |
| 2 | `qata_b4_e50_cxrbert_frozen_v2` | legacy CXR-BERT frozen, hard HH zero/drop, decoder-style | 50 | 0.816895 | n/a | 0.816895 | 0.725376 | complete, legacy protocol |
| 3 | `screening0506_qata_cxr_frozen_keep_both_seed42` | CXR-BERT frozen, native, HH keep, fusion both | 50 | 0.806679 | 0.35 | 0.815488 | 0.724245 | complete |
| 4 | `screening0506_qata_cxr_frozen_keep_decoder_seed42` | CXR-BERT frozen, native, HH keep, fusion decoder | 50 | 0.803550 | 0.40 | 0.812502 | 0.719764 | complete |
| 5 | `qata_diag0516_qata_cxr_frozen_learned_both_seed42` | CXR-BERT frozen, native, HH learned, fusion both | 15 | 0.803095 | 0.35 | 0.811856 | 0.716754 | complete |
| 6 | `qata_b4_e50_cxrbert_frozen_v2_rerun` | legacy CXR-BERT frozen, hard HH zero/drop, decoder-style rerun | 50 | 0.798982 | n/a | 0.803312 | 0.708446 | complete, legacy protocol |
| 7 | `qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42` | CXR-BERT frozen, shuffled prompt, HH keep, fusion both | 25 | 0.756165 | 0.40 | 0.764517 | 0.662672 | complete |
| 8 | `qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42` | CXR-BERT frozen, empty prompt, HH keep, fusion both | 17 | 0.755809 | 0.45 | 0.764167 | 0.664734 | complete |
| 9 | `qata_diag0516_qata_simple_generic_keep_both_seed42` | simple text, generic prompt, HH keep, fusion both | 15 | 0.749395 | 0.45 | 0.759944 | 0.657492 | complete |
| 10 | `qata_faenet_notext_valclean_e5` | FAENet visual-only, SGD/poly, deep supervision | 28 | 0.648414 | n/a | 0.642565 | 0.515424 | complete, older visual baseline |
| 11 | `qata_faenet_notext_adamw_cosine_e30` | FAENet visual-only, AdamW/cosine, deep supervision | 19 | 0.534319 | n/a | 0.544604 | 0.410539 | complete, older visual baseline |
| n/a | `qata_diag0516_qata_cxr_lora8_keep_both_seed42` | CXR-BERT frozen + LoRA rank 8, native, HH keep, fusion both | 4 | 0.796615 | 0.55 | n/a | n/a | incomplete |

## 7. Per-Run Details

### 7.1 `runs/qata_b4_e50_cxrbert_frozen_v2`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen
- Epochs: 50
- Batch size: 4
- LR: `0.02`
- Weight decay: `1e-4`
- Deep supervision: off
- Legacy `drop_hh_in_decoder=True`
- No recorded `prompt_mode`, `hh_drop_mode`, `fusion_mode`, or threshold sweep

Training behavior:

- Last epoch train Dice: `0.985713`
- Last epoch val Dice: `0.807385`
- Best epoch by val Dice: `26`
- Best val Dice: `0.816895`

Final test:

- Loss: `0.149432`
- IoU: `0.725376`
- Dice: `0.816895`

Interpretation:

This remains a strong historical CXR-BERT/hard-HH baseline. Because it lacks the newer ablation metadata and threshold-sweep record, it should not be treated as perfectly protocol-matched against the `screening0506` and `qata_diag0516` runs.

### 7.2 `runs/qata_b4_e50_cxrbert_frozen_v2_rerun`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen
- Optimizer: SGD
- LR scheduler: poly
- Epochs: 50
- Batch size: 4
- Deep supervision: off
- Legacy `drop_hh_in_decoder=True`
- Val ratio: `0.2`

Training behavior:

- Last epoch train Dice: `0.986906`
- Last epoch val Dice: `0.789352`
- Best epoch by val Dice: `6`
- Best val Dice: `0.798982`

Final test:

- Loss: `0.147879`
- IoU: `0.708446`
- Dice: `0.803312`

Interpretation:

This rerun is much lower than the original `qata_b4` run by about `-1.36` Dice points. That is a warning that single-seed or legacy-protocol numbers should not be overinterpreted. Confirmatory seeds are necessary for any final paper claim.

### 7.3 `runs/screening0506_qata_cxr_frozen_keep_decoder_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen
- Prompt mode: `native`
- HH mode: `keep`
- Fusion mode: `decoder`
- Epochs: 50
- Batch size: 4
- Optimizer: SGD
- LR scheduler: poly
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Training behavior:

- Last epoch train Dice: `0.988742`
- Last epoch val Dice: `0.794166`
- Best epoch by val Dice: `24`
- Best val Dice: `0.803550`
- Best val threshold: `0.40`

Final test:

- Loss: `0.150958`
- IoU: `0.719764`
- Dice: `0.812502`
- Best epoch: `24`
- Best threshold: `0.40`

Interpretation:

This is the clean decoder-only CXR-BERT frozen reference under the newer protocol with HH kept. It is slightly below `fusion_mode=both`.

### 7.4 `runs/screening0506_qata_cxr_frozen_keep_both_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen
- Prompt mode: `native`
- HH mode: `keep`
- Fusion mode: `both`
- Epochs: 50
- Batch size: 4
- Optimizer: SGD
- LR scheduler: poly
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Training behavior:

- Last epoch train Dice: `0.983725`
- Last epoch val Dice: `0.794865`
- Best epoch by val Dice: `25`
- Best val Dice: `0.806679`
- Best val threshold: `0.35`

Final test:

- Loss: `0.142660`
- IoU: `0.724245`
- Dice: `0.815488`
- Best epoch: `25`
- Best threshold: `0.35`

Interpretation:

Adding encoder-side text injection to decoder TGFS gives a small gain over decoder-only under the same CXR-BERT/HH-keep setting:

- Test Dice gain over decoder-only: `+0.002986`
- Test IoU gain over decoder-only: `+0.004481`

This supports keeping `fusion_mode=both` in the candidate set, but on QaTa alone the gain is modest rather than decisive.

### 7.5 `runs/qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen
- Prompt mode: `empty`
- HH mode: `keep`
- Fusion mode: `both`
- Epochs: 35
- Early stopping patience: 8
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Training behavior:

- Stopped at epoch: `17`
- Last epoch train Dice: `0.858992`
- Last epoch val Dice: `0.748190`
- Best epoch by val Dice: `9`
- Best val Dice: `0.755809`
- Best val threshold: `0.45`

Final test:

- Loss: `0.181119`
- IoU: `0.664734`
- Dice: `0.764167`
- Best epoch: `9`
- Best threshold: `0.45`

Interpretation:

Removing prompt content drops test Dice by about `-0.051321` relative to CXR-BERT native `keep_both`. This is a strong sanity check that the text path is not just a harmless bypass.

### 7.6 `runs/qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen
- Prompt mode: `shuffle`
- HH mode: `keep`
- Fusion mode: `both`
- Epochs: 35
- Early stopping patience: 8
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Training behavior:

- Stopped at epoch: `25`
- Last epoch train Dice: `0.927333`
- Last epoch val Dice: `0.733325`
- Best epoch by val Dice: `17`
- Best val Dice: `0.756165`
- Best val threshold: `0.40`

Final test:

- Loss: `0.185526`
- IoU: `0.662672`
- Dice: `0.764517`
- Best epoch: `17`
- Best threshold: `0.40`

Interpretation:

Shuffled text performs almost the same as empty text:

- Empty test Dice: `0.764167`
- Shuffle test Dice: `0.764517`
- Difference: `+0.000350`

This is useful for the paper story. Correct text matters; random text does not preserve the gain. The model is not simply benefiting from text tokens existing.

### 7.7 `runs/qata_diag0516_qata_simple_native_keep_both_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: simple embedding + GRU
- Prompt mode: `native`
- HH mode: `keep`
- Fusion mode: `both`
- Epochs: 35
- Early stopping patience: 8
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Training behavior:

- Stopped at epoch: `19`
- Last epoch train Dice: `0.884429`
- Last epoch val Dice: `0.800330`
- Best epoch by val Dice: `11`
- Best val Dice: `0.811177`
- Best val threshold: `0.55`

Final test:

- Loss: `0.135435`
- IoU: `0.730517`
- Dice: `0.821199`
- Best epoch: `11`
- Best threshold: `0.55`

Interpretation:

This is the best completed run in the current list.

Compared with CXR-BERT frozen native `keep_both`:

- Test Dice gain: `+0.005711`
- Test IoU gain: `+0.006272`

Compared with CXR-BERT frozen native `keep_decoder`:

- Test Dice gain: `+0.008697`
- Test IoU gain: `+0.010753`

This strongly supports the deep-research hypothesis that the simple encoder is not just a toy ablation. For formulaic dataset prompts, a small trainable text encoder can be more useful than frozen CXR-BERT.

### 7.8 `runs/qata_diag0516_qata_simple_generic_keep_both_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: simple embedding + GRU
- Prompt mode: `generic`
- HH mode: `keep`
- Fusion mode: `both`
- Epochs: 35
- Early stopping patience: 8
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Training behavior:

- Stopped at epoch: `15`
- Last epoch train Dice: `0.845545`
- Last epoch val Dice: `0.747944`
- Best epoch by val Dice: `7`
- Best val Dice: `0.749395`
- Best val threshold: `0.45`

Final test:

- Loss: `0.184855`
- IoU: `0.657492`
- Dice: `0.759944`
- Best epoch: `7`
- Best threshold: `0.45`

Interpretation:

This is a negative prompt-policy result on QaTa. Simple text encoder is good only when it receives native prompt content. Replacing all prompts with one generic sentence removes lesion/location/image-specific information and drops test Dice by:

- `-0.061255` compared with simple native
- `-0.055544` compared with CXR-BERT native keep/both

This argues against using a one-size generic prompt for QaTa.

### 7.9 `runs/qata_diag0516_qata_cxr_frozen_learned_both_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen
- Prompt mode: `native`
- HH mode: `learned`
- Fusion mode: `both`
- Epochs: 35
- Early stopping patience: 8
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Training behavior:

- Stopped at epoch: `15`
- Last epoch train Dice: `0.855306`
- Last epoch val Dice: `0.782045`
- Best epoch by val Dice: `7`
- Best val Dice: `0.803095`
- Best val threshold: `0.35`

Final test:

- Loss: `0.143434`
- IoU: `0.716754`
- Dice: `0.811856`
- Best epoch: `7`
- Best threshold: `0.35`

Interpretation:

Learned HH is slightly worse than HH keep under CXR-BERT frozen `both` on QaTa:

- Test Dice vs HH keep/both: `-0.003633`
- Test IoU vs HH keep/both: `-0.007490`

This does not mean learned HH is useless globally. It only says it did not help this single-seed QaTa setting. The deep-research motivation for learned HH is stronger for MosMed/Breast, where boundary and high-frequency priors may behave differently.

### 7.10 `runs/qata_diag0516_qata_cxr_lora8_keep_both_seed42`

Configuration:

- Model: `lfaenet_tgfs_v2`
- Text encoder: CXR-BERT
- Text backbone: frozen base with LoRA rank 8 in query/value layers
- Prompt mode: `native`
- HH mode: `keep`
- Fusion mode: `both`
- Epochs target: 35
- Early stopping patience: 8
- Threshold sweep: `0.35,0.40,0.45,0.50,0.55`
- Deep supervision: off

Current status:

- Only 4 epochs are logged.
- No `final_test.json`.
- `final_test.txt` is blank.
- No Python training process was running when this report was generated.

Best partial validation:

- Epoch: `4`
- Val Dice: `0.796615`
- Val IoU: `0.693803`
- Val threshold: `0.55`

Interpretation:

This run is incomplete and should not be ranked. The partial validation curve is not obviously bad, but it has not reached a comparable checkpoint or final test. If LoRA is important for the paper story, rerun it cleanly after confirming runtime budget.

### 7.11 `runs/qata_faenet_notext_adamw_cosine_e30`

Configuration:

- Model: `faenet`
- Visual-only: `no_text=True`
- Optimizer: AdamW
- LR: `0.001`
- LR scheduler: cosine
- Weight decay: `0.001`
- Deep supervision: on
- Early stopping patience: 6
- Epochs configured: 22

Training behavior:

- Stopped at epoch: `19`
- Last epoch train Dice: `0.822787`
- Last epoch val Dice: `0.419066`
- Best epoch by val Dice: `13`
- Best val Dice: `0.534319`

Final test:

- Loss: `0.371742`
- IoU: `0.410539`
- Dice: `0.544604`

Interpretation:

This visual-only run is far below all strong text-frequency variants. However, it is not necessarily the cleanest no-text comparison because the optimization recipe differs substantially from the best current QaTa settings.

### 7.12 `runs/qata_faenet_notext_valclean_e5`

Configuration:

- Model: `faenet`
- Text is ignored because model type is visual-only even though `no_text=False` is recorded.
- Optimizer field is missing in config, but LR is `0.02`.
- Weight decay: `1e-4`
- Deep supervision: on
- Epochs configured: 45

Training behavior:

- Logged epochs: `28`
- Last epoch train Dice: `0.952982`
- Last epoch val Dice: `0.273049`
- Best epoch by val Dice: `2`
- Best val Dice: `0.648414`

Final test:

- Loss: `0.354291`
- IoU: `0.515424`
- Dice: `0.642565`

Interpretation:

This run overfits heavily. It is a better no-text reference than the AdamW/cosine run, but still much lower than the text-guided frequency models.

## 8. Main Findings

### 8.1 Text is actually being used

The strongest sanity check is real text versus empty/shuffled text under the same CXR-BERT frozen, HH keep, fusion both setup.

| Setting | Test Dice | Delta vs native CXR keep/both |
|---|---:|---:|
| Native prompt | 0.815488 | 0.000000 |
| Empty prompt | 0.764167 | -0.051321 |
| Shuffled prompt | 0.764517 | -0.050972 |

Conclusion:

The current text branch is not dead. Removing or corrupting prompt semantics costs about 5 Dice points on QaTa.

### 8.2 Empty and shuffled text are almost identical

The difference between empty and shuffled prompts is only `0.000350` Dice. This suggests that wrong text is effectively as bad as no text in this setup. That is good evidence for a text-guided segmentation paper because it separates semantic text utility from the mere presence of a language encoder.

### 8.3 Simple native is currently the best QaTa candidate

The simple native run reaches:

- Test Dice: `0.821199`
- Test IoU: `0.730517`

It beats the best clean CXR-BERT frozen `keep_both` run by:

- `+0.005711` Dice
- `+0.006272` IoU

This matters because the deep-research report argued that CXR-BERT frozen may not be the best universal choice across QaTa/MosMed/Brain/Breast. Current QaTa evidence supports that argument.

### 8.4 Generic prompt is bad on QaTa

Simple generic drops to:

- Test Dice: `0.759944`
- Test IoU: `0.657492`

This is worse than empty/shuffle CXR-BERT by a small amount and much worse than simple native. For QaTa, native prompt content is important. Generic prompts may still be useful for MosMed or other datasets, but QaTa should not default to generic.

### 8.5 `fusion_mode=both` helps, but the gain is small on QaTa

Clean CXR-BERT frozen comparison:

| Fusion | Test Dice | Test IoU |
|---|---:|---:|
| decoder | 0.812502 | 0.719764 |
| both | 0.815488 | 0.724245 |

Gain:

- `+0.002986` Dice
- `+0.004481` IoU

Conclusion:

Keep `both` as the default candidate for further experiments, but do not oversell it on QaTa alone. The stronger reason to keep `both` is still the MosMed/deep-research hypothesis that CT may need earlier and deeper text injection.

### 8.6 HH keep/learned/zero is unresolved on QaTa

Current relevant numbers:

| HH setting | Run | Test Dice |
|---|---|---:|
| legacy hard zero/drop | `qata_b4_e50_cxrbert_frozen_v2` | 0.816895 |
| keep | `screening0506_qata_cxr_frozen_keep_both_seed42` | 0.815488 |
| learned | `qata_diag0516_qata_cxr_frozen_learned_both_seed42` | 0.811856 |

On this single-seed QaTa evidence, HH keep does not clearly beat legacy hard-zero, and learned HH is slightly worse. But the comparison is not perfectly clean because the legacy hard-zero run lacks the new threshold/protocol metadata.

Practical conclusion:

For QaTa only, HH keep is safe but not proven superior. For MosMed/Breast, learned/keep HH is still scientifically justified because hard-dropping high-frequency detail may hurt boundary-sensitive tasks.

### 8.7 Visual-only FAENet is far behind current text-frequency variants

Best visual-only result in this list:

- `qata_faenet_notext_valclean_e5`: Dice `0.642565`

Strong text-frequency variants:

- CXR-BERT frozen keep/both: Dice `0.815488`
- Simple native keep/both: Dice `0.821199`

The gap is huge. However, the no-text runs are old and use different recipes. For paper-grade ablation, a cleaner visual-only baseline should be rerun under the same split, optimizer, epoch budget, threshold sweep, and checkpoint selection.

### 8.8 Single-seed variance is visible

The two legacy CXR-BERT hard-HH runs differ substantially:

- `qata_b4_e50_cxrbert_frozen_v2`: Dice `0.816895`
- `qata_b4_e50_cxrbert_frozen_v2_rerun`: Dice `0.803312`

That is a difference of `0.013583` Dice. Any final claim needs confirmatory seeds.

## 9. Current Best Candidate Set

For immediate confirmatory reruns on QaTa:

| Priority | Candidate | Reason |
|---:|---|---|
| 1 | `simple_native_keep_both` | Best completed test Dice and supports universal text-encoder story. |
| 2 | `cxr_frozen_keep_both` | Clean CXR-BERT frozen baseline under current protocol. |
| 3 | `cxr_frozen_keep_decoder` | Clean decoder-only comparison to isolate `fusion_mode=both`. |
| 4 | `cxr_frozen_learned_both` | Needed if the paper claims relaxed frequency priors; current QaTa result is slightly negative. |
| 5 | legacy hard-zero equivalent rerun under current protocol | Needed only if we want a clean zero-vs-keep-vs-learned HH table. |

Not ready:

- `cxr_lora8_keep_both`, because it is incomplete.

Not worth expanding on QaTa first:

- `simple_generic_keep_both`, because it is clearly bad on QaTa.
- More empty/shuffle variants, because the text-sensitivity conclusion is already strong.

## 10. What This Means for the Paper Story

The strongest current story is not simply "we added text to FAENet." That is too weak and too close to a generic late-fusion recipe.

The better story is:

```text
FAENet shows that frequency-aware feature refinement helps segmentation.
Our model makes frequency selection language-conditioned.
Text does not just gate channels after visual processing;
it controls LL/LH/HL/HH selection and token-grounded spatial masks inside decoder and optionally encoder stages.
```

QaTa evidence currently supports three claims:

- Text semantics matter: empty/shuffle costs about 5 Dice.
- A lightweight trainable text encoder can beat frozen CXR-BERT on formulaic prompts.
- Fusion both is slightly better than decoder-only under CXR-BERT frozen.

QaTa evidence does not yet strongly support:

- Learned HH is better than keep/zero.
- LoRA helps.
- Hybrid raw LF/HF dual-branch is necessary.

For MosMed, the deep-research report still points to a different likely bottleneck:

- The gap to FMISeg is too large to fix only with prompt/fusion tweaks.
- Current architecture does feature-level DWT, not raw-image LF/HF dual-branch.
- If MosMed remains around Dice `0.67`, the next real architectural move is likely a raw LF/HF visual redesign plus TGFS decoder, not more small prompt ablations.

## 11. Recommended Next Actions

Short-term, cheap:

- Finish or rerun `cxr_lora8_keep_both` only if LoRA is needed for the text-encoder ablation table.
- Rerun `simple_native_keep_both`, `cxr_frozen_keep_both`, and a clean `cxr_frozen_zero_both` or `cxr_frozen_zero_decoder` with seeds `42, 3407, 2026`.
- Add a clean FAENet no-text run under the same current protocol if the paper needs a fair no-text baseline.

Medium-term:

- Run the same top 2 or top 3 settings on MosMed, but avoid a huge MosMed grid because previous MosMed runs are slow and unstable.
- For MosMed, prefer fewer diagnostic runs with clear purpose: CXR native keep/both, simple native or generic keep/both, and maybe learned HH.
- Keep AMP off and gradient clip on for MosMed unless non-finite loss is fully resolved.

Architecture-term:

- Implement raw-image DWT LF/HF split before the visual encoder.
- Add two lightweight visual branches for LF and HF.
- Add FFBI-lite at bottleneck or the two deepest stages.
- Keep TGFS-v2 in decoder so the final story becomes: raw LF/HF visual separation plus language-guided sub-band selection.

## 12. Compact Result Notes for Lab Book

Best completed QaTa run:

```text
runs/qata_diag0516_qata_simple_native_keep_both_seed42
best_epoch=11
best_threshold=0.55
test_loss=0.135435
test_iou=0.730517
test_dice=0.821199
```

Best clean CXR-BERT frozen current-protocol run:

```text
runs/screening0506_qata_cxr_frozen_keep_both_seed42
best_epoch=25
best_threshold=0.35
test_loss=0.142660
test_iou=0.724245
test_dice=0.815488
```

Text sensitivity:

```text
native CXR keep/both  : dice=0.815488
empty CXR keep/both   : dice=0.764167
shuffle CXR keep/both : dice=0.764517
```

Fusion sensitivity:

```text
CXR keep decoder : dice=0.812502
CXR keep both    : dice=0.815488
delta            : +0.002986
```

Simple encoder prompt sensitivity:

```text
simple native  : dice=0.821199
simple generic : dice=0.759944
delta          : -0.061255
```

HH prior sensitivity under CXR frozen:

```text
legacy hard zero/drop : dice=0.816895
keep/both             : dice=0.815488
learned/both          : dice=0.811856
```

Incomplete:

```text
runs/qata_diag0516_qata_cxr_lora8_keep_both_seed42
logged_epochs=4
best_partial_val_dice=0.796615
final_test.json missing
```

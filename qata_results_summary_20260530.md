# QaTa / MosMed Run Summary - 2026-05-30

This file summarizes the runs currently present under `runs/` that were listed for paper/ablation tracking.

## Short Conclusion

Best completed QaTa result in the current table is:

| Rank | Run | Test Dice | Test IoU | Note |
|---:|---|---:|---:|---|
| 1 | `qata_paper0516_qata_simple_native_learned_both_seed42` | 0.8272 | 0.7391 | Scratch TGFS-v2, simple native text, both fusion, learned HH. |
| 2 | `qata_paper0516_qata_simple_native_zero_both_seed42` | 0.8259 | 0.7372 | Scratch TGFS-v2, simple native text, both fusion, HH zero. |
| 3 | `qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42` | 0.8224 | 0.7343 | ResNet50 ImageNet TGFS-v2, simple native text, decoder fusion, learned HH. |
| 4 | `qata_diag0516_qata_simple_native_keep_both_seed42` | 0.8212 | 0.7305 | Scratch TGFS-v2, simple native text, both fusion, HH keep. |
| 5 | `qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42` | 0.8210 | 0.7324 | ResNet50 ImageNet TGFS-v2, native text, drop HL. |

Main paper story from these numbers:

- Clean visual-only baseline is around 0.776-0.779 Dice, not the older 0.54/0.64 runs.
- Text-guided TGFS improves QaTa by roughly +4 to +5 Dice over clean visual-only FAENet.
- Simple text encoder is competitive or better than frozen CXR-BERT on QaTa.
- Native prompt matters: empty/shuffle/generic prompts drop clearly.
- `fusion_mode=both` and learned/zero HH priors are the strongest scratch-family settings.
- ResNet50 ImageNet encoder alone does not dominate the scratch TGFS-v2 setup on QaTa.
- The v3 remote-main recipe underperformed on QaTa in this run: 0.7993 Dice.

## Completed QaTa Runs

| Run | Model / Encoder | Text / Prompt | Fusion | Frequency Setting | Epoch / Thr | Test Dice | Test IoU | Comment |
|---|---|---|---|---|---:|---:|---:|---|
| `qata_paper0516_qata_simple_native_learned_both_seed42` | `lfaenet_tgfs_v2`, scratch | simple, native | both | HH learned | 17 / 0.50 | 0.8272 | 0.7391 | Best current QaTa run. Clean same-family text/TGFS setting. |
| `qata_paper0516_qata_simple_native_zero_both_seed42` | `lfaenet_tgfs_v2`, scratch | simple, native | both | HH zero | 18 / 0.35 | 0.8259 | 0.7372 | Nearly tied with learned HH; supports the original hard HH-drop prior. |
| `qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, native | decoder | HH learned | 12 / 0.40 | 0.8224 | 0.7343 | Best ResNet50 TGFS-v2 result. |
| `qata_diag0516_qata_simple_native_keep_both_seed42` | `lfaenet_tgfs_v2`, scratch | simple, native | both | HH keep | 11 / 0.55 | 0.8212 | 0.7305 | Strong simple encoder baseline with both fusion. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, native | decoder | drop HL | 11 / 0.45 | 0.8210 | 0.7324 | Frequency-drop ablation; dropping HL does not hurt much here. |
| `qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, native | decoder | HH zero | 24 / 0.35 | 0.8208 | 0.7310 | ResNet50 counterpart of hard HH-zero prior. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, native | decoder | drop HH | 13 / 0.50 | 0.8196 | 0.7285 | FP32 rerun replacing the crashed AMP version. |
| `qata_b4_e50_cxrbert_frozen_v2` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, legacy/native | legacy | legacy | n/a | 0.8169 | 0.7254 | Old strong CXR-BERT run; config is less explicit than newer runs. |
| `qata_paper0516_qata_simple_native_keep_decoder_seed42` | `lfaenet_tgfs_v2`, scratch | simple, native | decoder | HH keep | 17 / 0.35 | 0.8163 | 0.7243 | Same-family decoder-only comparison for simple native. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, native | decoder | drop LH | 12 / 0.35 | 0.8163 | 0.7259 | Frequency-drop ablation. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, native | decoder | HH keep | 12 / 0.50 | 0.8157 | 0.7249 | Stable FP32 ResNet50 main setting. |
| `screening0506_qata_cxr_frozen_keep_both_seed42` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, native | both | HH keep | 25 / 0.35 | 0.8155 | 0.7242 | CXR-BERT both-fusion screening run. |
| `qata_diag0516_qata_cxr_frozen_learned_both_seed42` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, native | both | HH learned | 7 / 0.35 | 0.8119 | 0.7168 | CXR-BERT learned-HH run; below simple learned/both. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, native | decoder | drop LL | 17 / 0.35 | 0.8108 | 0.7185 | Dropping LL hurts more than dropping LH/HL/HH. Useful for frequency importance. |
| `screening0506_qata_cxr_frozen_keep_decoder_seed42` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, native | decoder | HH keep | 24 / 0.40 | 0.8125 | 0.7198 | CXR-BERT decoder-only comparison. |
| `qata_b4_e50_cxrbert_frozen_v2_rerun` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, legacy/native | legacy | legacy | n/a | 0.8033 | 0.7084 | Rerun underperformed original; keep as seed/protocol variability note, not main table. |
| `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | CXR-BERT frozen, native | decoder | HH keep | 13 / 0.35 | 0.8030 | 0.7093 | CXR-BERT with ResNet50 underperforms simple text ResNet50. |
| `qata_v3_remote_main_cxr_seed42` | `lfaenet_tgfs_v3`, ResNet50 ImageNet | CXR-BERT frozen, native | both | HH learned | 5 / 0.45 | 0.7993 | 0.7057 | Remote-main v3-like recipe: 320px, GN, depth3, cross-attn, AdamW. Underperformed on QaTa. |
| `qata_resnet0523_qata_resnet50_simple_shuffle_keep_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, shuffle | decoder | HH keep | 34 / 0.35 | 0.7932 | 0.6978 | Prompt sanity: shuffled text drops vs native. |
| `qata_resnet0523_qata_resnet50_simple_generic_keep_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, generic | decoder | HH keep | 20 / 0.35 | 0.7923 | 0.6964 | Generic prompt drops vs native. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42` | `resnet50_tgfs_v2`, ImageNet ResNet50 | simple, empty | decoder | HH keep | 37 / 0.35 | 0.7860 | 0.6883 | Empty prompt drops vs native. |
| `qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` | `resnet50_faenet`, ImageNet ResNet50 | no text | n/a | FAENet visual-only | 34 / 0.35 | 0.7790 | 0.6781 | Clean ResNet50 visual-only baseline. |
| `qata_paper0516_qata_faenet_visual_clean_seed42` | `faenet`, scratch | no text | n/a | FAENet visual-only | 30 / 0.35 | 0.7757 | 0.6765 | Clean scratch visual-only baseline. Use this instead of old no-text runs. |
| `qata_paper0516_qata_simple_empty_keep_both_seed42` | `lfaenet_tgfs_v2`, scratch | simple, empty | both | HH keep | 15 / 0.35 | 0.7656 | 0.6664 | Same-family prompt sanity for final simple family. |
| `qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, shuffle | both | HH keep | 17 / 0.40 | 0.7645 | 0.6627 | CXR prompt sanity. |
| `qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, empty | both | HH keep | 9 / 0.45 | 0.7642 | 0.6647 | CXR prompt sanity. |
| `qata_diag0516_qata_simple_generic_keep_both_seed42` | `lfaenet_tgfs_v2`, scratch | simple, generic | both | HH keep | 7 / 0.45 | 0.7599 | 0.6575 | Same-family generic prompt sanity. |
| `qata_paper0516_qata_simple_shuffle_keep_both_seed42` | `lfaenet_tgfs_v2`, scratch | simple, shuffle | both | HH keep | 8 / 0.35 | 0.7585 | 0.6575 | Same-family shuffle prompt sanity. |
| `qata_faenet_notext_valclean_e5` | `faenet`, scratch | no text | n/a | old visual-only | n/a | 0.6426 | 0.5154 | Old baseline; not comparable with clean protocol. |
| `qata_faenet_notext_adamw_cosine_e30` | `faenet`, scratch | no text | n/a | old visual-only | n/a | 0.5446 | 0.4105 | Old baseline; not comparable with clean protocol. |

## Incomplete / Superseded Runs

These runs should not be used as final paper rows unless resumed/re-run. Several have FP32 replacements.

| Run | Status | Last / Best Val | Why Not Use |
|---|---|---:|---|
| `qata_diag0516_qata_cxr_lora8_keep_both_seed42` | incomplete | epoch 4, val Dice 0.7966 | No `final_test.json`; LoRA run stopped early. |
| `qata_resnet0523_qata_resnet50_simple_empty_keep_decoder_seed42` | incomplete/crashed | no epoch | Replaced by `qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42`. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hh_decoder_seed42` | incomplete/crashed | no epoch | Replaced by `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42`. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_ll_decoder_seed42` | incomplete/crashed | no epoch | Replaced by `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42`. |
| `qata_resnet0523_qata_resnet50_simple_native_keep_decoder_seed42` | incomplete/crashed | epoch 1, val Dice 0.7610 | Replaced by `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42`. |

## MosMed Run

| Run | Model / Encoder | Text / Prompt | Fusion | Frequency Setting | Epoch / Thr | Test Dice | Test IoU | Comment |
|---|---|---|---|---|---:|---:|---:|---|
| `screening0506_mosmed_cxr_frozen_keep_both_seed42` | `lfaenet_tgfs_v2`, scratch | CXR-BERT frozen, native | both | HH keep | 7 / 0.50 | 0.6737 | 0.5290 | MosMed was slow and underperformed; useful as secondary evidence only. |

## Ablation Interpretation

### Clean Visual-Only Baselines

| Variant | Run | Test Dice | Test IoU | Interpretation |
|---|---|---:|---:|---|
| Scratch FAENet visual-only | `qata_paper0516_qata_faenet_visual_clean_seed42` | 0.7757 | 0.6765 | Correct no-text baseline for paper. |
| ResNet50 FAENet visual-only | `qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` | 0.7790 | 0.6781 | ResNet50 ImageNet gives only a small visual-only gain on QaTa. |
| Old scratch no-text | `qata_faenet_notext_valclean_e5` | 0.6426 | 0.5154 | Do not compare against current TGFS runs. |
| Old scratch no-text | `qata_faenet_notext_adamw_cosine_e30` | 0.5446 | 0.4105 | Do not compare against current TGFS runs. |

### Prompt Semantics

| Family | Native | Empty | Shuffle | Generic | Conclusion |
|---|---:|---:|---:|---:|---|
| Scratch simple, both fusion | 0.8212 | 0.7656 | 0.7585 | 0.7599 | Native text adds roughly +5.5 to +6.3 Dice over weak/incorrect prompts. |
| ResNet50 simple, decoder fusion | 0.8157 | 0.7860 | 0.7932 | 0.7923 | Native text still helps, but gap is smaller because ResNet visual encoder is stronger. |
| Scratch CXR-BERT, both fusion | 0.8155 | 0.7642 | 0.7645 | n/a | Same semantic sanity holds for CXR-BERT. |

### Fusion Locus

| Family | Decoder | Both | Delta |
|---|---:|---:|---:|
| Scratch CXR-BERT, HH keep | 0.8125 | 0.8155 | +0.0030 |
| Scratch simple, HH keep | 0.8163 | 0.8212 | +0.0049 |

`fusion_mode=both` helps modestly but consistently in the completed scratch-family comparisons.

### HH / Frequency Priors

| Family | Keep | Zero | Learned | Comment |
|---|---:|---:|---:|---|
| Scratch simple, both fusion | 0.8212 | 0.8259 | 0.8272 | Learned HH is best, zero HH close second. |
| ResNet50 simple, decoder fusion | 0.8157 | 0.8208 | 0.8224 | Same trend: learned/zero HH beats keep. |
| Scratch CXR-BERT, both fusion | 0.8155 | n/a | 0.8119 | CXR learned-HH run did not improve. |

### Drop One Wavelet Band

| ResNet50 Simple Decoder Variant | Dropped Band | Test Dice | Test IoU | Interpretation |
|---|---|---:|---:|---|
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42` | none | 0.8157 | 0.7249 | Baseline for drop-band comparison. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42` | LL | 0.8108 | 0.7185 | LL removal hurts the most among clean FP32 drop runs. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42` | LH | 0.8163 | 0.7259 | Similar to baseline. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42` | HL | 0.8210 | 0.7324 | Surprisingly above baseline; likely noise/protocol difference because this is AMP lr=0.005, not FP32 lr=0.003. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42` | HH | 0.8196 | 0.7285 | Similar/slightly above baseline; should be interpreted cautiously. |

Important caveat: drop-band ablations are not fully protocol-matched because LL/HH replacements used FP32 lr=0.003 while LH/HL are older AMP lr=0.005 runs.

### Encoder / Text Encoder

| Variant | Test Dice | Comment |
|---|---:|---|
| Scratch simple native learned both | 0.8272 | Best current. |
| ResNet50 simple native learned decoder | 0.8224 | Strong but below scratch best. |
| Scratch CXR-BERT native keep both | 0.8155 | Below simple native both. |
| ResNet50 CXR-BERT native keep decoder | 0.8030 | CXR-BERT does not help QaTa here. |
| v3 remote-main CXR recipe | 0.7993 | Remote brain/breast recipe does not transfer directly to QaTa in this run. |

## Paper-Ready Rows I Would Use

For the main QaTa ablation table, use the following concise set:

| Purpose | Run |
|---|---|
| Visual-only baseline | `qata_paper0516_qata_faenet_visual_clean_seed42` |
| ResNet50 visual-only baseline | `qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` |
| Final/best model | `qata_paper0516_qata_simple_native_learned_both_seed42` |
| HH prior ablation | `qata_diag0516_qata_simple_native_keep_both_seed42`, `qata_paper0516_qata_simple_native_zero_both_seed42`, `qata_paper0516_qata_simple_native_learned_both_seed42` |
| Prompt sanity | `qata_diag0516_qata_simple_native_keep_both_seed42`, `qata_paper0516_qata_simple_empty_keep_both_seed42`, `qata_paper0516_qata_simple_shuffle_keep_both_seed42`, `qata_diag0516_qata_simple_generic_keep_both_seed42` |
| Fusion locus | `qata_paper0516_qata_simple_native_keep_decoder_seed42`, `qata_diag0516_qata_simple_native_keep_both_seed42` |
| ResNet50 comparison | `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42`, `qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42` |
| CXR-BERT comparison | `screening0506_qata_cxr_frozen_keep_both_seed42`, `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` |

## Missing / Recommended Additional Runs

Minimum missing pieces for a cleaner paper:

1. Three-seed confirmation for the selected final model: `simple_native_learned_both` with seeds `42`, `3407`, `2026`.
2. Three-seed clean visual-only baseline: `faenet_visual_clean` with seeds `42`, `3407`, `2026`.
3. Protocol-matched frequency drop ablation for all bands under the final family, ideally scratch simple both fusion:
   - `simple_native_drop_ll_both`
   - `simple_native_drop_lh_both`
   - `simple_native_drop_hl_both`
   - `simple_native_drop_hh_both`
4. If using ResNet50 frequency-drop table, rerun LH/HL with the same FP32 lr=0.003 protocol as LL/HH/keep.
5. If claiming CXR-BERT is weaker than simple, run one clean same-protocol CXR-BERT counterpart for the final setting:
   - `cxr_native_learned_both`
6. For qualitative figures, keep checkpoints for:
   - `qata_paper0516_qata_simple_native_learned_both_seed42`
   - `qata_paper0516_qata_simple_empty_keep_both_seed42`
   - `qata_paper0516_qata_simple_shuffle_keep_both_seed42`
   - `qata_paper0516_qata_faenet_visual_clean_seed42`

Optional but useful:

1. A QaTa v3 simple-text run, because current v3 run is CXR-BERT only and underperformed:
   - `lfaenet_tgfs_v3`, ResNet50, simple native, both, learned HH, 320px.
2. MosMed should only be expanded if it becomes part of the main story. Current single MosMed run is too weak/slow for a central claim.

## Detailed Notes

### How To Read The Run Names

Most run names encode four things: dataset, architecture family, text setting, and frequency/fusion setting.

| Name Fragment | Meaning |
|---|---|
| `qata_*` | QaTa-COV19-v2 experiment. |
| `mosmed_*` | MosMed experiment. |
| `faenet_visual_clean` | Visual-only FAENet baseline. No prompt/text branch should be used. |
| `simple_*` | Uses the repo's simple text encoder, not CXR-BERT. |
| `cxr_*` or `cxrbert_frozen` | Uses BiomedVLP/CXR-BERT frozen as the text encoder. |
| `lora8` | CXR-BERT with LoRA rank 8. The listed LoRA run is incomplete and should not be used. |
| `native` | Uses the dataset-provided prompt/text. This is the real text-guided setting. |
| `empty` | Text is replaced by an empty string. This tests whether the text branch helps without semantics. |
| `shuffle` | Text prompts are shuffled across samples. This tests whether sample-specific semantics matter. |
| `generic` | All prompts are replaced by a generic segmentation prompt. This tests whether the model only needs a generic task cue. |
| `keep` | Keep HH in the decoder TGFS block. |
| `zero` | Force HH to zero in decoder TGFS. This is the original hard high-frequency prior. |
| `learned` | Learn the HH retention scale instead of hard dropping/keeping it. |
| `drop_ll/lh/hl/hh` | Remove one DWT sub-band for frequency importance ablation. |
| `decoder` | Text is injected in decoder TGFS only. |
| `both` | Text is injected in encoder-side fusion and decoder TGFS. |
| `resnet50` | ImageNet-pretrained ResNet50 image encoder family. |
| `fp32fix` | Rerun with AMP disabled / more stable FP32 settings after earlier crashes. |
| `v3_remote_main` | Port of the newer remote-main v3-like recipe with ResNet50, 320px, GroupNorm, depth-3 conv blocks, cross-attention fusion, AdamW. |

### Protocol Groups

The runs are not all from one identical protocol. That matters for paper wording.

| Group | Runs | Protocol Notes | Paper Use |
|---|---|---|---|
| Old no-text baselines | `qata_faenet_notext_adamw_cosine_e30`, `qata_faenet_notext_valclean_e5` | Old optimizer/schedule/config; results 0.54/0.64 are not comparable to current clean runs. | Do not use as primary baseline. Mention only as deprecated if needed. |
| Clean scratch FAENet baseline | `qata_paper0516_qata_faenet_visual_clean_seed42` | Same general QaTa protocol as paper0516 text runs; SGD/poly, 224px, selected by val threshold sweep. | Use as main visual-only baseline. |
| Clean scratch TGFS-v2 paper runs | `qata_paper0516_*`, `qata_diag0516_qata_simple_*` | Scratch encoder, TGFS-v2, mostly 35 epochs with early stopping. | Use for main ablation table. |
| CXR screening runs | `screening0506_qata_*`, `qata_diag0516_qata_cxr_*` | Scratch TGFS-v2 with CXR-BERT frozen. Some are 50 epochs, some 35 epochs. | Use as secondary text-encoder comparison. |
| ResNet50 TGFS-v2 runs | `qata_resnet0523_*` and `qata_resnet0523_fp32fix_*` | ImageNet ResNet50 encoder, TGFS-v2 decoder. Some old AMP/lr=0.005, some FP32/lr=0.003. | Use carefully; avoid overclaiming drop-band results unless protocol-matched. |
| ResNet50 CXR run | `qata_resnet0524_cxr_*` | ImageNet ResNet50 + CXR-BERT frozen, decoder-only. | Shows CXR-BERT does not beat simple on QaTa. |
| v3 remote-main run | `qata_v3_remote_main_cxr_seed42` | 320px, AdamW, CXR-BERT, ResNet50, both/cross-attn, learned HH, GN/depth3/dropout. | Negative transfer result; do not use as final unless more v3 runs are tested. |
| MosMed run | `screening0506_mosmed_*` | Slow run, early-stopped/partial-looking but has final test. | Supplementary only unless expanded. |

### Main Claim Support

Claim 1: Text-guided TGFS improves over visual-only segmentation.

| Comparison | Dice Gain |
|---|---:|
| `simple_native_learned_both` 0.8272 vs scratch `faenet_visual_clean` 0.7757 | +0.0515 |
| `simple_native_zero_both` 0.8259 vs scratch `faenet_visual_clean` 0.7757 | +0.0502 |
| ResNet50 `simple_native_learned_decoder` 0.8224 vs ResNet50 `faenet_visual_clean` 0.7790 | +0.0433 |

This is the cleanest result story: the gain remains about +4 to +5 Dice even when the visual encoder is stronger.

Claim 2: The text branch uses semantic content, not just an extra conditioning path.

| Family | Native | Empty | Shuffle | Generic |
|---|---:|---:|---:|---:|
| Scratch simple both | 0.8212 | 0.7656 | 0.7585 | 0.7599 |
| ResNet50 simple decoder | 0.8157 | 0.7860 | 0.7932 | 0.7923 |
| Scratch CXR-BERT both | 0.8155 | 0.7642 | 0.7645 | n/a |

The cleanest prompt sanity table is the scratch simple family because it matches the best/final model family more closely than CXR-BERT.

Claim 3: Frequency priors matter, but hard high-frequency assumptions should be softened.

| Setting | Scratch Simple Both | ResNet50 Simple Decoder |
|---|---:|---:|
| HH keep | 0.8212 | 0.8157 |
| HH zero | 0.8259 | 0.8208 |
| HH learned | 0.8272 | 0.8224 |

Both scratch and ResNet50 families favor `learned` or `zero` over `keep`. This supports the paper story that HH should not be blindly kept, but learned HH is the safer final design because it avoids hard-coded removal.

Claim 4: ResNet50 is not the main contribution on QaTa.

| Comparison | Dice |
|---|---:|
| Scratch FAENet clean | 0.7757 |
| ResNet50 FAENet clean | 0.7790 |
| Scratch simple learned both | 0.8272 |
| ResNet50 simple learned decoder | 0.8224 |

ResNet50 slightly improves the visual-only baseline, but does not beat the best scratch TGFS-v2 setting. The core gain is from text-guided TGFS/frequency selection, not simply ImageNet pretraining.

### Run-By-Run Notes

#### Best / Final Candidate Runs

`qata_paper0516_qata_simple_native_learned_both_seed42`

- Best completed QaTa run: Dice 0.8272, IoU 0.7391.
- Uses scratch `lfaenet_tgfs_v2`, simple text encoder, native prompts, `fusion_mode=both`, learned HH.
- This should be treated as the current final candidate.
- Limitation: only one seed so far.

`qata_paper0516_qata_simple_native_zero_both_seed42`

- Dice 0.8259, IoU 0.7372.
- Same family as best run, but HH is hard-zeroed.
- Useful to show the old hard HH prior is strong, but learned HH is slightly better and more defensible.

`qata_diag0516_qata_simple_native_keep_both_seed42`

- Dice 0.8212, IoU 0.7305.
- Same simple/native/both family, but HH is kept.
- Use as the `HH keep` point in the HH ablation.

#### Clean Visual Baselines

`qata_paper0516_qata_faenet_visual_clean_seed42`

- Dice 0.7757, IoU 0.6765.
- This is the correct scratch no-text baseline.
- Use this in the main table instead of the old 0.54/0.64 baselines.

`qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42`

- Dice 0.7790, IoU 0.6781.
- Clean ResNet50 ImageNet visual-only baseline.
- Important because it shows the visual backbone alone does not explain the +0.82 Dice text-guided results.

#### Prompt Sanity Runs

`qata_paper0516_qata_simple_empty_keep_both_seed42`

- Dice 0.7656.
- Same simple/both/HH-keep family but with empty prompts.
- Shows the text branch without semantic content is much weaker.

`qata_paper0516_qata_simple_shuffle_keep_both_seed42`

- Dice 0.7585.
- Same family with shuffled prompts.
- This is the strongest evidence that sample-specific text matters.

`qata_diag0516_qata_simple_generic_keep_both_seed42`

- Dice 0.7599.
- Generic prompt is also much lower than native.
- Useful to argue that the model is not merely benefiting from a generic “segment lesion” instruction.

#### Fusion Runs

`qata_paper0516_qata_simple_native_keep_decoder_seed42`

- Dice 0.8163.
- Decoder-only text injection, HH keep.
- Compare to `qata_diag0516_qata_simple_native_keep_both_seed42` at 0.8212.
- The gain from `both` is modest (+0.0049) but consistent with the CXR-BERT screening pair.

`screening0506_qata_cxr_frozen_keep_decoder_seed42` and `screening0506_qata_cxr_frozen_keep_both_seed42`

- Decoder: 0.8125.
- Both: 0.8155.
- Confirms the same trend for CXR-BERT frozen.

#### CXR-BERT Runs

`screening0506_qata_cxr_frozen_keep_both_seed42`

- Dice 0.8155.
- Good CXR-BERT frozen run, but below simple/native/both.
- Use as a fair CXR-BERT comparison point, not as final model.

`qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42` and `qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42`

- Empty: 0.7642.
- Shuffle: 0.7645.
- Confirms text semantics matters even with CXR-BERT.

`qata_diag0516_qata_cxr_lora8_keep_both_seed42`

- Incomplete after epoch 4, no final test.
- Do not include in result tables.
- If LoRA is mentioned, it needs a complete rerun.

`qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42`

- Dice 0.8030.
- ResNet50 + CXR-BERT frozen underperforms ResNet50 + simple text.
- Useful if the paper needs a “domain-specific chest language encoder is not automatically better” observation.

#### ResNet50 Runs

`qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42`

- Dice 0.8157.
- Stable FP32 ResNet50 simple/native/HH-keep baseline.
- Use this as the anchor for ResNet50 decoder-family comparison.

`qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42`

- Dice 0.8224.
- Best ResNet50 TGFS-v2 setting.
- It supports learned HH, but was run under the older AMP/lr=0.005 protocol, so compare cautiously with FP32 runs.

`qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42`

- Dice 0.8208.
- Strong HH-zero ResNet50 result.

`qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42`

- Dice 0.7860.
- Empty prompt sanity for ResNet50 family.

`qata_resnet0523_qata_resnet50_simple_shuffle_keep_decoder_seed42`

- Dice 0.7932.
- Shuffle prompt sanity for ResNet50 family.

`qata_resnet0523_qata_resnet50_simple_generic_keep_decoder_seed42`

- Dice 0.7923.
- Generic prompt sanity for ResNet50 family.

#### Frequency Drop Runs

`qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42`

- Dice 0.8108.
- Dropping LL lowers performance relative to ResNet50 keep baseline 0.8157.
- This is directionally sensible because LL carries low-frequency shape/region information.

`qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42`

- Dice 0.8163.
- Very close to keep baseline.
- Protocol differs from FP32 runs.

`qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42`

- Dice 0.8210.
- Higher than keep baseline, which is suspicious if interpreted causally.
- Treat as noisy/protocol-mismatched unless rerun under identical FP32 settings.

`qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42`

- Dice 0.8196.
- Similar to or slightly above keep baseline.
- This aligns with the earlier observation that blindly keeping HH is not always best on QaTa.

#### v3 Remote-Main Run

`qata_v3_remote_main_cxr_seed42`

- Dice 0.7993, IoU 0.7057.
- Uses the remote-main v3-style architecture/recipe:
  - ResNet50 image encoder.
  - Image size 320.
  - CXR-BERT frozen.
  - `fusion_mode=both`.
  - encoder text fusion via cross-attention.
  - learned HH.
  - GroupNorm.
  - depth-3 conv blocks.
  - dropout 0.1.
  - AdamW with differential LR.
- It underperformed on QaTa, so remote-main brain/breast improvements do not transfer directly.
- Possible reasons:
  - CXR-BERT may not match QaTa prompts as well as simple/native in this setup.
  - 320px + heavy v3 recipe may overfit/optimize differently on QaTa split.
  - QaTa may favor simpler TGFS-v2 scratch training with SGD/poly.
  - The v3 run early-stopped at best epoch 5, suggesting optimization/regularization mismatch.

#### MosMed Run

`screening0506_mosmed_cxr_frozen_keep_both_seed42`

- Dice 0.6737, IoU 0.5290.
- Useful as a secondary note, but it is not strong enough to anchor a paper claim.
- MosMed is slower and needs a dedicated smaller ablation plan if used seriously.

### Main Table Recommendation

Use one compact main table instead of dumping every run.

| Row Label | Run | Dice | Why Include |
|---|---|---:|---|
| FAENet visual-only | `qata_paper0516_qata_faenet_visual_clean_seed42` | 0.7757 | Clean no-text baseline. |
| FAENet ResNet50 visual-only | `qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` | 0.7790 | Shows backbone pretraining alone is insufficient. |
| TGFS simple native decoder | `qata_paper0516_qata_simple_native_keep_decoder_seed42` | 0.8163 | Text-guided decoder-only baseline. |
| TGFS simple native both | `qata_diag0516_qata_simple_native_keep_both_seed42` | 0.8212 | Shows both fusion helps. |
| TGFS simple native both + zero HH | `qata_paper0516_qata_simple_native_zero_both_seed42` | 0.8259 | Strong hard-prior result. |
| TGFS simple native both + learned HH | `qata_paper0516_qata_simple_native_learned_both_seed42` | 0.8272 | Final candidate. |
| TGFS simple empty | `qata_paper0516_qata_simple_empty_keep_both_seed42` | 0.7656 | Prompt semantics ablation. |
| TGFS simple shuffle | `qata_paper0516_qata_simple_shuffle_keep_both_seed42` | 0.7585 | Prompt semantics ablation. |

Put the long ResNet50/frequency/CXR/v3 tables in supplementary unless the paper has enough space.

### Supplementary Table Recommendation

Recommended supplementary tables:

1. Full completed run ranking.
2. Prompt sanity table for simple/CXR/ResNet50 families.
3. Frequency prior table: HH keep/zero/learned.
4. Drop-band table with a clear caveat about protocol mismatch.
5. ResNet50 and v3 transfer table.
6. MosMed secondary result.

### What Is Still Missing For A Stronger Paper

Critical:

1. Three-seed final model:
   - `simple_native_learned_both`, seeds 42/3407/2026.
2. Three-seed clean visual baseline:
   - `faenet_visual_clean`, seeds 42/3407/2026.
3. Three-seed prompt sanity is optional, but at least final and visual baseline should have mean/std.

Important:

1. Protocol-matched scratch drop-band ablation under final family:
   - `simple_native_drop_ll_both`
   - `simple_native_drop_lh_both`
   - `simple_native_drop_hl_both`
   - `simple_native_drop_hh_both`
2. Protocol-matched CXR counterpart to final:
   - `cxr_native_learned_both`
3. Complete LoRA run if claiming LoRA was evaluated:
   - `cxr_lora8_keep_both` or `cxr_lora8_learned_both`.

Optional:

1. v3 simple-text run:
   - Current v3 only tests CXR-BERT; the best QaTa family uses simple text.
2. MosMed dedicated fast ablation:
   - simple native vs CXR native, decoder vs both, keep vs learned HH.

### Safe Wording For The Paper

Use:

> On QaTa-COV19, the proposed text-guided frequency selection consistently improves over visual-only FAENet. The strongest single-seed setting uses a lightweight native-prompt text encoder with both encoder and decoder fusion and learned HH retention, improving Dice from 77.57 to 82.72.

Avoid:

> ResNet50 is the reason for the improvement.

Reason: ResNet50 visual-only is only 77.90 Dice, and the best scratch TGFS-v2 is still higher than ResNet50 TGFS-v2.

Avoid:

> CXR-BERT is the best text encoder.

Reason: simple/native text is better in the current QaTa runs.

Avoid:

> Dropping HL improves performance.

Reason: HL drop result is protocol-mismatched and could be noise.

Use cautiously:

> The frequency-drop study suggests LL is important and HH can be suppressed or learned, but protocol-matched drop-band reruns are needed for a definitive ranking of all sub-bands.

## Full Variant Catalog

This section explains each listed run more explicitly. The goal is to make every folder name interpretable without opening `config.json`.

### Legacy / Early QaTa Runs

| Run | Detailed Meaning | What It Was Testing | Result | How To Treat It |
|---|---|---|---:|---|
| `qata_b4_e50_cxrbert_frozen_v2` | Early TGFS-v2 run on QaTa with CXR-BERT frozen, batch size 4, 50 epochs. Config is older and does not explicitly record all newer flags such as prompt mode, HH mode, and fusion mode. | Early proof that text-guided TGFS can reach around 0.81 Dice on QaTa. | Dice 0.8169 | Useful historical reference, but not the cleanest paper row because protocol metadata is incomplete. |
| `qata_b4_e50_cxrbert_frozen_v2_rerun` | Rerun of the previous CXR-BERT frozen setup under more explicit SGD/poly config. | Reproducibility check for the early CXR-BERT run. | Dice 0.8033 | Shows variance/instability. Do not use as final headline. |
| `qata_faenet_notext_adamw_cosine_e30` | Old visual-only FAENet run using AdamW/cosine. Config says 22 epochs; not aligned with later clean protocol. | Early no-text baseline attempt. | Dice 0.5446 | Do not compare directly to current TGFS runs. Superseded by `faenet_visual_clean`. |
| `qata_faenet_notext_valclean_e5` | Old visual-only FAENet run with a cleaner validation attempt, but still not aligned with the later paper protocol. | Another early no-text baseline attempt. | Dice 0.6426 | Do not use as main baseline. Superseded by `faenet_visual_clean`. |

### Scratch TGFS-v2: CXR-BERT Prompt/Fusion/Frequency Runs

| Run | Detailed Meaning | What It Was Testing | Result | How To Treat It |
|---|---|---|---:|---|
| `screening0506_qata_cxr_frozen_keep_decoder_seed42` | Scratch `lfaenet_tgfs_v2`, CXR-BERT frozen, native prompt, HH keep, decoder-only fusion, 50 epochs. | Baseline CXR-BERT decoder-only text injection. | Dice 0.8125 | Use as CXR-BERT decoder-fusion comparison. |
| `screening0506_qata_cxr_frozen_keep_both_seed42` | Scratch `lfaenet_tgfs_v2`, CXR-BERT frozen, native prompt, HH keep, both encoder and decoder fusion, 50 epochs. | Whether adding encoder-side text fusion helps CXR-BERT setting. | Dice 0.8155 | Useful CXR-BERT both-fusion comparison. Shows both > decoder by +0.003. |
| `qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42` | Same CXR-BERT/both/HH-keep family, but prompt is empty. | Tests whether CXR-BERT branch helps without actual text semantics. | Dice 0.7642 | Strong prompt sanity evidence. Use in supplementary or prompt-ablation table. |
| `qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42` | Same CXR-BERT/both/HH-keep family, but prompts are shuffled across samples. | Tests sample-specific text alignment. | Dice 0.7645 | Strong prompt sanity evidence. Native beats shuffled by about +0.051. |
| `qata_diag0516_qata_cxr_frozen_learned_both_seed42` | Scratch TGFS-v2, CXR-BERT frozen, native prompt, both fusion, learned HH retention. | Whether learned HH helps under frozen CXR-BERT. | Dice 0.8119 | Negative/neutral result; learned HH did not improve CXR-BERT family here. |
| `qata_diag0516_qata_cxr_lora8_keep_both_seed42` | Scratch TGFS-v2, CXR-BERT with LoRA rank 8, native prompt, both fusion, HH keep. | Whether lightweight CXR-BERT adaptation helps. | No final test | Incomplete. Do not report as evaluated unless rerun to completion. |

### Scratch TGFS-v2: Simple Text Main Paper Runs

| Run | Detailed Meaning | What It Was Testing | Result | How To Treat It |
|---|---|---|---:|---|
| `qata_diag0516_qata_simple_native_keep_both_seed42` | Scratch `lfaenet_tgfs_v2`, simple text encoder, native prompt, HH keep, both fusion. | Strong simple-text baseline and same-family `HH keep` point. | Dice 0.8212 | Use in main or supplementary. It supports simple text and both fusion. |
| `qata_diag0516_qata_simple_generic_keep_both_seed42` | Same as above, but all prompts are generic. | Whether a generic task instruction is enough. | Dice 0.7599 | Use for prompt semantics. It shows native prompt is much better. |
| `qata_paper0516_qata_simple_empty_keep_both_seed42` | Scratch TGFS-v2, simple text encoder, empty prompt, both fusion, HH keep. | Whether the architecture works without text content. | Dice 0.7656 | Use for prompt sanity. Empty text is far below native. |
| `qata_paper0516_qata_simple_shuffle_keep_both_seed42` | Scratch TGFS-v2, simple text encoder, shuffled prompt, both fusion, HH keep. | Whether image-text pairing matters. | Dice 0.7585 | Use for prompt sanity. This is one of the clearest semantic controls. |
| `qata_paper0516_qata_simple_native_keep_decoder_seed42` | Scratch TGFS-v2, simple text encoder, native prompt, HH keep, decoder-only fusion. | Same-family decoder-only baseline for fusion ablation. | Dice 0.8163 | Compare with `simple_native_keep_both` to show both fusion helps. |
| `qata_paper0516_qata_simple_native_zero_both_seed42` | Scratch TGFS-v2, simple text encoder, native prompt, both fusion, hard-zero HH in decoder. | Whether original hard HH suppression prior is useful. | Dice 0.8259 | Very strong. Use in HH prior ablation. |
| `qata_paper0516_qata_simple_native_learned_both_seed42` | Scratch TGFS-v2, simple text encoder, native prompt, both fusion, learned HH retention. | Final candidate: lets the model learn whether HH should be retained instead of forcing zero/keep. | Dice 0.8272 | Best current run. Use as main final model unless three-seed rerun changes conclusion. |
| `qata_paper0516_qata_faenet_visual_clean_seed42` | Scratch FAENet visual-only baseline, no text branch. | Clean baseline to measure text/TGFS gain. | Dice 0.7757 | Use as main no-text baseline. |

### ResNet50 Visual-Only And Simple Text Runs

| Run | Detailed Meaning | What It Was Testing | Result | How To Treat It |
|---|---|---|---:|---|
| `qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` | ResNet50 ImageNet encoder with FAENet-style visual-only decoder, no text. FP32 stable rerun. | Whether ImageNet ResNet50 alone gives a much stronger visual baseline. | Dice 0.7790 | Use as visual-pretraining control. It only slightly beats scratch visual-only. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42` | ResNet50 ImageNet encoder, TGFS-v2 decoder, simple native text, HH keep, decoder-only fusion, FP32. | Stable ResNet50 text-guided baseline. | Dice 0.8157 | Use as ResNet50 simple/native anchor. |
| `qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42` | ResNet50 ImageNet encoder, simple native text, decoder fusion, learned HH. Older AMP/lr=0.005 protocol. | Whether learned HH helps in ResNet50 family. | Dice 0.8224 | Best ResNet50 TGFS result, but note protocol differs from FP32 runs. |
| `qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42` | ResNet50 ImageNet encoder, simple native text, decoder fusion, HH zero. Older AMP/lr=0.005 protocol. | Whether hard HH-zero helps in ResNet50 family. | Dice 0.8208 | Supports same trend as scratch: zero/learned > keep. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42` | ResNet50 ImageNet encoder, simple empty prompt, HH keep, decoder fusion, FP32. | Empty-prompt sanity in ResNet50 family. | Dice 0.7860 | Use in prompt sanity supplementary table. |
| `qata_resnet0523_qata_resnet50_simple_shuffle_keep_decoder_seed42` | ResNet50 ImageNet encoder, simple shuffled prompt, HH keep, decoder fusion. | Prompt pairing sanity in ResNet50 family. | Dice 0.7932 | Shows native > shuffled, but gap is smaller than scratch. |
| `qata_resnet0523_qata_resnet50_simple_generic_keep_decoder_seed42` | ResNet50 ImageNet encoder, generic prompt, HH keep, decoder fusion. | Generic prompt control in ResNet50 family. | Dice 0.7923 | Use as supplementary prompt control if needed. |

### ResNet50 Drop-Band Frequency Ablations

| Run | Detailed Meaning | What It Was Testing | Result | How To Treat It |
|---|---|---|---:|---|
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42` | ResNet50 TGFS-v2, simple native text, decoder fusion, LL sub-band removed, FP32. | Importance of LL low-frequency component. | Dice 0.8108 | Directionally useful: removing LL hurts. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42` | ResNet50 TGFS-v2, simple native text, decoder fusion, LH removed. Older AMP/lr=0.005 protocol. | Importance of LH high-frequency component. | Dice 0.8163 | Similar to keep, but protocol mismatch limits conclusion. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42` | ResNet50 TGFS-v2, simple native text, decoder fusion, HL removed. Older AMP/lr=0.005 protocol. | Importance of HL high-frequency component. | Dice 0.8210 | Suspiciously high; do not claim dropping HL is beneficial without protocol-matched rerun. |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42` | ResNet50 TGFS-v2, simple native text, decoder fusion, HH removed, FP32. | Importance of HH diagonal high-frequency component. | Dice 0.8196 | Suggests HH suppression is not harmful, consistent with zero/learned HH results. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_ll_decoder_seed42` | Earlier AMP version of LL-drop run. | Original attempt before FP32 fix. | No final test | Superseded by FP32 LL-drop. |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hh_decoder_seed42` | Earlier AMP version of HH-drop run. | Original attempt before FP32 fix. | No final test | Superseded by FP32 HH-drop. |

### ResNet50 CXR-BERT And v3 Runs

| Run | Detailed Meaning | What It Was Testing | Result | How To Treat It |
|---|---|---|---:|---|
| `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` | ResNet50 ImageNet encoder, CXR-BERT frozen, native prompt, HH keep, decoder-only fusion, FP32. | Direct CXR-BERT vs simple text comparison in ResNet50 family. | Dice 0.8030 | Use to show CXR-BERT is not automatically better on QaTa. |
| `qata_v3_remote_main_cxr_seed42` | Port of remote-main v3 recipe: ResNet50, CXR-BERT frozen, 320px, both fusion, encoder cross-attention, learned HH, GroupNorm, depth-3 conv, dropout, AdamW, differential LR. | Whether the brain/breast remote-main recipe transfers to QaTa. | Dice 0.7993 | Negative transfer result. Do not use as main model. Good to note internally. |

### MosMed Run

| Run | Detailed Meaning | What It Was Testing | Result | How To Treat It |
|---|---|---|---:|---|
| `screening0506_mosmed_cxr_frozen_keep_both_seed42` | MosMed text TGFS-v2, scratch encoder, CXR-BERT frozen, native prompt, HH keep, both fusion. | First MosMed transfer/screening result. | Dice 0.6737 | Secondary only. It is too weak and slow to be central without more MosMed-specific work. |

### Superseded / Incomplete ResNet50 Attempts

| Run | Detailed Meaning | Status | Replacement |
|---|---|---|---|
| `qata_resnet0523_qata_resnet50_simple_native_keep_decoder_seed42` | Original ResNet50 simple native keep decoder with AMP/lr=0.005. | Crashed/incomplete after epoch 1. | `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42` |
| `qata_resnet0523_qata_resnet50_simple_empty_keep_decoder_seed42` | Original ResNet50 simple empty keep decoder with AMP/lr=0.005. | Crashed/incomplete. | `qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42` |
| `qata_resnet0523_qata_resnet50_simple_native_drop_ll_decoder_seed42` | Original ResNet50 LL-drop with AMP/lr=0.005. | Crashed/incomplete. | `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42` |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hh_decoder_seed42` | Original ResNet50 HH-drop with AMP/lr=0.005. | Crashed/incomplete. | `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42` |

## Mean Dice vs Global Dice

Global Dice is computed by summing intersections and foreground pixels over the whole test set, then applying Dice once. Mean Dice is the existing per-image averaged `final_test.json` Dice. Runs without `best.pt` or without a compatible evaluator are marked `-`.

| Run | Threshold | Mean Dice | Global Dice | Mean IoU | Global IoU | Status / Note |
|---|---:|---:|---:|---:|---:|---|
| `qata_b4_e50_cxrbert_frozen_v2` | 0.5000 | 0.8169 | 0.2706 | 0.7254 | 0.1565 | ok |
| `qata_b4_e50_cxrbert_frozen_v2_rerun` | 0.5000 | 0.8033 | 0.0000 | 0.7084 | 0.0000 | ok |
| `qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42` | 0.4500 | 0.7642 | 0.8476 | 0.6647 | 0.7355 | ok |
| `qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42` | 0.4000 | 0.7645 | 0.8485 | 0.6627 | 0.7369 | ok |
| `qata_diag0516_qata_cxr_frozen_learned_both_seed42` | 0.3500 | 0.8119 | 0.8865 | 0.7168 | 0.7961 | ok |
| `qata_diag0516_qata_cxr_lora8_keep_both_seed42` | - | - | - | - | - | final_test.json missing; incomplete run |
| `qata_diag0516_qata_simple_generic_keep_both_seed42` | 0.4500 | 0.7599 | 0.8416 | 0.6575 | 0.7265 | ok |
| `qata_diag0516_qata_simple_native_keep_both_seed42` | 0.5500 | 0.8212 | 0.8945 | 0.7305 | 0.8091 | ok |
| `qata_faenet_notext_adamw_cosine_e30` | 0.5000 | 0.5446 | 0.1955 | 0.4105 | 0.1083 | ok |
| `qata_faenet_notext_valclean_e5` | 0.5000 | 0.6426 | 0.7373 | 0.5154 | 0.5838 | ok |
| `qata_paper0516_qata_faenet_visual_clean_seed42` | 0.3500 | 0.7757 | 0.8612 | 0.6765 | 0.7562 | ok |
| `qata_paper0516_qata_simple_empty_keep_both_seed42` | 0.3500 | 0.7656 | 0.8543 | 0.6664 | 0.7457 | ok |
| `qata_paper0516_qata_simple_native_keep_decoder_seed42` | 0.3500 | 0.8163 | 0.8948 | 0.7243 | 0.8097 | ok |
| `qata_paper0516_qata_simple_native_learned_both_seed42` | 0.5000 | 0.8272 | 0.8996 | 0.7391 | 0.8176 | ok |
| `qata_paper0516_qata_simple_native_zero_both_seed42` | 0.3500 | 0.8259 | 0.9004 | 0.7372 | 0.8188 | ok |
| `qata_paper0516_qata_simple_shuffle_keep_both_seed42` | 0.3500 | 0.7585 | 0.8466 | 0.6575 | 0.7340 | ok |
| `qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42` | 0.3500 | 0.7790 | 0.8649 | 0.6781 | 0.7619 | ok |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42` | 0.3500 | 0.7860 | 0.8708 | 0.6883 | 0.7711 | ok |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42` | 0.5000 | 0.8196 | - | 0.7285 | - | best.pt missing/deleted |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42` | 0.3500 | 0.8108 | - | 0.7185 | - | best.pt missing/deleted |
| `qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42` | 0.5000 | 0.8157 | 0.8958 | 0.7249 | 0.8112 | ok |
| `qata_resnet0523_qata_resnet50_simple_empty_keep_decoder_seed42` | - | - | - | - | - | best.pt missing/deleted |
| `qata_resnet0523_qata_resnet50_simple_generic_keep_decoder_seed42` | 0.3500 | 0.7923 | 0.8750 | 0.6964 | 0.7778 | ok |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hh_decoder_seed42` | - | - | - | - | - | best.pt missing/deleted |
| `qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42` | 0.4500 | 0.8210 | - | 0.7324 | - | best.pt missing/deleted |
| `qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42` | 0.3500 | 0.8163 | - | 0.7259 | - | best.pt missing/deleted |
| `qata_resnet0523_qata_resnet50_simple_native_drop_ll_decoder_seed42` | - | - | - | - | - | best.pt missing/deleted |
| `qata_resnet0523_qata_resnet50_simple_native_keep_decoder_seed42` | - | - | - | - | - | final_test.json missing; incomplete run |
| `qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42` | 0.4000 | 0.8224 | 0.9000 | 0.7343 | 0.8182 | ok |
| `qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42` | 0.3500 | 0.8208 | 0.9003 | 0.7310 | 0.8187 | ok |
| `qata_resnet0523_qata_resnet50_simple_shuffle_keep_decoder_seed42` | 0.3500 | 0.7932 | 0.8766 | 0.6978 | 0.7803 | ok |
| `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42` | 0.3500 | 0.8030 | 0.8883 | 0.7093 | 0.7991 | ok |
| `qata_v3_remote_main_cxr_seed42` | 0.4500 | 0.7993 | 0.8829 | 0.7057 | 0.7903 | ok |
| `screening0506_mosmed_cxr_frozen_keep_both_seed42` | 0.5000 | 0.6737 | - | 0.5290 | - | MosMed run: checkpoint was deleted or different evaluator/dataset loader needed. |
| `screening0506_qata_cxr_frozen_keep_both_seed42` | 0.3500 | 0.8155 | 0.8937 | 0.7242 | 0.8079 | ok |
| `screening0506_qata_cxr_frozen_keep_decoder_seed42` | 0.4000 | 0.8125 | 0.8896 | 0.7198 | 0.8011 | ok |

### Global Dice Reading Notes

- If `Global Dice` is higher than `Mean Dice`, the model is likely doing better on larger masks than on small/empty-hard cases.
- If `Global Dice` is lower than `Mean Dice`, large masks or total false positives are hurting more than the average image-level score suggests.
- For paper comparison, only compare against other papers if you know whether they report per-image mean Dice or global Dice.

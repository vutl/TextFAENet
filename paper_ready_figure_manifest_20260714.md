# Paper-ready figure manifest - 2026-07-14

## Main paper

### 1. QaTa comparison using the picked model

- PDF: `generated_figures/paper_qualitative_panels/qata_picked_high_gap/fig_qata_external_qualitative.pdf`
- PNG: `generated_figures/paper_qualitative_panels/qata_picked_high_gap/fig_qata_external_qualitative.png`
- Case metadata: `generated_figures/paper_qualitative_panels/qata_picked_high_gap/fig_qata_external_qualitative_cases.json`
- Status: ready.
- Models shown: FAENet visual-only, official FMISeg, best internal ablation, and the picked ResNet50+CXR-BERT model.
- Caption disclosure: cases were selected by the largest picked-model Dice margin subject to picked-model Dice >= 0.80. They are selected success cases, not random test examples.

### 2. Same-checkpoint QaTa prompt intervention

- PDF: `generated_figures/paper_qualitative_panels/qata_picked_prompt_intervention/fig_qata_picked_prompt_intervention_compact.pdf`
- PNG: `generated_figures/paper_qualitative_panels/qata_picked_prompt_intervention/fig_qata_picked_prompt_intervention_compact.png`
- Case metadata: `generated_figures/paper_qualitative_panels/qata_picked_prompt_intervention/fig_qata_picked_prompt_intervention_compact_cases.json`
- Status: ready; compact paper layout generated on 2026-07-14.
- Checkpoint is fixed to `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42`; only native, empty, and shuffled text change.
- Caption disclosure: three cases were selected for a strong native-versus-corrupted-prompt gap. `M_s` denotes the decoder spatial-grounding mask.

### 3. Brain/Breast prompt-protocol comparison

- PDF: `generated_figures/paper_qualitative_panels/brain_breast_prompt_protocol/fig_brain_breast_prompt_protocol.pdf`
- PNG: `generated_figures/paper_qualitative_panels/brain_breast_prompt_protocol/fig_brain_breast_prompt_protocol.png`
- Case metadata: `generated_figures/paper_qualitative_panels/brain_breast_prompt_protocol/fig_brain_breast_prompt_protocol_cases.json`
- Status: ready.
- Shows MedCLIP-style versus structured prompts with prediction Dice and structured-prompt error maps.
- Do not claim structured prompts improve Brain globally; the quantitative gain is specific to Breast in the current runs.

### 4. Four-dataset qualitative segmentation

- PDF: `generated_figures/paper_qualitative_panels/cross_dataset_four/fig_cross_dataset_qata_mosmed_brain_breast.pdf`
- PNG: `generated_figures/paper_qualitative_panels/cross_dataset_four/fig_cross_dataset_qata_mosmed_brain_breast.png`
- Case metadata: `generated_figures/paper_qualitative_panels/cross_dataset_four/fig_cross_dataset_qata_mosmed_brain_breast_cases.json`
- Status: ready; now includes QaTa-COV19, MosMedData+, Brain MRI, and Breast US.
- QaTa uses the picked checkpoint. MosMed uses the recovered v9e checkpoint. Brain/Breast use the structured-prompt checkpoints.
- Caption disclosure: these are selected high-quality examples. The MosMed case is the highest-Dice case among 24 evenly spaced test candidates with foreground area >= 0.1%.

## Supplementary

### Prompt-semantics quantitative plot

- PDF: `generated_figures/paper_results_20260624/fig_qata_prompt_semantics.pdf`
- PNG: `generated_figures/paper_results_20260624/fig_qata_prompt_semantics.png`
- Status: usable in supplementary; all plotted values are mean per-image Dice.

### Raw wavelet-band intuition

- QaTa: `generated_figures/frequency_bands/qata/qata_test_01_idx1340_frequency_bands.png`
- MosMed: `generated_figures/frequency_bands/mosmed/mosmed_test_01_idx0159_frequency_bands.png`
- Brain: `generated_figures/frequency_bands/brain/brain_test_01_idx0433_frequency_bands.png`
- Breast: `generated_figures/frequency_bands/breast/breast_test_01_idx0039_frequency_bands.png`
- Status: usable only as decomposition intuition. These raw DWT panels are not evidence that learned TGFS gates specialize by band.

## Prefer tables instead of these figures

- `generated_figures/paper_results_20260624/fig_qata_frequency_ablation.pdf`: values are valid, but a table is clearer and avoids overinterpreting small single-seed differences.
- `generated_figures/paper_results_20260624/fig_qata_main_comparison.pdf`: duplicates the main result table.
- `generated_figures/paper_results_20260624/fig_qata_per_image_vs_global.pdf`: protocol explanation only, not a main result figure.

## Do not use

- Older verbose prompt panel: `generated_figures/paper_qualitative_panels/qata_picked_prompt_intervention/fig_qata_picked_prompt_intervention_strongest.pdf`.
- Older three-dataset grid: `generated_figures/paper_qualitative_panels/cross_dataset_qata_brain_breast/fig_cross_dataset_qata_brain_breast.pdf`.
- `generated_figures/current_ablation/fig_tgfs_module_schematic.png`: contains internal wording such as "What the block claims" and an implementation limitation; use the architecture/TGFS diagrams already formatted in the manuscript instead.
- Gate-statistics figure where all gates remain near 0.5; it does not support the specialization claim.

## Metric integrity

The figures above use actual retained checkpoints and per-case predictions. No score is transferred between model labels. The picked QaTa model can be the visual target and named selected architecture, but its verified aggregate metrics remain attached to its own run. Adding 2 percentage points or swapping the simple model's aggregate score into the picked-model row would make the table inconsistent with these prediction artifacts.

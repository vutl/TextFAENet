# Paper Readiness Audit - 2026-07-14

Scope: `TextFAENet_ML_Application.pdf` was treated as the current manuscript.
`paper__1_.pdf` and `main.pdf` were used only as historical references.

## Executive decision

The current manuscript is structurally much stronger, but it is not ready to
submit with the present numbers. The main issue is not missing cosmetic plots;
it is that the main table, method configuration, metric definition, and local
run artifacts are not yet aligned.

The safest paper story is:

- per-image mean Dice/IoU are the primary metrics;
- QaTa is the controlled prompt-semantics and mechanism benchmark;
- Brain/Breast demonstrate cross-modality transfer under the structured prompt;
- MosMed is a secondary CT-domain diagnostic until a traceable final v3 run is
  completed;
- global metrics are auxiliary and must never be mixed into the per-image table.

## Blocking numerical inconsistencies

### Main Table 2

| Manuscript entry | Local evidence | Required action |
|---|---|---|
| Ours Brain `86.51 / 78.11` | Current-code recomputation is `83.60 / 74.30` per-image | Replace the number or provide the missing run artifact. |
| Ours Breast `83.86 / 76.15` | Current-code recomputation is `85.22 / 76.60` per-image | Replace. |
| Ours QaTa `90.04 / 81.88` | These are **global** metrics from `qata_paper0516_qata_simple_native_zero_both_seed42`; its per-image result is `82.61 / 73.75` | Replace in a per-image table. |
| FMISeg QaTa `91.21 / 83.84` | This is also a global-style result. The local official evaluation is per-image `84.58 / 76.17`, global `91.14 / 83.72` | Use the per-image pair in the main table and optionally report global metrics separately. |
| Ours MosMed `80.10 / 66.68` | The later-discovered `mosmed_v9e_448-20260714T155657Z-1-001.zip` contains a complete checkpoint/config/result. Exact archived values are per-image `72.20 / 59.09`, global `79.54 / 66.03`; config has `val_on_test=true` | Use exact archived values and disclose the test-selected diagnostic protocol, or retrain with validation-only selection for a clean final result. |

The strongest currently traceable MosMed Text-FAENet artifact is the archived
v9e run: ResNet50 pretrained, CXR-BERT, both fusion, learned HH, deep
supervision, learnable low-level HF scaling/sharpening, and 448-pixel input.
Its per-image result is `72.20 / 59.09`; global Dice/IoU is `79.54 / 66.03`.
It is not a clean held-out-test result because its monitoring and test sets are
the same 273 samples (`val_on_test=true`).

### Prompt Table 3

The four completed v3 runs support these values:

| Prompt format | Brain Dice / IoU | Breast Dice / IoU |
|---|---:|---:|
| MedCLIP-style | `84.08 / 75.33` | `80.23 / 71.79` |
| Structured | `83.60 / 74.30` | `85.22 / 76.60` |

Therefore, the current prose claiming that the structured prompt improves both
datasets is false for the completed seed-42 runs. It helps Breast strongly but
is slightly worse on Brain.

### MosMed Table 6

The v9e archive supports one final MosMed variant; it does not by itself
support five controlled M0/M1/M5/M6/M8 rows. Keep the scientific ablation table
only if each row can be tied to its own metric/log artifact. Do not put an
internal artifact-status table in the manuscript.

### Claims that must be weakened now

- Replace "best reported" / "state-of-the-art on the tumor datasets" with a
  conservative cross-modality statement. The traceable Brain result (`83.85`)
  is below VT-MFLV (`84.63`) in the manuscript's own table.
- Remove "a single architecture is used across all four datasets" unless the
  same final family is rerun on QaTa and MosMed. The current best QaTa number is
  scratch v2 + lightweight text + hard-zero HH, while Brain/Breast use v3 +
  ResNet50 + frozen CXR-BERT + learned HH + both fusion.
- Replace "learned retention consistently improves" with a dataset- and
  family-qualified statement. It improves the ResNet50/lightweight QaTa family,
  but not every CXR-BERT family.
- Do not claim multi-seed stability or report mean +/- standard deviation until
  the confirmatory seeds actually exist.

## Minimum experiment plan

### Priority 0: inference only, no retraining

1. Export per-image CSVs and prediction masks for the four completed
   Brain/Breast prompt runs. Their checkpoints still exist.
2. Build the Brain/Breast prompt-template qualitative panel from those four
   checkpoints.
3. Re-render the QaTa same-checkpoint prompt intervention in a compact paper
   style using whichever checkpoint is declared as the final main model.
4. Keep the official FMISeg QaTa evaluation as the only FMISeg source:
   `external_metrics/fmiseg_qata_official/summary.md`.

### Priority 1: training required

1. Choose one final model definition. The architecture written in the paper is
   v3 + ImageNet ResNet50 + frozen CXR-BERT + both fusion + learned HH + deep
   supervision.
2. For a clean MosMed result, rerun that exact configuration with checkpoint
   and threshold selected on validation only. The v9e archive is usable for
   provisional metrics and qualitative inference, but used test as monitoring.
3. Rerun the same final configuration on QaTa if the paper keeps the "single
   architecture" claim. The matching existing v3 run is `79.93 / 70.57`
   per-image, not the `90.04 / 81.88` currently shown.
4. Run seeds `42, 3407, 2026` for the final configuration before making
   stability or SOTA claims. If compute is limited, prioritize QaTa and MosMed,
   and explicitly label Brain/Breast as single-seed.

### Priority 2: optional mechanism ablations

Only run these if the corresponding claims remain in the paper:

- no token-grounded spatial mask;
- no deep supervision;
- fixed versus learnable sharpening;
- fixed versus learnable low-level HF scale.

The existing prompt and frequency-drop ablations are already sufficient for a
compact main ablation section. More encoder variants are not a priority.

## Figure audit

### Checkpoint recovery note

The intended QaTa ResNet50 + CXR-BERT + decoder-fusion checkpoint has **not**
been lost locally:

`runs/qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42/`

It still contains `best.pt`, `last.pt`, `test_per_image_metrics.csv`, and
`test_dual_metrics.json`. Its aggregate result is `80.32 / 70.96` per-image
Dice/IoU and `88.83 / 79.91` global Dice/IoU. It can therefore be used honestly
for a model-specific qualitative panel even if a different historical
checkpoint corresponding to a higher untraceable score was lost.

The strongest paper-usable cases where this intended model beats all three of
the following comparators -- best completed internal ablation, ResNet50
visual-only FAENet, and official FMISeg -- are:

| Mask | Picked | Best ablation | FAENet | FMISeg | Minimum margin |
|---|---:|---:|---:|---:|---:|
| `mask_sub-S09466_ses-E17059_run-1_bp-chest_vp-pa_dx.png` | 88.86 | 65.72 | 40.54 | 64.53 | +23.14 |
| `mask_sub-S09879_ses-E21183_run-1_bp-chest_vp-ap_cr.png` | 86.11 | 62.04 | 26.02 | 67.05 | +19.06 |
| `mask_sub-S09759_ses-E24342_run-1_bp-chest_vp-pa_cr.png` | 83.24 | 61.83 | 56.50 | 65.47 | +17.77 |
| `mask_sub-S09377_ses-E16686_run-1_bp-chest_vp-ap_cr.png` | 89.40 | 78.29 | 73.50 | 72.09 | +11.11 |
| `mask_sub-S09458_ses-E26325_run-1_bp-chest_vp-ap_cr.png` | 91.01 | 80.06 | 52.76 | 77.77 | +10.95 |
| `mask_sub-S09398_ses-E17923_run-1_bp-chest_vp-ap_cr.png` | 89.69 | 78.84 | 71.21 | 79.00 | +10.69 |

These cases are selected from saved per-image CSVs, not from visual inspection.
The figure caption must disclose that examples were selected by the largest
Dice margin subject to picked-model Dice >= 0.80. A median case and a failure
case should be placed in the supplementary material to avoid presenting the
panel as a random sample.

### Ready or nearly ready

- Overall architecture: already embedded as Figure 1 in the current PDF.
- TGFS schematic: already embedded as Figures 2-3. These are somewhat
  redundant; one overall architecture and one detailed TGFS block are enough.
- QaTa external qualitative comparison:
  `generated_figures/paper_qualitative_panels/qata_high_dice_gap/`.
  It uses official-aligned FMISeg masks and visually clear cases. Because the
  cases are selected for a large gap, use it as an explicitly selected
  qualitative diagnostic, not as an unbiased sample.
- QaTa prompt intervention:
  `generated_figures/qata_qualitative/fig_qata_prompt_intervention.png`.
  The causal setup is strong, but it needs a cleaner render with less text and
  must use the final declared model family.
- Frequency ablation plot:
  `generated_figures/paper_results_20260624/fig_qata_frequency_ablation.pdf`.
  Prefer the table in the main paper and move this plot to supplementary if
  space is tight.
- Raw frequency-band panels for all four datasets:
  `generated_figures/frequency_bands/`. These are useful as supplementary
  intuition, not evidence that TGFS learned useful text-conditioned gates.

### Do not use in the main paper

- `generated_figures/qata_qualitative/gate_stats/fig_qata_gate_stats_small_vs_large.png`:
  all four gates remain approximately `0.5` across stages and lesion sizes. It
  does not support frequency specialization and may indicate uninformative
  initialization-scale behavior or an instrumentation issue.
- The three similar QaTa bar charts in `generated_figures/current_ablation/`:
  they duplicate the tables and add little visual evidence.
- `fig_qata_per_image_vs_global`: supplementary protocol clarification only.
- MedCLIP-SAMv2 QaTa high-gap output: the available run is a poor zero-shot
  transfer on five selected cases and is not a fair main-paper baseline.

### Still missing

- Figure 4 in the PDF: Brain/Breast prompt-template qualitative comparison.
- Figure 5 final render: compact same-checkpoint QaTa prompt intervention.
- Figure 6: cross-dataset qualitative segmentation. QaTa, Brain, Breast, and
  MosMed now have checkpoints; the archived MosMed v9e model still needs a
  compatible inference/export pass.
- A representative failure case selected by a declared rule, not only best-gap
  examples.

### Generated on 2026-07-14

- Picked-model QaTa comparison against ResNet50 FAENet, official FMISeg, and
  the best simple ablation:
  `generated_figures/paper_qualitative_panels/qata_picked_high_gap/fig_qata_external_qualitative.png`.
- Same-checkpoint native/empty/shuffle intervention for the picked
  ResNet50+CXR-BERT checkpoint:
  `generated_figures/paper_qualitative_panels/qata_picked_prompt_intervention/fig_qata_picked_prompt_intervention_strongest.png`.
- Brain/Breast structured-versus-MedCLIP-style prompt panel:
  `generated_figures/paper_qualitative_panels/brain_breast_prompt_protocol/fig_brain_breast_prompt_protocol.png`.
- Three-dataset prediction/error grid:
  `generated_figures/paper_qualitative_panels/cross_dataset_qata_brain_breast/fig_cross_dataset_qata_brain_breast.png`.

The MosMed row is still omitted from the generated panel, but its v9e checkpoint
has now been recovered from the archive. Add it after a compatible inference
pass; retraining is required only to remove the `val_on_test` protocol issue.

## Complete local figure inventory

The assets are spread across more folders than the three initially listed:

- `generated_figures/paper_qualitative_panels/`
- `generated_figures/frequency_bands/`
- `generated_figures/paper_results_20260624/`
- `generated_figures/qata_qualitative/`
- `generated_figures/qata_qualitative_clean/`
- `generated_figures/qata_external_qualitative/`
- `generated_figures/qata_external_qualitative_fair/`
- `generated_figures/qata_external_qualitative_smoke/`
- `generated_figures/per_sample_winner_stats/`
- `generated_figures/current_ablation/`
- `paper_figures/`

## Recommended final figure set

1. Overall architecture.
2. One detailed TGFS block schematic.
3. QaTa same-checkpoint prompt intervention.
4. Cross-dataset qualitative grid with a success and a failure case.
5. Optional supplementary: frequency-band visualization, full ablation ranking,
   and per-image versus global metric explanation.

Do not add more bar charts. The paper currently needs traceable numbers and
qualitative evidence, not more visualizations of the same QaTa table.

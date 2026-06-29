# Paper Figure Plan - 2026-06-24

This note tracks the figure assets for the current LFAENet-TGFS paper draft.

## What Similar Papers Usually Show

For papers in this area, the recurring figure pattern is:

1. Overall pipeline / architecture schematic.
2. Module-level schematic for the new fusion/attention block.
3. Qualitative segmentation grid: input, GT, predictions, and sometimes error
   maps or attention/saliency maps.
4. Compact ablation plots/tables for module contribution.
5. Supplementary ranking/diagnostic plots.

Relevant references checked:

- FMISeg frames the frequency-language contribution around frequency-domain
  visual features and decoder-side language interaction.
- MedCLIP-SAMv2 frames the promptable model around text prompts, visual prompt
  generation, and qualitative segmentation examples.
- FAENet-style work motivates architecture/module schematic plus qualitative
  segmentation and ablation evidence.

## Existing Local Figures

### Current ablation figures

Folder: `generated_figures/current_ablation`

| File | Status | Suggested use |
|---|---|---|
| `fig_tgfs_module_schematic.png` | Ready but should be polished for final camera-ready style. | Method figure / TGFS block. |
| `fig_qata_ablation_summary.png` | Ready. | Supplementary or compressed ablation visual. |
| `fig_qata_full_ablation_ranking.png` | Ready. | Supplementary only. |
| `fig_qata_diagnostic_panels.png` | Ready. | Main or supplementary ablation figure. |
| `qata_ablation_summary.csv` | Ready. | Source table for plots. |

### Qualitative/debug figures

Folder: `generated_figures/qata_qualitative`

| File | Status | Suggested use |
|---|---|---|
| `fig_qata_qualitative_segmentation.png` | Ready. | Main qualitative result for QaTa. |
| `fig_qata_prompt_intervention.png` | Ready and important. | Main prompt-semantics figure. |
| `gate_stats/fig_qata_gate_stats_small_vs_large.png` | Ready. | Supplementary interpretability figure. |
| `gate_stats/gate_stats_summary.csv` | Ready. | Source for gate plot. |

### Clean paper-style qualitative figures

Folder: `generated_figures/qata_qualitative_clean`

| File | Status | Suggested use |
|---|---|---|
| `fig_qata_qualitative_clean_selected.pdf/png` | Ready and cleaner than older qualitative grids. | Main qualitative QaTa figure. |
| `fig_qata_qualitative_clean_best.pdf/png` | Ready, but cherry-picked toward high Dice / large lesions. | Supplementary or backup. |
| `fig_qata_qualitative_clean_mixed.pdf/png` | Ready, includes a hard/failure case. | Supplementary or limitations. |
| `fig_qata_qualitative_clean_*_cases.json` | Ready. | Metadata with prompts, filenames, Dice. |

Script:

```powershell
D:\anaconda3\python.exe -u scripts\make_qata_paper_qualitative_clean.py --sample-ids 1182 240 1069 853 --tile 164
```

### Older paper figures

Folder: `paper_figures`

| File | Status | Suggested use |
|---|---|---|
| `fig2_qata_dataset_overview.png` | Existing. | Dataset/protocol figure if space allows. |
| `fig3_qata_qualitative.png` | Existing. | Superseded by newer qualitative grid unless visually better. |

## Newly Generated Result Plots

Folder: `generated_figures/paper_results_20260624`

All figures are saved as both `.png` and `.pdf`.

| File | Source | Suggested use |
|---|---|---|
| `fig_qata_main_comparison.pdf/png` | Completed QaTa run folders. | Main or supplementary result plot: visual-only vs text-guided TGFS. |
| `fig_qata_prompt_semantics.pdf/png` | Completed native/empty/shuffle/generic runs. | Strong candidate for main ablation figure. |
| `fig_qata_frequency_ablation.pdf/png` | HH-prior + sub-band drop runs. | Main/supplementary frequency mechanism figure. |
| `fig_qata_per_image_vs_global.pdf/png` | `qata_dual_metrics_resnet_cxr_vs_best_ablation.json`. | Supplementary metric-protocol clarification. |
| `qata_main_comparison.csv` | Generated source table. | Reproducibility/source. |
| `qata_prompt_semantics.csv` | Generated source table. | Reproducibility/source. |
| `qata_frequency_ablation.csv` | Generated source table. | Reproducibility/source. |

Script:

```powershell
D:\anaconda3\python.exe -u scripts\make_paper_result_plots.py
```

## Recommended Main-Paper Figure Set

### Fig. 1 - Overall Architecture

Status: needs final drawing.

Should show:

- Image encoder.
- Text encoder.
- Frequency-aware encoder/decoder path.
- TGFS placement.
- Validation/test protocol not needed in this figure.

This should not be a generic UNet diagram. It should make clear that our
contribution is text-guided frequency/sub-band selection.

### Fig. 2 - TGFS Block

Candidate file:

- `generated_figures/current_ablation/fig_tgfs_module_schematic.png`

Need polish:

- Remove casual/internal labels.
- Use concise academic labels: DWT, sub-band gates, token-grounded spatial mask,
  iDWT, residual refinement.
- Avoid limitation wording inside the figure.

### Fig. 3 - QaTa Prompt Intervention

Candidate file:

- `generated_figures/qata_qualitative/fig_qata_prompt_intervention.png`

This is one of the strongest figures because it changes the prompt while keeping
the image/checkpoint fixed. It directly supports the claim that text semantics
affect segmentation.

### Fig. 4 - QaTa Qualitative Segmentation

Candidate file:

- `generated_figures/qata_qualitative_clean/fig_qata_qualitative_clean_selected.pdf`

Use this before bar charts if the main paper is short on figure slots. It follows
the standard segmentation-paper layout: input, ground truth, baseline
prediction, proposed prediction, and error map.

### Fig. 5 - QaTa Ablation / Prompt Semantics

Candidate file:

- `generated_figures/paper_results_20260624/fig_qata_prompt_semantics.pdf`

Use this if there is room for a plot. Otherwise keep the table in the main paper
and move the plot to supplementary.

### Fig. 6 - Frequency Mechanism

Candidates:

- `generated_figures/paper_results_20260624/fig_qata_frequency_ablation.pdf`
- `generated_figures/qata_qualitative/gate_stats/fig_qata_gate_stats_small_vs_large.png`

Best story:

- Main paper: frequency ablation plot.
- Supplementary: gate statistics small-vs-large.

## Figures That Need More Work

### Cross-Dataset Qualitative Segmentation

Not ready.

Needs completed checkpoints/results for:

- Text-FAENet Brain structured prompt.
- Text-FAENet Breast structured prompt.
- Ideally FMISeg Brain structured prompt.

Suggested layout:

```text
Dataset / Case | Input | GT | MedCLIP-SAMv2 | FMISeg | Ours | Error map
```

Use one representative case each for Brain, Breast, QaTa, and MosMed if
MosMed remains in the main story.

### Brain/Breast Prompt-Template Diagnostic

Partially ready.

Needs:

- Text-FAENet Brain/Breast structured-prompt full runs.
- Text-FAENet Brain/Breast MedCLIP-style prompt full runs.

Suggested layout:

```text
Input | GT | Ours + MedCLIP-style prompt | Ours + structured prompt | Error maps
```

MedCLIP-SAMv2 local output can be included only with a protocol caveat because
the local Brain mask output does not reproduce the currently cited
MedCLIP-SAMv2 paper number.

## Supplementary Figures

Good supplementary candidates:

1. `fig_qata_full_ablation_ranking.png`
2. `fig_qata_per_image_vs_global.pdf`
3. `gate_stats/fig_qata_gate_stats_small_vs_large.png`
4. More qualitative examples from `generated_figures/qata_qualitative`

## Immediate Next Figure Task

The most valuable missing visual is not another bar chart. It is a polished
cross-dataset qualitative grid after Brain/Breast full runs finish.

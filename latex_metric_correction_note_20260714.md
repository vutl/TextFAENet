# LaTeX Metric Correction Note - 2026-07-14

This note is intended to be pasted into the web ChatGPT session that edits the
LaTeX manuscript. It separates the selected main model from the highest-scoring
ablation and prevents per-image and global metrics from being mixed.

## Missing experiment matrix

The completed local runs are already sufficient for the QaTa single-seed
ablation tables and the Brain/Breast prompt-protocol table. The remaining gaps
depend on which claims and tables are retained in the manuscript.

### Submission blockers under the current four-dataset narrative

| Missing result | Why it is missing | Minimum defensible action |
|---|---|---|
| Clean MosMed result for the declared final protocol | `mosmed_v9e_448-20260714T155657Z-1-001.zip`, now extracted to `runs/mosmed_v9e_448`, contains a complete v9e artifact and checkpoint, but its config has `val_on_test=true`; the 273 monitoring samples are the same 273 samples reported as test | Use the archived result as a traceable provisional/diagnostic result, or retrain with checkpoint and threshold selected on a separate validation split for a clean final-test claim |
| MosMed qualitative row | The v9e checkpoint exists in the archive but predictions have not yet been exported locally | Reconstruct the archived v9e model from its config, run inference, and add selected MosMed cases; no new training is required merely to make this panel |
| QaTa result for the exact unified Brain/Breast architecture | Brain/Breast use v3 + ResNet50 + CXR-BERT + both fusion + learned HH, whereas the picked QaTa row uses decoder fusion + keep HH and the highest QaTa row uses the scratch/lightweight family | Either rerun the exact unified v3 configuration on QaTa, or delete the claim that one identical architecture/configuration is used on all four datasets |

The archive means MosMed is no longer missing because of a lost checkpoint.
New MosMed training is required only if the submission must follow a clean
validation-selected test protocol. The QaTa unified run is avoidable by
describing the reported dataset-specific settings honestly.

### Required only if the corresponding paper content is retained

| Paper content or claim | Missing experiments | Alternative without training |
|---|---|---|
| MosMed M0/M1/M5/M6/M8 ablation table | The v9e archive supports one final variant, not five independently controlled M0/M1/M5/M6/M8 rows | Keep the scientific ablation table only if the metric/log artifact for each row is located; never replace it with an artifact-status table in the paper |
| Mean $\pm$ standard deviation / stability claim | Additional seeds `3407` and `2026` for every reported final model; MosMed also still needs seed `42` | State clearly that all local Text-FAENet results are single-seed and remove stability claims |
| Mechanism claim that the spatial mask is necessary | No-TGFS-spatial-mask ablation | Remove or soften the causal claim |
| Deep-supervision contribution | Matched run with deep supervision disabled | Describe deep supervision as an implementation choice rather than a validated contribution |
| Learnable sharpening contribution | Fixed versus learnable spatial-sharpening pair | Remove the improvement claim |
| Learnable low-level HF scaling contribution | Fixed versus learnable low-level-HF-scale pair | Remove the improvement claim |
| Fair FMISeg comparison on Brain structured prompts | Brain-trained FMISeg run with the same split and prompt protocol | Use only published FMISeg values with a protocol caveat, or omit that cell |
| MedCLIP-SAMv2 under our structured Brain/Breast prompts | MedCLIP-SAMv2 rerun using our prompt files | Keep the local MedCLIP-style result explicitly labelled by its own prompt protocol |

### Already complete; do not rerun merely to fill the paper

- QaTa visual-only baselines with scratch and ImageNet ResNet50 encoders.
- QaTa selected ResNet50+CXR-BERT decoder model and its dual metrics.
- QaTa lightweight-text prompt semantics: native, empty, shuffled, and generic.
- QaTa CXR-BERT prompt semantics: native, empty, and shuffled.
- QaTa decoder-only versus encoder+decoder fusion comparisons.
- QaTa HH keep/zero/learned and drop-one-band LL/LH/HL/HH diagnostics.
- Brain and Breast structured-prompt v3 runs.
- Brain and Breast MedCLIP-style-prompt v3 runs.
- ClinicalBERT and revised BERT/cross-attention QaTa diagnostics. These runs are
  useful for discussion, but none beats the traceable lightweight-text best
  run and they are not required in the main table.

### Smallest credible completion plan

1. Use the archived v9e MosMed result as a provisional traceable row, but do
   not call it a clean held-out final test because `val_on_test=true`.
2. Export MosMed predictions from the archived checkpoint and add them to the
   cross-dataset figure.
3. Either run the same exact final configuration on QaTa or explicitly report
   dataset-specific final settings and remove the single-identical-model claim.
4. Keep all results labelled `single seed`; do not run extra seeds unless the
   manuscript must report mean $\pm$ standard deviation.
5. If a clean protocol is mandatory, retrain only the final MosMed variant with
   validation-only checkpoint/threshold selection. Do not place an internal
   artifact-status table in the manuscript.

## Correction to `experiments_filled_metric_corrected.tex`

The generated LaTeX incorrectly converts the internal artifact audit into a
paper subsection titled `Secondary MosMedData+ Status` and a table whose rows
say `Retrain required` and `Not architecture-matched`. Delete that entire
status table. Provenance tracking belongs in project notes, not in the Results
section of a scientific manuscript.

Apply these concrete changes:

1. In the selected LFAENet-TGFS row of the main four-dataset benchmark, fill the
   MosMed columns with `72.20` Dice and `59.09` IoU. These are the per-image
   metrics in the recovered v9e artifact.
2. Do not use `79.54/66.03` in that main table because those are global
   pixel-pooled metrics. They may be reported in a separately labelled
   supplementary global-metric table.
3. Replace the `Secondary MosMedData+ Status` subsection with a normal result
   paragraph describing the v9e configuration and result. Do not discuss lost
   checkpoints, provenance, or retraining status in the manuscript body.
4. Do not state that no MosMed checkpoint exists. The recovered archive contains
   `best.pt` at epoch 62, threshold 0.55, plus its config, history, and logs.
5. Do not restore the M0--M8 table using the v9e score repeatedly. The archive
   is evidence for one final variant only. Keep M0--M8 only if the actual score
   for each controlled variant is available from its own run record.
6. The archived configuration used the same 273 cases as monitoring and test
   (`val_on_test=true`). This protocol issue must not be hidden. For a clean
   final-test claim, rerun with validation-only checkpoint/threshold selection;
   otherwise qualify the v9e result as a diagnostic evaluation.

Suggested replacement paragraph:

> On MosMedData+, the v9e configuration uses an ImageNet-pretrained ResNet50,
> CXR-BERT conditioning, encoder--decoder fusion, learned HH retention, deep
> supervision, and 448-pixel inputs. It obtains 72.20% mean per-image Dice and
> 59.09% mean per-image IoU with test-time augmentation. Global pixel-pooled
> Dice and IoU are 79.54% and 66.03%, respectively, and are reported separately
> because they use a different aggregation rule.

## Final gap checklist after converting the manuscript to per-image metrics

### Numerically complete local Text-FAENet tables

The following sections no longer have missing per-image Dice/IoU cells:

- Selected-model main row: Brain `83.60/74.30`, Breast `85.22/76.60`,
  QaTa `80.32/70.96`, and archived MosMed v9e `72.20/59.09`.
- Best-observed QaTa diagnostic: `82.61/73.75`.
- Brain/Breast structured-versus-MedCLIP-style prompt table.
- QaTa visual-only, text-encoder, prompt-semantics, fusion-locus,
  HH keep/zero/learned, and drop-one-band ablations.

The drop-one-band rows do not need global metrics when the ablation table is
defined as per-image. A dash in an auxiliary global-metric table is acceptable.

### Items still missing or requiring a manuscript decision

1. **Clean MosMed final-test protocol.** The v9e numbers exist, but
   `val_on_test=true`; checkpoint and threshold selection used the same 273
   cases reported as test. Retraining is required only if a clean held-out-test
   claim is required.
2. **M0--M8 MosMed rows.** The recovered v9e archive supplies one final variant,
   not the individual M0/M1/M5/M6/M8 scores. Locate those row-level results or
   omit that ablation table; do not replace it with a status table.
3. **Unified-configuration claim.** QaTa selected uses decoder fusion + HH keep,
   while Brain/Breast and MosMed use later v3 settings with both fusion and
   learned priors. Either remove the claim of one identical configuration or
   rerun the exact same final configuration on QaTa.
4. **Multi-seed statistics.** No mean $\pm$ standard deviation is supported.
   Keep all Text-FAENet results labelled single-seed unless seeds 3407 and 2026
   are completed.
5. **External baseline aggregation.** Published UNet/nnUNet/TransUNet/VT-MFLV/
   STPNet/MedCLIP-SAMv2 values cannot be converted mathematically from global to
   per-image metrics. Each row needs either an explicitly per-image source,
   local predictions for recomputation, or a protocol footnote. Do not assume
   every number copied from a paper uses the same aggregation.
6. **Optional mechanism claims.** No-spatial-mask, no-deep-supervision,
   fixed-versus-learnable sharpening, and fixed-versus-learnable low-level HF
   scale remain unrun. Remove or soften the corresponding causal claims if
   these experiments are not added.
7. **Frequency conclusion.** In the actual per-image drop-band results, no-drop
   is `81.60/72.53` and drop-HL is `82.10/73.24`. The paper cannot state that
   no-drop is numerically highest; it may state that the differences are small
   single-seed diagnostics and that retaining all bands avoids a fixed removal
   prior.

### Figures after the MosMed checkpoint recovery

No qualitative figure is numerically missing. The four-dataset panel now exists
at `generated_figures/paper_qualitative_panels/cross_dataset_four/fig_cross_dataset_qata_mosmed_brain_breast.pdf`.
The complete approved figure list is in `paper_ready_figure_manifest_20260714.md`.

## Context from the author for the LaTeX editor

The paper is under a tight deadline and the local experiment history is not
clean. The author had selected the ResNet50 + CXR-BERT + decoder-TGFS model as
the intended paper model because it is the architecture that best matches the
method narrative: FAENet-style ResNet visual encoding plus domain-specific
clinical text conditioning in the TGFS decoder. A historically stronger result
for the intended/picked direction may have existed, but the corresponding
checkpoint and complete provenance were lost. It must not be treated as a
currently reproducible result unless a surviving log, JSON, CSV, or checkpoint
is found.

The highest **currently traceable** QaTa run is instead a different
configuration: scratch visual encoder + lightweight/simple text encoder + both
fusion + hard-zero HH. This creates a presentation conflict:

- the picked ResNet50+CXR-BERT model is more aligned with the architecture the
  author wants to present;
- the simple model has the highest surviving numerical result;
- several qualitative figures therefore use the picked checkpoint and select
  cases where it genuinely outperforms the internal ablation and external
  baselines;
- the quantitative table must still preserve which configuration produced each
  score.

The author asked to "swap" the highest simple result and the picked model in
the paper presentation, except for the frequency ablations. Interpret this as a
request to reorganize the paper narrative and table placement, **not** as
permission to relabel one model's measurements as another model's output. Two
defensible presentation options are available:

1. Keep ResNet50+CXR-BERT as the selected main architecture and report its
   verified score (`80.32/70.96` per-image Dice/IoU). Report the simple model
   separately as the "best observed diagnostic configuration"
   (`82.61/73.75`).
2. Promote `82.61/73.75` to the main result, but then rename the main row and
   revise the method/implementation text so it explicitly says that the
   reported QaTa model uses the lightweight text encoder, scratch visual
   family, both fusion, and zero HH.

Do not create a hybrid row that combines the ResNet50+CXR-BERT label with the
simple model's score. If the author temporarily keeps an unreproduced historical
number in a private draft, mark it clearly as `UNVERIFIED / RETRAIN REQUIRED`
and do not include it in a submission PDF.

### Why frequency rows are an exception to narrative reordering

The frequency ablation rows are mechanistic experiments. Their meaning depends
on the exact shared family and on changing only HH handling or one wavelet band.
They must remain attached to the actual ResNet50/lightweight or scratch family
that generated them. Moving their numbers between model labels would destroy
the controlled comparison. Keep these rows exactly as listed later in this
note, even if the main-table narrative is reorganized.

### Metric context requested by the author

The manuscript should use **per-image mean Dice and per-image mean IoU** as the
main metrics because that is the intended evaluation protocol. Some older
tables and paper-source values use global/pixel-pooled Dice and IoU. This caused
numbers around `90.xx` to be placed next to per-image values around `80.xx`.
These are different aggregation methods, not a real ten-point model gain.

For every table edited from this note:

- use per-image Dice/IoU in the main paper;
- place global Dice/IoU only in a clearly labelled supplementary table;
- do not compare a published baseline's global number against our per-image
  number without an explicit protocol warning;
- use `-` for a result that cannot be traced or reproduced;
- retain the run/configuration identity in comments in the LaTeX source.

### Figure context

The new qualitative figures were intentionally generated from surviving
checkpoints rather than from the unverified historical result. The picked QaTa
panel selects cases by a declared per-image Dice-margin rule, so it is a
best-case qualitative diagnostic rather than a random test sample. The caption
must state this selection rule. A median case or failure case should remain in
the supplementary material.

Use these completed assets to replace the current placeholders:

- Picked QaTa versus FAENet, official FMISeg, and best internal ablation:
  `generated_figures/paper_qualitative_panels/qata_picked_high_gap/fig_qata_external_qualitative.pdf`.
- Same-checkpoint prompt intervention for the picked QaTa model:
  `generated_figures/paper_qualitative_panels/qata_picked_prompt_intervention/fig_qata_picked_prompt_intervention_strongest.pdf`.
- Brain/Breast prompt-format comparison:
  `generated_figures/paper_qualitative_panels/brain_breast_prompt_protocol/fig_brain_breast_prompt_protocol.pdf`.
- Current cross-dataset panel for QaTa, Brain MRI, and Breast US:
  `generated_figures/paper_qualitative_panels/cross_dataset_qata_brain_breast/fig_cross_dataset_qata_brain_breast.pdf`.

MosMed is currently absent from the cross-dataset panel, but the checkpoint was
later found in `mosmed_v9e_448-20260714T155657Z-1-001.zip`. The MosMed row can
therefore be generated by inference after reconstructing the archived v9e
configuration; it does not require retraining solely for visualization.

### Concrete instructions for the web ChatGPT LaTeX rewrite

- Treat this file as the metric/provenance source of truth, not the current PDF
  numbers.
- Replace all current main-table QaTa values with per-image values.
- Present "selected main model" and "best observed configuration" as separate
  rows or separate paragraphs.
- Update Brain/Breast prompt-table values using the 2026-07-14 recomputation.
- Replace MosMed `80.10/66.68` with the exact archived values unless another
  artifact supporting that exact pair is found. Do not replace the M0--M8
  scientific table with an artifact-status table; retain only individually
  traceable ablation rows.
- Do not write that structured prompts improve Brain; they improve Breast but
  are slightly worse on Brain in the current runs.
- Do not claim a single identical architecture across four datasets while the
  reported QaTa best result comes from the simple scratch family.
- Keep frequency-prior and drop-band numbers bound to their original family.
- Replace Figure 4--6 placeholders with the assets above; Figure 6 currently
  covers three datasets and must be captioned accordingly.
- Mark all reported Text-FAENet numbers as single-seed unless three-seed results
  are later supplied.

## Non-negotiable reporting rule

- Primary paper metric: arithmetic mean of per-image Dice and per-image IoU.
- Global Dice/IoU may appear only in a separately labelled auxiliary table.
- Never place a global score in a column labelled Dice/IoU if the caption says
  metrics are averaged per image.
- Keep every score attached to the run that produced it. Do not assign the
  lightweight/simple model's score to the selected ResNet50+CXR-BERT model.

## Important correction to the current PDF

The current controlled QaTa ablation Tables 4 and 5 are already mostly
**per-image** values. The main inconsistency is Table 2, where the QaTa entry
`90.04 / 81.88` is global Dice/IoU even though the protocol says per-image.

## Selected model versus best observed model

Use two explicit labels:

| Paper label | Exact run/configuration | Per-image Dice | Per-image IoU | Global Dice | Global IoU |
|---|---|---:|---:|---:|---:|
| Selected main model | `qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42`; ResNet50 + frozen CXR-BERT + decoder TGFS + HH keep | 80.32 | 70.96 | 88.83 | 79.91 |
| Best observed QaTa configuration | `qata_paper0516_qata_simple_native_zero_both_seed42`; scratch visual encoder + lightweight text + both fusion + HH zero | 82.61 | 73.75 | 90.04 | 81.88 |

Recommended wording: "The selected architecture obtains 80.32/70.96
per-image Dice/IoU. A lightweight-text diagnostic configuration reaches the
highest observed single-run QaTa result of 82.61/73.75."

Do not swap these numerical rows. It is acceptable to discuss the simple model
as an ablation that outperforms the selected CXR-BERT setting.

## Correct main-table values currently supported by local artifacts

| Dataset / model | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Status |
|---|---:|---:|---:|---:|---|
| Brain, selected v3 structured prompt | 83.60 | 74.30 | 83.65 | 71.89 | Recomputed 2026-07-14, checkpoint retained |
| Breast, selected v3 structured prompt | 85.22 | 76.60 | 87.84 | 78.31 | Recomputed 2026-07-14, checkpoint retained |
| QaTa, selected ResNet50+CXR-BERT model | 80.32 | 70.96 | 88.83 | 79.91 | Complete, checkpoint retained |
| QaTa, best observed simple ablation | 82.61 | 73.75 | 90.04 | 81.88 | Complete, checkpoint retained |
| MosMed, archived v9e model | 72.20 | 59.09 | 79.54 | 66.03 | Complete archive and checkpoint; `val_on_test=true`, so test-selected diagnostic rather than clean held-out final |
| MosMed, old v2 CXR-BERT diagnostic | 67.37 | 52.90 | - | - | Not architecture-matched; no checkpoint |

The manuscript's current Brain `86.51/78.11` and Breast `83.86/76.15` entries
are not supported by the completed local artifacts inspected on 2026-07-14.
The MosMed `80.10/66.68` pair is close to, but not exactly supported by, the
v9e archive: it reports per-image `72.20/59.09` and global `79.54/66.03`.

The recomputed Brain/Breast values above come from
`brain_breast_per_image_eval_20260714.json` and the newly written per-image
CSVs. They differ slightly from older `final_test.json` summaries, so the
recomputed values should be preferred for a table generated from the current
codebase.

## Correct Brain/Breast prompt-protocol table

| Model prompt protocol | Brain Dice | Brain IoU | Breast Dice | Breast IoU |
|---|---:|---:|---:|---:|
| LFAENet-TGFS, MedCLIP-style prompt | 84.08 | 75.33 | 80.23 | 71.79 |
| LFAENet-TGFS, structured prompt | 83.60 | 74.30 | 85.22 | 76.60 |

Do not claim that structured prompts improve both datasets. They improve
Breast substantially but are slightly worse on Brain in the completed seed-42
runs.

## Per-image QaTa ablation values

### A. Visual-only versus text-guided decoding

| Variant | Dice | IoU |
|---|---:|---:|
| FAENet visual-only, scratch | 77.57 | 67.66 |
| FAENet visual-only, ResNet50 | 77.94 | 67.87 |
| ResNet50 + CXR-BERT + decoder TGFS | 80.32 | 70.96 |
| ResNet50 + lightweight text + learned HH + decoder TGFS | 81.99 | 72.97 |

### B. Prompt semantics

| Text family | Prompt | Dice | IoU |
|---|---|---:|---:|
| Lightweight | Native | 82.13 | 73.07 |
| Lightweight | Empty | 76.59 | 66.68 |
| Lightweight | Shuffled | 75.86 | 65.80 |
| Lightweight | Generic | 75.99 | 65.75 |
| Frozen CXR-BERT | Native | 81.54 | 72.42 |
| Frozen CXR-BERT | Empty | 76.41 | 66.48 |
| Frozen CXR-BERT | Shuffled | 76.42 | 66.24 |

### C. Fusion locus

| Text family | Fusion | Dice | IoU |
|---|---|---:|---:|
| Lightweight | Decoder | 81.64 | 72.44 |
| Lightweight | Encoder + decoder | 82.13 | 73.07 |
| Frozen CXR-BERT | Decoder | 81.27 | 72.01 |
| Frozen CXR-BERT | Encoder + decoder | 81.54 | 72.42 |

### D. Frequency-prior handling, ResNet50/lightweight family

| HH handling | Dice | IoU |
|---|---:|---:|
| Keep | 81.60 | 72.53 |
| Zero | 82.09 | 73.12 |
| Learned | 81.99 | 72.97 |

### E. Drop-one-band diagnostics

| Variant | Dice | IoU | Global available? |
|---|---:|---:|---|
| Full LL/LH/HL/HH | 81.60 | 72.53 | Yes |
| Without LL | 81.08 | 71.85 | No |
| Without LH | 81.63 | 72.59 | No |
| Without HL | 82.10 | 73.24 | No |
| Without HH | 81.96 | 72.85 | No |

These frequency rows should remain attached to their actual ablation family.
They must not be relabelled as CXR-BERT or as the selected main model.

## FMISeg QaTa correction

Official local evaluation:

| Metric aggregation | Dice | IoU |
|---|---:|---:|
| Per-image mean | 84.58 | 76.17 |
| Global/pixel-pooled | 91.14 | 83.72 |

Use the per-image row in the main table. The old legacy wrapper producing about
`0.51` Dice is invalid and must not be cited.

## Suggested LaTeX table captions

Main table:

> Dice and IoU are computed independently for each test image and then averaged
> over the test set. Global pixel-pooled scores are reported separately in the
> supplementary material. All LFAENet-TGFS rows are single-seed results unless
> otherwise stated.

Ablation table:

> Controlled QaTa-COV19 ablations reported as mean per-image Dice and IoU. Rows
> from different architecture families are grouped explicitly and are not used
> as pairwise controls unless the visual encoder, text encoder, fusion locus,
> and frequency prior match.

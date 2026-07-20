# Tables 2--5 metric fill map

Scope: only identify missing cells and per-image metric replacements in
`TextFAENet_ML_Application.pdf`.

## Aggregation clarification

The current manuscript text says all tables use mean per-image metrics, but the
numbers do not follow one aggregation system:

- The QaTa/MosMed baseline block in Table 2 is copied from the FMISeg comparison
  table. FMISeg's published QaTa `91.21/83.84` matches pixel-pooled/global
  evaluation (our official local reproduction is global `91.14/83.72`), not
  its mean per-image result (`84.58/76.17`). The other pulmonary baseline rows
  copied from that same table should therefore be treated as source-reported/
  FMISeg-protocol values unless their prediction masks are recomputed.
- LFAENet-TGFS QaTa `90.04/81.88` in Table 2 is also global.
- Tables 4 and 5 are not global. `train_qata.py::batch_metrics` computes Dice
  and IoU independently along the image dimension and averages them. The old
  trainer then averages batch means, so the values are per-image-style with a
  very small last-batch weighting discrepancy. Exact test-wide arithmetic
  means from the recovered per-image CSVs differ only slightly.

Consequently, "only MedCLIP-SAMv2/FMISeg cells are missing" is true for blank
cells, but not sufficient to claim that Table 2 and Tables 4--5 already use one
identical aggregation protocol.

Two defensible choices exist:

1. Keep Table 2 as source-reported/global for the pulmonary benchmarks and
   label it explicitly; label Tables 4--5 as mean per-image ablations.
2. Make Table 2 strictly per-image, which requires recomputing every baseline
   row from test predictions. Published global pairs cannot be converted into
   per-image pairs algebraically.

## Table 2

### Missing external-method cells

- MedCLIP-SAMv2 on QaTa-COV19: Dice and IoU missing.
- MedCLIP-SAMv2 on MosMedData+: Dice and IoU missing.
- FMISeg on Brain MRI: Dice and IoU missing.

### Existing cells that are not on the stated per-image scale

- FMISeg QaTa `91.21/83.84` is a global-style pair. Local official per-image
  evaluation is `84.58/76.17` (exact artifact: `84.5759/76.1711`).
- LFAENet-TGFS QaTa `90.04/81.88` is global. Best-observed simple model is
  `82.61/73.75` per-image; picked ResNet50+CXR-BERT is `80.32/70.96`.
- LFAENet-TGFS MosMed `80.10/66.68` is not the archived per-image pair. The
  recovered v9e result is `72.20/59.09` per-image.

### External values already locally available

- FMISeg Breast structured prompt: `84.23/75.35` per-image.
- FMISeg QaTa official: `84.58/76.17` per-image.

## Table 3

### Missing external-method cells

- MedCLIP-SAMv2 with our structured prompt, Brain: Dice and IoU.
- MedCLIP-SAMv2 with our structured prompt, Breast: Dice and IoU.
- FMISeg with MedCLIP-style prompt, Brain: Dice and IoU.
- FMISeg with MedCLIP-style prompt, Breast: Dice and IoU.
- FMISeg with structured prompt, Brain: Dice and IoU.

FMISeg with structured prompt on Breast is no longer missing locally:
`84.23/75.35` per-image.

### LFAENet-TGFS per-image replacements

| Prompt format | Brain Dice | Brain IoU | Breast Dice | Breast IoU |
|---|---:|---:|---:|---:|
| MedCLIP-style | 84.08 | 75.33 | 80.23 | 71.79 |
| Structured | 83.60 | 74.30 | 85.22 | 76.60 |

## Table 4

Complete. Every row already has per-image Dice and IoU in the PDF. No missing
cell and no global-to-per-image conversion is required.

## Table 5

Complete. Every row already has per-image Dice and IoU in the PDF. No missing
cell and no global-to-per-image conversion is required.

## Table 6

Leave unchanged as requested.

## Bottom line

After keeping Table 6 unchanged, the only genuinely missing experiment cells
are external MedCLIP-SAMv2/FMISeg cells in Tables 2 and 3. Tables 4 and 5 are
already complete on the per-image metric system.

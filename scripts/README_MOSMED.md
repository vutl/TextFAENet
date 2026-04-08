# MosMedData Training Quickstart

This project now includes a simple pipeline to train segmentation on MosMedData CT studies.

## 1) Prepare 2D slices from 3D NIfTI

```powershell
conda run -n medclipsamv2 python scripts/prepare_mosmed_2d.py --data-root "datasets/MosMedData Chest CT Scans with COVID-19 Related Findings COVID19_1110 1.0" --out-root datasets/mosmed_2d_prepared --neg-per-pos 0.25
```

Outputs:
- `datasets/mosmed_2d_prepared/images/*.png`
- `datasets/mosmed_2d_prepared/masks/*.png`
- `datasets/mosmed_2d_prepared/splits/train.csv`
- `datasets/mosmed_2d_prepared/splits/val.csv`
- `datasets/mosmed_2d_prepared/splits/test.csv`
- `datasets/mosmed_2d_prepared/summary.json`

## 2) Train baseline FAENet (2D)

```powershell
conda run -n medclipsamv2 python scripts/train_mosmed.py --prepared-root datasets/mosmed_2d_prepared --save-dir runs/mosmed_faenet --epochs 30 --batch-size 8 --num-workers 2
```

Outputs:
- `runs/mosmed_faenet/best.pt`
- `runs/mosmed_faenet/last.pt`
- `runs/mosmed_faenet/history.json`
- `runs/mosmed_faenet/final_test.json`

## Notes
- MosMedData masks are limited (50 annotated studies), so study-level split is used to avoid leakage.
- Foreground lesion area is extremely small at the pixel level, so the training script now auto-estimates a capped `pos_weight` from the train split and selects the best binarization threshold on the validation split instead of assuming `0.5`.
- This baseline is image-only segmentation; language-guided training can be added after confirming data pipeline quality.
- License is CC BY-NC-ND 3.0 (non-commercial, no derivatives distribution of dataset content).

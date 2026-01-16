# Few-Shot Synthetic Data Generation for Industrial Defect Detection

## Context

Asset maintenance and inspection is critical for operating power lines and substations at scale. Early identification of visible degradation, faults, or defects enables substantial cost savings. However, we typically have only a handful of anomalous image samples per defect type. This project explores using synthetic data generation to address that challenge by creating class-conditional synthetic images and segmentation masks for rare defect types.

## Dataset

Download `DefectSpectrum/Defect_Spectrum` from Hugging Face — a collection of images of damaged industrial products with segmentation masks and captions. The dataset provides real anomalous examples which we use for few-shot fine-tuning and for validating synthetic data quality.

## Layout and Notes

- `src/` : dataset, dataloader helpers.
- `notebooks/` : fine-tuning implementation notebooks.

See `docs/` for task-specific instructions.

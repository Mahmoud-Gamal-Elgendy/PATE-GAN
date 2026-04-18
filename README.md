# PATE-GAN (Synthcity) Training and Utility Evaluation

This repository contains an experimental pipeline for training **PATE-GAN** on tabular data with **Synthcity**, generating synthetic datasets, and evaluating downstream utility with multiple classifiers.

## Overview

The workflow is organized into five main stages:

1. Preprocess the Adult dataset (train/test split-safe).
2. Train PATE-GAN on preprocessed training data.
3. Export synthetic data for each run.
4. Evaluate synthetic data utility on a fixed real test set.
5. Aggregate metrics across multiple runs.

The current setup is configured for the **UCI Adult** dataset with five mapped random seeds (`Run_1` to `Run_5`).

## Repository Structure

- `PATEGAN_Train_Synthcity.py`: Main PATE-GAN training script with checkpoint/resume support.
- `Scripts/Adult_Preprocessing.py`: Adult preprocessing (imputation, one-hot encoding, scaling, target encoding).
- `Scripts/Adult_HPO_RandomizedSearchCV.py`: Randomized hyperparameter search for downstream classifiers.
- `Scripts/Utility_Evaluation.py`: Utility evaluation over generated synthetic data per run.
- `Scripts/Aggregate_Results.py`: Aggregates per-run result tables into mean +- std summaries.
- `Datasets/`: Raw and preprocessed datasets.
- `Results/`: Run artifacts, model checkpoints, synthetic datasets, and summary tables.

## Environment Setup

### Recommended Python Version

Use Python `3.10` (Python `3.11` may also work depending on your local package builds).

### Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost torch joblib synthcity scipy
```

Notes:
- If your CUDA stack is unavailable, use CPU execution by setting `DEVICE = "cpu"` in `PATEGAN_Train_Synthcity.py`.
- `Adult_HPO_RandomizedSearchCV.py` can auto-install `xgboost` if it is missing.

## Data Preparation

### 1) Adult Dataset Preprocessing

Run:

```bash
python Scripts/Adult_Preprocessing.py
```

This script:
- Loads `adult.data` and `adult.test`.
- Drops `education-num`.
- Cleans test labels (`<=50K.` -> `<=50K`, `>50K.` -> `>50K`).
- Fits imputers/encoders/scalers on training data only.
- Writes:
  - `Datasets/Adult/Adult_after/adult_train_preprocessed.csv`
  - `Datasets/Adult/Adult_after/adult_test_preprocessed.csv`

Important:
- The script currently uses hardcoded paths in `TRAIN_PATH`, `TEST_PATH`, and `OUTPUT_DIR`.
- Update these constants to your local workspace paths before running.

## PATE-GAN Training

Main script:

```bash
python PATEGAN_Train_Synthcity.py
```

Behavior:
- Uses mapped seeds to create run folders (`Run_1` to `Run_5`).
- Supports checkpointing (`checkpoint_iter_*.joblib`).
- Resumes from latest checkpoint unless forced restart is requested.
- Saves:
  - `synthetic_data.csv`
  - `training_history.csv`
  - `pategan_final_model.joblib`
  - `training.log`

Important:
- Set `TRAIN_DATA_PATH` and `OUTPUT_DIR` in `PATEGAN_Train_Synthcity.py` for your environment.

## Utility Evaluation

Script:

```bash
python Scripts/Utility_Evaluation.py
```

What it does:
- For each run folder (`Run_1` ... `Run_5`), loads `synthetic_data.csv` as train data.
- Uses fixed real preprocessed test data.
- Trains and evaluates:
  - Logistic Regression
  - AdaBoost
  - KNN
  - XGBoost
- Exports per run:
  - `Table_AUC.csv`
  - `Table_Metrics.csv`
  - `Confusion_Matrices.txt`

Important:
- Update `TEST_PATH` and `BASE_RUN_DIR` in `Scripts/Utility_Evaluation.py`.

## Aggregate Results

Script:

```bash
python Scripts/Aggregate_Results.py --base-dir <path-to-results-root>
```

Example:

```bash
python Scripts/Aggregate_Results.py --base-dir Results
```

Outputs:
- `Aggregation_Summary/Final_Aggregated_AUC.csv`
- `Aggregation_Summary/Final_Aggregated_Metrics.csv`

## Optional: Hyperparameter Search

Script:

```bash
python Scripts/Adult_HPO_RandomizedSearchCV.py
```

This performs `RandomizedSearchCV` (AUCPR objective) for the same model family used in utility evaluation.

Important:
- Update `TRAIN_PATH` in the script before running.

## Reproducibility Notes

- Seed-to-run mapping is defined in `RUN_MAPPING` inside `PATEGAN_Train_Synthcity.py`.
- The default expected target column is `salary`.
- Most scripts include absolute path placeholders and should be adapted locally.

## Current Artifacts in This Workspace

This repository already includes:
- Preprocessed Adult train/test CSVs.
- Multiple run outputs under `Results/Run_1` ... `Results/Run_5`.
- Aggregated summary files under `Results/Aggregation_Summary`.

## Troubleshooting

- **File not found errors**: verify hardcoded path constants in each script.
- **CUDA errors**: switch models/training to CPU mode where applicable.
- **Column mismatch during evaluation**: ensure synthetic and real test sets use consistent preprocessing schema.

## Citation


```bibtex
@misc{https://doi.org/10.48550/arxiv.2301.07573,
  doi = {10.48550/ARXIV.2301.07573},
  url = {https://arxiv.org/abs/2301.07573},
  author = {Qian, Zhaozhi and Cebere, Bogdan-Constantin and van der Schaar, Mihaela},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Synthcity: facilitating innovative use cases of synthetic data in different data modalities},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

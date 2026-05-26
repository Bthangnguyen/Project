# Project: Linear Regression on Real Data

This repository contains the coursework project for applying data fitting and linear regression methods to a real-world dataset.

## Part 2 goal

The Part 2 modelling task predicts Melbourne housing prices using tabular regression models. The target variable is `Price`, and the model is trained on `log1p(Price)` because house prices are right-skewed.

The completed pipeline covers:

- exploratory data analysis requirements;
- missing-value handling;
- leakage-safe preprocessing;
- OLS Full model;
- OLS Selected model using correlation ranking and VIF pruning;
- Ridge regression with cross-validation;
- Lasso regression with cross-validation;
- MAE, RMSE, R², and Log RMSE evaluation;
- residual summary;
- feature-importance exports;
- unit tests.

## Repository structure

```text
Project/
├── part1/                         # referenced implementation from Part 1, currently stored as a git subproject pointer
├── part2/
│   ├── data/
│   │   └── Melbourne_housing_FULL.csv
│   ├── data_pipeline.py           # reusable preprocessing pipeline
│   ├── model_pipeline.py          # complete train/evaluate/export pipeline
│   ├── modeling_report.md         # report notes for Part 2
│   ├── part2_notebook.ipynb       # notebook version
│   ├── model_comparison_results.csv
│   └── top_feature_coefficients.csv
├── tests/
│   ├── test_data_pipeline.py
│   └── test_model_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

On macOS/Linux, use:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full modelling pipeline

From the repository root:

```bash
python -m part2.model_pipeline
```

The script regenerates:

- `part2/model_comparison_results.csv`
- `part2/top_feature_coefficients.csv`
- `part2/outputs/missing_value_summary.csv`
- `part2/outputs/model_comparison_results.csv`
- `part2/outputs/all_model_coefficients.csv`
- `part2/outputs/top_feature_coefficients.csv`
- `part2/outputs/ols_selected_features.csv`
- `part2/outputs/residual_summary.csv`

## Run tests

```bash
pytest
```

## Modelling choices

### Missing values

Rows with missing `Price` are removed because `Price` is the supervised target. Numerical feature missing values are filled with the training median. Categorical missing values are filled with the training mode, with `Unknown` as fallback.

### Preprocessing

The preprocessing pipeline is fitted only on the training set. It then transforms the test set using the fitted medians, modes, dummy-column structure, means, and standard deviations. This prevents train/test leakage.

### Models

- **OLS Full** uses all preprocessed features.
- **OLS Selected** ranks features by absolute correlation with the log target and removes severe multicollinearity using VIF.
- **Ridge CV** chooses the regularization strength through 5-fold cross-validation.
- **Lasso CV** chooses the regularization strength through 5-fold cross-validation and can shrink weak coefficients to zero.

### Evaluation

The models are evaluated on the held-out test set using MAE, RMSE, R², and Log RMSE. MAE and RMSE are reported on the original price scale after converting predictions back from `log1p(Price)`.

## Notes

The existing `part1` entry is a subproject pointer. For a fully self-contained submission, clone with submodules or copy the Part 1 source files directly into `part1/` before final submission.

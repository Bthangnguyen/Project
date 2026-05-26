# Part 2 Modelling Report: Melbourne Housing Price Prediction

## 1. Problem definition

The task is to build and compare linear-regression-based models for a real-world continuous target. The dataset is Melbourne Housing and the target variable is `Price`. Because house prices are strongly right-skewed, the pipeline trains on `log1p(Price)` and converts predictions back to the original price scale for MAE, RMSE, and R².

## 2. Data and EDA requirements

The project covers these requirements:

- real-world dataset: Melbourne Housing;
- continuous target: `Price`;
- missing values in numeric and categorical columns;
- numeric features such as `Rooms`, `Distance`, `Landsize`, and `BuildingArea`;
- categorical features such as `Type`, `Method`, `Regionname`, and `CouncilArea`;
- reproducible split using `RANDOM_STATE = 42`.

The script exports `part2/outputs/missing_value_summary.csv`. The notebook/report should discuss dataset shape, duplicate rows, missing-value rates, target distribution, descriptive statistics, categorical cardinality, and outliers.

## 3. Missing value handling

Rows with missing `Price` are removed because `Price` is the supervised target. Feature missing values are handled by the preprocessing pipeline instead of dropping rows.

`DataPipeline` handles missing values as follows:

- numeric features: median imputation fitted on the training set;
- categorical features: mode imputation fitted on the training set, with `Unknown` as fallback;
- test data: transformed using only fitted training parameters.

Median imputation is used because housing variables often contain large outliers. The missingness is unlikely to be purely MCAR; fields such as `BuildingArea` and `YearBuilt` may depend on property type, suburb, or the collection process. A simple MAR assumption is reasonable for this baseline.

## 4. Feature engineering and preprocessing

The model script creates:

- `LogLandsize = log1p(Landsize)`;
- `LogBuildingArea = log1p(BuildingArea)`;
- `RoomsPerBathroom = Rooms / Bathroom`.

The preprocessing pipeline fits only on the training set, one-hot encodes categorical features, aligns test columns to train columns, and standardizes features with training means and standard deviations.

## 5. Models implemented

The final pipeline compares four models:

1. **OLS Full**: all preprocessed features.
2. **OLS Selected**: features ranked by absolute correlation with the log target, then pruned with VIF to reduce severe multicollinearity.
3. **Ridge CV**: Ridge regression with 5-fold cross-validation over a log-spaced alpha grid.
4. **Lasso CV**: Lasso regression with 5-fold cross-validation; useful for automatic feature selection because some coefficients become zero.

## 6. Evaluation outputs

All models are evaluated on the held-out test set using MAE, RMSE, R², and Log RMSE. The script writes outputs to:

- `part2/model_comparison_results.csv`;
- `part2/top_feature_coefficients.csv`;
- `part2/outputs/model_comparison_results.csv`;
- `part2/outputs/all_model_coefficients.csv`;
- `part2/outputs/top_feature_coefficients.csv`;
- `part2/outputs/ols_selected_features.csv`;
- `part2/outputs/residual_summary.csv`.

## 7. How to run

From the repository root:

```bash
pip install -r requirements.txt
python -m part2.model_pipeline
pytest
```

## 8. Final report conclusion guide

The final write-up should state which model has the lowest test RMSE, compare OLS Full against Ridge CV, explain whether OLS Selected trades predictive power for interpretability, use Lasso CV to discuss sparse feature selection, and mention limitations such as outliers, non-linear relationships, and location effects that a linear model may not fully capture.

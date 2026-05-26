"""End-to-end modelling pipeline for Part 2.

Run from the repository root:

    python -m part2.model_pipeline

The script performs EDA summaries, train/test split, leakage-safe preprocessing,
OLS Full, OLS Selected, Ridge CV, Lasso CV, residual diagnostics, feature
importance export, and saves reproducible CSV artifacts under part2/outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from .data_pipeline import DataPipeline
except ImportError:  # allows: python part2/model_pipeline.py
    from data_pipeline import DataPipeline

RANDOM_STATE = 42
TARGET = "Price"
LOG_TARGET = "LogPrice"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "part2" / "data" / "Melbourne_housing_FULL.csv"
OUTPUT_DIR = REPO_ROOT / "part2" / "outputs"

DROP_COLUMNS = [
    "Price",
    "LogPrice",
    "Address",
    "SellerG",
    "Date",
    "Postcode",
    "Lattitude",
    "Longtitude",
]


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def safe_expm1(x: np.ndarray) -> np.ndarray:
    """Avoid overflow when converting log-price predictions back to dollars."""
    return np.expm1(np.clip(x, a_min=None, a_max=25))


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS with a numerically stable pseudo-inverse."""
    X_design = add_intercept(X)
    return np.linalg.pinv(X_design) @ y


def predict_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ beta


def evaluate_price_model(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true = safe_expm1(y_true_log)
    y_pred = safe_expm1(y_pred_log)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
        "Log_RMSE": float(mean_squared_error(y_true_log, y_pred_log, squared=False)),
    }


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_rate": df.isna().mean(),
            "dtype": df.dtypes.astype(str),
        }
    )
    return table.sort_values("missing_rate", ascending=False)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create small, interpretable feature-engineering additions."""
    df = df.copy()
    if "Landsize" in df.columns:
        df["LogLandsize"] = np.log1p(df["Landsize"].clip(lower=0))
    if "BuildingArea" in df.columns:
        df["LogBuildingArea"] = np.log1p(df["BuildingArea"].clip(lower=0))
    if {"Rooms", "Bathroom"}.issubset(df.columns):
        df["RoomsPerBathroom"] = df["Rooms"] / df["Bathroom"].replace(0, np.nan)
    return df


def prepare_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET]).copy()
    df[LOG_TARGET] = np.log1p(df[TARGET])
    return create_features(df)


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """Compute VIF for every feature using auxiliary least squares.

    VIF = 1 / (1 - R^2_j), where R^2_j comes from regressing feature j on the
    remaining features. Constant/degenerate columns get infinite VIF.
    """
    values = X.to_numpy(dtype=float)
    result = {}
    for j, col in enumerate(X.columns):
        y_j = values[:, j]
        X_others = np.delete(values, j, axis=1)
        if np.std(y_j) == 0 or X_others.shape[1] == 0:
            result[col] = np.inf
            continue
        beta = fit_ols(X_others, y_j)
        y_hat = predict_linear(X_others, beta)
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        result[col] = float(1.0 / max(1.0 - r2, 1e-12))
    return pd.Series(result).sort_values(ascending=False)


def select_features_by_correlation_and_vif(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    max_features: int = 40,
    vif_threshold: float = 10.0,
    protected_keywords: Iterable[str] = ("Rooms", "Distance", "Bathroom", "Car", "LogLandsize", "LogBuildingArea"),
) -> List[str]:
    """Select features using target correlation first, then VIF pruning.

    This is a statistically grounded replacement for hand-picking features by
    column-name keywords. Correlation ranks candidate signal; VIF removes severe
    multicollinearity from the selected candidate set.
    """
    corr = X_train.apply(lambda col: np.corrcoef(col.to_numpy(), y_train)[0, 1])
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0).abs().sort_values(ascending=False)

    selected = list(corr.head(max_features).index)
    for keyword in protected_keywords:
        matches = [col for col in X_train.columns if keyword in col]
        for col in matches:
            if col not in selected:
                selected.append(col)

    selected = [col for col in selected if col in X_train.columns]
    if len(selected) <= 2:
        return selected

    while len(selected) > 2:
        vif_values = variance_inflation_factors(X_train[selected])
        worst_feature = vif_values.index[0]
        worst_vif = vif_values.iloc[0]
        if worst_vif <= vif_threshold:
            break
        selected.remove(worst_feature)

    return selected


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: np.ndarray,
    y_test_log: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    X_train_np = X_train.to_numpy(dtype=float)
    X_test_np = X_test.to_numpy(dtype=float)

    predictions: Dict[str, np.ndarray] = {}
    coefficients = []

    # OLS Full
    beta_full = fit_ols(X_train_np, y_train_log)
    pred_full = predict_linear(X_test_np, beta_full)
    predictions["OLS Full"] = pred_full
    coefficients.extend(
        [{"Model": "OLS Full", "Feature": "Intercept", "Coefficient": beta_full[0]}]
        + [
            {"Model": "OLS Full", "Feature": feature, "Coefficient": coef}
            for feature, coef in zip(X_train.columns, beta_full[1:])
        ]
    )

    # OLS Selected: correlation ranking + VIF pruning
    selected_features = select_features_by_correlation_and_vif(X_train, y_train_log)
    beta_selected = fit_ols(X_train[selected_features].to_numpy(dtype=float), y_train_log)
    pred_selected = predict_linear(X_test[selected_features].to_numpy(dtype=float), beta_selected)
    predictions["OLS Selected"] = pred_selected
    coefficients.extend(
        [{"Model": "OLS Selected", "Feature": "Intercept", "Coefficient": beta_selected[0]}]
        + [
            {"Model": "OLS Selected", "Feature": feature, "Coefficient": coef}
            for feature, coef in zip(selected_features, beta_selected[1:])
        ]
    )

    selected_feature_table = pd.DataFrame(
        {
            "feature": selected_features,
            "selection_method": "top target-correlation, then VIF pruning",
        }
    )

    # Ridge CV
    ridge_alphas = np.logspace(-3, 4, 40)
    ridge = RidgeCV(alphas=ridge_alphas, scoring="neg_root_mean_squared_error", cv=5)
    ridge.fit(X_train_np, y_train_log)
    pred_ridge = ridge.predict(X_test_np)
    predictions["Ridge CV"] = pred_ridge
    coefficients.extend(
        [{"Model": "Ridge CV", "Feature": "Intercept", "Coefficient": ridge.intercept_}]
        + [
            {"Model": "Ridge CV", "Feature": feature, "Coefficient": coef}
            for feature, coef in zip(X_train.columns, ridge.coef_)
        ]
    )

    # Lasso CV
    lasso = LassoCV(
        alphas=np.logspace(-4, 1, 50),
        cv=5,
        random_state=RANDOM_STATE,
        max_iter=20000,
        n_jobs=None,
    )
    lasso.fit(X_train_np, y_train_log)
    pred_lasso = lasso.predict(X_test_np)
    predictions["Lasso CV"] = pred_lasso
    coefficients.extend(
        [{"Model": "Lasso CV", "Feature": "Intercept", "Coefficient": lasso.intercept_}]
        + [
            {"Model": "Lasso CV", "Feature": feature, "Coefficient": coef}
            for feature, coef in zip(X_train.columns, lasso.coef_)
        ]
    )

    rows = []
    for model_name, y_pred_log in predictions.items():
        row = {"Model": model_name}
        row.update(evaluate_price_model(y_test_log, y_pred_log))
        if model_name == "Ridge CV":
            row["BestAlpha"] = float(ridge.alpha_)
        elif model_name == "Lasso CV":
            row["BestAlpha"] = float(lasso.alpha_)
            row["NonZeroCoefficients"] = int(np.sum(np.abs(lasso.coef_) > 1e-12))
        else:
            row["BestAlpha"] = np.nan
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("RMSE")
    coefficient_table = pd.DataFrame(coefficients)
    return results, coefficient_table, selected_feature_table, predictions


def residual_summary(y_true_log: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    y_true = safe_expm1(y_true_log)
    for model_name, pred_log in predictions.items():
        residual = y_true - safe_expm1(pred_log)
        rows.append(
            {
                "Model": model_name,
                "ResidualMean": float(np.mean(residual)),
                "ResidualMedian": float(np.median(residual)),
                "ResidualStd": float(np.std(residual)),
                "ResidualP05": float(np.quantile(residual, 0.05)),
                "ResidualP95": float(np.quantile(residual, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    df: pd.DataFrame,
    results: pd.DataFrame,
    coefficients: pd.DataFrame,
    selected_features: pd.DataFrame,
    residuals: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summarize_missing_values(df).to_csv(OUTPUT_DIR / "missing_value_summary.csv")
    results.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "all_model_coefficients.csv", index=False)

    top_coefficients = (
        coefficients[coefficients["Feature"] != "Intercept"]
        .assign(abs_coefficient=lambda x: x["Coefficient"].abs())
        .sort_values(["Model", "abs_coefficient"], ascending=[True, False])
        .groupby("Model")
        .head(20)
    )
    top_coefficients.to_csv(OUTPUT_DIR / "top_feature_coefficients.csv", index=False)
    selected_features.to_csv(OUTPUT_DIR / "ols_selected_features.csv", index=False)
    residuals.to_csv(OUTPUT_DIR / "residual_summary.csv", index=False)

    # Backward-compatible paths already used by the notebook/report.
    results.to_csv(REPO_ROOT / "part2" / "model_comparison_results.csv", index=False)
    top_coefficients.to_csv(REPO_ROOT / "part2" / "top_feature_coefficients.csv", index=False)


def main() -> None:
    np.random.seed(RANDOM_STATE)

    df = prepare_dataset(DATA_PATH)
    X = df.drop(columns=[TARGET, LOG_TARGET], errors="ignore")
    y_log = df[LOG_TARGET].to_numpy(dtype=float)

    X_train_raw, X_test_raw, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    pipeline = DataPipeline(drop_columns=DROP_COLUMNS, categorical_missing_strategy="mode")
    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    results, coefficients, selected_features, predictions = train_models(
        X_train, X_test, y_train_log, y_test_log
    )
    residuals = residual_summary(y_test_log, predictions)
    save_outputs(df, results, coefficients, selected_features, residuals)

    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

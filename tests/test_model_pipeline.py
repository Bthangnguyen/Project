import numpy as np
import pandas as pd

from part2.model_pipeline import evaluate_price_model, fit_ols, predict_linear, select_features_by_correlation_and_vif


def test_fit_ols_recovers_simple_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    beta = fit_ols(X, y)
    assert np.allclose(beta, np.array([1.0, 2.0]), atol=1e-8)
    assert np.allclose(predict_linear(X, beta), y, atol=1e-8)


def test_evaluate_price_model_perfect_prediction():
    y_log = np.log1p(np.array([100.0, 200.0, 300.0]))
    metrics = evaluate_price_model(y_log, y_log)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0
    assert metrics["Log_RMSE"] == 0.0


def test_selected_features_are_columns_from_input_frame():
    X = pd.DataFrame({
        "x1": np.arange(20, dtype=float),
        "x2": np.arange(20, dtype=float) * 2.0,
        "x3": np.ones(20),
    })
    y = X["x1"].to_numpy()
    selected = select_features_by_correlation_and_vif(X, y, max_features=3)
    assert len(selected) >= 1
    assert set(selected).issubset(set(X.columns))

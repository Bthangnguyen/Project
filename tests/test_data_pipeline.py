import numpy as np
import pandas as pd
import pytest

from part2.data_pipeline import DataPipeline


def test_pipeline_fit_transform_removes_missing_and_scales_columns():
    X_train = pd.DataFrame({
        "numeric": [1.0, 2.0, np.nan, 4.0],
        "category": ["a", "b", np.nan, "a"],
        "drop_me": [1, 2, 3, 4],
    })

    pipeline = DataPipeline(drop_columns=["drop_me"])
    X_processed = pipeline.fit_transform(X_train)

    assert "drop_me" not in X_processed.columns
    assert X_processed.isna().sum().sum() == 0
    assert list(X_processed.columns) == pipeline.get_feature_names()


def test_pipeline_transform_aligns_unseen_test_categories():
    X_train = pd.DataFrame({
        "numeric": [1.0, 2.0, 3.0],
        "category": ["a", "b", "a"],
    })
    X_test = pd.DataFrame({
        "numeric": [4.0, np.nan],
        "category": ["c", np.nan],
    })

    pipeline = DataPipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    assert list(X_test_processed.columns) == list(X_train_processed.columns)
    assert X_test_processed.isna().sum().sum() == 0


def test_pipeline_requires_fit_before_transform():
    pipeline = DataPipeline()
    with pytest.raises(ValueError):
        pipeline.transform(pd.DataFrame({"x": [1, 2]}))

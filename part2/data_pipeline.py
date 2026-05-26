"""Preprocessing pipeline for the Melbourne Housing modelling task.

The pipeline is intentionally simple and reproducible:
- fit only on the training set;
- fill numeric missing values with training medians;
- fill categorical missing values with training modes, falling back to ``Unknown``;
- one-hot encode categorical variables;
- align the test matrix to the training feature columns;
- standardize all final columns using training means/stds.

This design prevents train/test data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataPipeline:
    """Reusable preprocessing pipeline for tabular regression data."""

    drop_columns: Optional[Iterable[str]] = None
    categorical_missing_strategy: str = "mode"  # "mode" or "constant"
    fill_categorical_value: str = "Unknown"

    numeric_cols: List[str] = field(default_factory=list, init=False)
    categorical_cols: List[str] = field(default_factory=list, init=False)
    numeric_medians: Dict[str, float] = field(default_factory=dict, init=False)
    categorical_fill_values: Dict[str, str] = field(default_factory=dict, init=False)
    means: Optional[pd.Series] = field(default=None, init=False)
    stds: Optional[pd.Series] = field(default=None, init=False)
    feature_columns: List[str] = field(default_factory=list, init=False)
    is_fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.drop_columns is None:
            self.drop_columns = []
        self.drop_columns = list(self.drop_columns)
        valid_strategies = {"mode", "constant"}
        if self.categorical_missing_strategy not in valid_strategies:
            raise ValueError(
                f"categorical_missing_strategy must be one of {valid_strategies}"
            )

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        return X

    def _drop_unused_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.drop_columns, errors="ignore")

    def _identify_column_types(self, X: pd.DataFrame) -> None:
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def _compute_numeric_medians(self, X: pd.DataFrame) -> None:
        self.numeric_medians = {}
        for col in self.numeric_cols:
            median_value = X[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            self.numeric_medians[col] = float(median_value)

    def _compute_categorical_fill_values(self, X: pd.DataFrame) -> None:
        self.categorical_fill_values = {}
        for col in self.categorical_cols:
            if self.categorical_missing_strategy == "constant":
                fill_value = self.fill_categorical_value
            else:
                mode_values = X[col].dropna().mode()
                fill_value = mode_values.iloc[0] if len(mode_values) else self.fill_categorical_value
            self.categorical_fill_values[col] = str(fill_value)

    def _fill_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.numeric_medians[col])
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.categorical_fill_values[col]).astype(str)
        return X

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = [col for col in self.categorical_cols if col in X.columns]
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
        return X.astype(float)

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_columns, fill_value=0.0)

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.means) / self.stds

    def fit(self, X: pd.DataFrame) -> "DataPipeline":
        """Fit preprocessing parameters on training data only."""
        X = self._validate_input(X).copy()
        X = self._drop_unused_columns(X)
        self._identify_column_types(X)
        self._compute_numeric_medians(X)
        self._compute_categorical_fill_values(X)

        X_processed = self._fill_missing_values(X)
        X_processed = self._encode_categorical(X_processed)

        self.means = X_processed.mean()
        self.stds = X_processed.std(ddof=0).replace(0, 1.0)
        self.feature_columns = X_processed.columns.tolist()
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing parameters to new data."""
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted. Call fit(X_train) first.")
        X = self._validate_input(X).copy()
        X = self._drop_unused_columns(X)
        X_processed = self._fill_missing_values(X)
        X_processed = self._encode_categorical(X_processed)
        X_processed = self._align_columns(X_processed)
        X_processed = self._scale_features(X_processed)
        return X_processed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def get_feature_names(self) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet.")
        return list(self.feature_columns)

import pandas as pd


class DataPipeline:
    def __init__(self, drop_columns=None, fill_categorical_value="Unknown"):
        if drop_columns is None:
            drop_columns = []

        self.drop_columns = drop_columns
        self.fill_categorical_value = fill_categorical_value
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_medians = {}
        self.means = None
        self.stds = None
        self.feature_columns = []
        self.is_fitted = False

    def _validate_input(self,X): #kiểm tra X có phải là DataFrame không
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X không phải là pandas DataFrame")
        return X
    def _drop_unused_columns(self, X): #Xoá các columns không sử dụng
        return X.drop(columns = self.drop_columns, errors = 'ignore')
    def _identify_columns_types(self, X): #Xác định cột nào là số, cột nào là phân loại
        self.numeric_cols = X.select_dtypes(include = ["int64", "float64"]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include = ['object']).columns.tolist()
    def _compute_numeric_medians(self, X): # Tính median cho từng cột
        self.numeric_medians = {}
        for col in self.numeric_cols:
            median_value = X[col].median()
            if pd.isna(median_value):
                median_value = 0
            self.numeric_medians[col] = median_value
    def _fill_missing_value(self, X): # điền các giá trị thiếu
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.numeric_medians[col]) # điền giá trị trung bình cho cột số
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_categorical_value) # điền Unknown vào cột phân object
        return X
    def _encode_categorical(self, X): # one-hot các cột object
        X = pd.get_dummies(X,
                           columns = self.categorical_cols,
                           drop_first = True # bỏ cột đầu, giúp giảm đa cộng tuyến
                           )
        X = X.astype(float)
        return X
    def _align_columns(self, X):
        return X.reindex(columns = self.feature_columns, fill_value = 0)
    def _scale_features(self, X):
        return (X-self.means) / self.stds
    def fit(self, X):
        X = self._validate_input(X)
        X = X.copy()
        # Bỏ các cột không sử dụng
        X = self._drop_unused_columns(X)
        # Xác định cột nào là numeric, categorical
        self._identify_columns_types(X)
        # Xác định means cho các cột
        self._compute_numeric_medians(X)
        # Tạm thời xử lý dữ liệu train
        X_processed = self._fill_missing_value(X)
        X_processed = self._encode_categorical(X_processed)
        # tính mean và std
        self.means = X_processed.mean()
        self.stds = X_processed.std()
        self.stds = self.stds.replace(0,1)
        self.feature_columns = X_processed.columns.tolist()
        self.is_fitted = True
        return self
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được fit. Hãy gọi fit(X_train) trước.")
        X = self._validate_input(X)
        X = X.copy()
        X = self._drop_unused_columns(X)
        X_processed = self._fill_missing_value(X)
        X_processed = self._encode_categorical(X_processed)
        X_processed = self._align_columns(X_processed)
        X_processed = self._scale_features(X_processed)
        return X_processed
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    


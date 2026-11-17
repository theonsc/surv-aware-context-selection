from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TrainOnlyPreprocessor:
    """Fit continuous imputation/scaling and optional PCA on train rows only."""
    def __init__(self, n_components: int | None = None, max_components: int = 10):
        self.n_components = n_components
        self.max_components = max_components

    @staticmethod
    def _key(value):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return ("__riskset_missing__",)
        return (type(value).__name__, value)

    def fit(self, X: np.ndarray, categorical_mask: np.ndarray | None = None,
            categorical_values: dict[int, np.ndarray] | None = None):
        X = np.asarray(X, dtype=np.float32)
        self.categorical_mask_ = (np.zeros(X.shape[1], dtype=bool) if categorical_mask is None
                                  else np.asarray(categorical_mask, dtype=bool))
        if self.categorical_mask_.shape != (X.shape[1],):
            raise ValueError("categorical_mask shape mismatch")
        self.continuous_mask_ = ~self.categorical_mask_
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        if self.continuous_mask_.any():
            continuous = self.imputer_.fit_transform(X[:, self.continuous_mask_])
            self.scaler_.fit(continuous)
        self.category_maps_ = []
        for j in np.flatnonzero(self.categorical_mask_):
            vals = ((categorical_values or {}).get(int(j), X[:, j]))
            lookup = {}
            for value in vals:
                key = self._key(value)
                if key not in lookup:
                    lookup[key] = len(lookup)
            self.category_maps_.append((int(j), lookup))
        self.pca_ = None
        if self.n_components is not None:
            p = int(self.continuous_mask_.sum())
            if p == 0:
                raise ValueError("PCA requires continuous features")
            n = min(int(self.n_components), self.max_components, p, len(X))
            self.pca_ = PCA(n_components=n, random_state=0).fit(
                self.scaler_.transform(self.imputer_.transform(X[:, self.continuous_mask_])))
        return self

    def transform(self, X: np.ndarray,
                  categorical_values: dict[int, np.ndarray] | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != len(self.categorical_mask_):
            raise ValueError("feature count differs from fitted data")
        cols = []
        if self.continuous_mask_.any():
            cont = self.scaler_.transform(self.imputer_.transform(X[:, self.continuous_mask_]))
            if self.pca_ is not None:
                cont = self.pca_.transform(cont)
            cols.append(cont.astype(np.float32))
        cats = []
        for j, lookup in self.category_maps_:
            values = (categorical_values or {}).get(j, X[:, j])
            cats.append(np.asarray([lookup.get(self._key(v), len(lookup))
                                    for v in values], dtype=np.float32)[:, None])
        if cats:
            cols.append(np.concatenate(cats, axis=1))
        if not cols:
            raise ValueError("no usable features")
        return np.concatenate(cols, axis=1).astype(np.float32)

    def fit_transform(self, X: np.ndarray, categorical_mask: np.ndarray | None = None,
                      categorical_values: dict[int, np.ndarray] | None = None):
        return self.fit(X, categorical_mask, categorical_values).transform(X, categorical_values)

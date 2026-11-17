from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SurvivalDataset:
    X: np.ndarray
    time: np.ndarray
    event: np.ndarray
    ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    categorical_mask: np.ndarray | None = None
    continuous_mask: np.ndarray | None = None
    splits: dict[str, np.ndarray] = field(default_factory=dict)
    categorical_values: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.time = np.asarray(self.time, dtype=np.float32).reshape(-1)
        self.event = np.asarray(self.event, dtype=bool).reshape(-1)
        if self.X.ndim != 2 or self.X.shape[0] != len(self.time) or len(self.event) != len(self.time):
            raise ValueError("X must be 2D and X, time, event must have equal row counts")
        if not np.isfinite(self.time).all() or (self.time < 0).any():
            raise ValueError("Survival times must be finite and non-negative")
        if self.ids is not None:
            self.ids = np.asarray(self.ids)
            if len(self.ids) != len(self.time):
                raise ValueError("ids must have one value per row")
        self.categorical_mask = self._mask(self.categorical_mask, "categorical_mask")
        if self.continuous_mask is None:
            self.continuous_mask = ~self.categorical_mask
        else:
            self.continuous_mask = self._mask(self.continuous_mask, "continuous_mask")
        if np.any(self.categorical_mask & self.continuous_mask):
            raise ValueError("categorical and continuous columns must be disjoint")
        self.categorical_values = {int(j): np.asarray(v, dtype=object).reshape(-1)
                                   for j, v in self.categorical_values.items()}
        for j, values in self.categorical_values.items():
            if not 0 <= j < self.X.shape[1] or len(values) != len(self.X):
                raise ValueError("categorical_values must match a feature column and row count")

    def _mask(self, mask, name: str) -> np.ndarray:
        if mask is None:
            return np.zeros(self.X.shape[1], dtype=bool)
        value = np.asarray(mask, dtype=bool).reshape(-1)
        if len(value) != self.X.shape[1]:
            raise ValueError("%s must have one entry per feature" % name)
        return value

    def subset(self, indices: np.ndarray, *, name: str | None = None) -> "SurvivalDataset":
        indices = np.asarray(indices, dtype=int)
        return SurvivalDataset(
            X=self.X[indices], time=self.time[indices], event=self.event[indices],
            ids=None if self.ids is None else self.ids[indices],
            metadata={**self.metadata, **({"subset": name} if name else {})},
            categorical_mask=self.categorical_mask.copy(),
            continuous_mask=self.continuous_mask.copy(),
            splits={k: v.copy() for k, v in self.splits.items()},
            categorical_values={j: v[indices].copy() for j, v in self.categorical_values.items()},
        )

    def __len__(self) -> int:
        return len(self.time)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SyntheticContext:
    X: Any
    time: np.ndarray
    event: np.ndarray
    categorical_mask: np.ndarray
    trainable_mask: np.ndarray
    source_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        shape = tuple(self.X.shape)
        if len(shape) != 2 or shape[0] != len(self.time) or len(self.event) != shape[0]:
            raise ValueError("context must have K rows and matching time/event arrays")
        if len(self.categorical_mask) != shape[1] or len(self.trainable_mask) != shape[1]:
            raise ValueError("feature masks must match the context width")
        if np.any(np.asarray(self.categorical_mask) & np.asarray(self.trainable_mask)):
            raise ValueError("categorical context columns are fixed")

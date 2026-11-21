from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .base import ContextMethod
from ..data.types import SurvivalDataset


class KNNContext(ContextMethod):
    name = "knn"

    def fit(self, reference_data, optimization_queries, validation_data, **kwargs):
        if self.k < 3 or self.k > len(reference_data) or reference_data.event.sum() < 3:
            raise ValueError("k-NN requires K >= 3, K <= reference size, and at least three reference events")
        self.reference = reference_data
        self.adapter.categorical_mask = reference_data.categorical_mask
        self.history = []
        self.start_step = 0
        self.optimization_seconds = 0.0
        self.best_validation_loss = None
        self.diagnostics = {"n_optimized_features": 0, "adjusted_queries": 0}
        return self

    def neighbors(self, query):
        X = self.reference.X
        categorical = self.reference.categorical_mask
        distance = np.square(X[:, ~categorical] - query[~categorical]).sum(axis=1)
        distance += (X[:, categorical] != query[categorical]).sum(axis=1)
        order = np.argsort(distance, kind="stable")
        selected = order[:self.k].copy()
        needed = max(0, 3 - int(self.reference.event[selected].sum()))
        if needed:
            candidates = order[self.k:][self.reference.event[order[self.k:]]][:needed]
            positions = np.flatnonzero(~self.reference.event[selected])[-needed:]
            selected[positions] = candidates
        return selected, needed > 0

    def predict(self, X_query, *, record_neighbors=False):
        start = time.perf_counter()
        risks, times, curves = [], [], []
        self.diagnostics["adjusted_queries"] = 0
        neighbors = []
        for query in np.asarray(X_query):
            indices, adjusted = self.neighbors(query)
            if record_neighbors:
                neighbors.append(indices)
            self.diagnostics["adjusted_queries"] += int(adjusted)
            ref = self.reference
            self.adapter.fit_context(ref.X[indices], ref.time[indices], ref.event[indices])
            result = self.adapter.predict(query[None, :])
            risks.append(result["risk"][0])
            times.append(result["times"])
            curves.append(result["survival"][0])
        self.inference_seconds = time.perf_counter() - start
        result = {"risk": np.asarray(risks), "times": np.asarray(times), "survival": np.asarray(curves)}
        if record_neighbors:
            result["source_indices"] = np.asarray(neighbors)
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        self.adapter._torch.save({"method": self.name, "K": self.k, "X": self.reference.X,
                                  "time": self.reference.time, "event": self.reference.event,
                                  "categorical_mask": self.reference.categorical_mask,
                                  "diagnostics": self.diagnostics, "history": []}, temporary)
        os.replace(temporary, path)

    def load(self, path):
        payload = self.adapter._torch.load(path, map_location="cpu", weights_only=False)
        if payload["method"] != self.name:
            raise ValueError("saved method is not knn")
        self.k = payload["K"]
        self.reference = SurvivalDataset(payload["X"], payload["time"], payload["event"],
                                         categorical_mask=payload["categorical_mask"])
        self.adapter.categorical_mask = self.reference.categorical_mask
        self.diagnostics = payload["diagnostics"]
        self.history = []
        return self

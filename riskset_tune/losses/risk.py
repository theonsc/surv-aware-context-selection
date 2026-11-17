from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskSetContrast:
    """Experimental event-vs-risk-set moment matching penalty."""
    reference_time: np.ndarray
    reference_event: np.ndarray
    reference_X: np.ndarray
    continuous_mask: np.ndarray
    n_bins: int = 4

    def __post_init__(self):
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")
        self.reference_time = np.asarray(self.reference_time, dtype=float)
        self.reference_event = np.asarray(self.reference_event, dtype=bool)
        self.reference_X = np.asarray(self.reference_X, dtype=float)
        self.continuous_mask = np.asarray(self.continuous_mask, dtype=bool)
        if not self.continuous_mask.any():
            raise ValueError("risk contrast requires continuous covariates")
        event_times = self.reference_time[self.reference_event]
        if len(event_times) == 0:
            raise ValueError("risk contrast requires reference events")
        self.edges = np.unique(np.quantile(event_times, np.linspace(0, 1, self.n_bins + 1)))
        if len(self.edges) == 1:
            self.edges = np.asarray([self.edges[0], np.nextafter(self.edges[0], np.inf)])
        self.scale = np.nanstd(self.reference_X[:, self.continuous_mask], axis=0).clip(min=1e-6)
        self.reference_contrasts = self._event_contrasts(
            self.reference_X, self.reference_time, self.reference_event)
        if not self.reference_contrasts:
            raise ValueError("reference has no non-empty event risk sets")
        self.reference_by_bin = self._bin_means(self.reference_contrasts)
        self.bin_weights = np.asarray([len(v) for v in self.reference_contrasts.values()], dtype=float)
        self.bin_weights /= self.bin_weights.sum()

    def _event_contrasts(self, X, time, event):
        cols = np.flatnonzero(self.continuous_mask)
        times = np.asarray(time)
        values = np.asarray(X)[:, cols]
        event = np.asarray(event, dtype=bool)
        grouped = {}
        for i in np.flatnonzero(event):
            at_risk = times >= times[i]
            if not at_risk.any():
                continue
            contrast = values[i] - np.nanmean(values[at_risk], axis=0)
            b = self._bin(times[i])
            grouped.setdefault(b, []).append(contrast / self.scale)
        return {b: np.asarray(v) for b, v in grouped.items() if v}

    def _bin(self, time):
        return int(np.searchsorted(self.edges[1:-1], time, side="right"))

    @staticmethod
    def _bin_means(grouped):
        return {b: np.nanmean(values, axis=0) for b, values in grouped.items()}

    def __call__(self, X, time, event):
        import torch

        event_array = np.asarray(event, dtype=bool)
        time_array = np.asarray(time, dtype=float)
        context_bins = {self._bin(time_array[i]) for i in np.flatnonzero(event_array)}
        bins = [b for b in self.reference_by_bin if b in context_bins]
        if not bins:
            raise ValueError("synthetic context has no events in reference time bins")
        vals, weights = [], []
        cols = torch.as_tensor(np.flatnonzero(self.continuous_mask), device=X.device)
        for b in bins:
            idx = np.flatnonzero(event_array)
            # Compute the contrasts in Torch so gradients flow to synthetic X.
            event_bins = []
            for i in idx.tolist():
                t = time_array[i]
                bi = self._bin(t)
                if bi == b:
                    risk = torch.as_tensor(time_array >= t, device=X.device)
                    delta = X[i, cols] - X[risk][:, cols].mean(0)
                    event_bins.append(delta / torch.as_tensor(self.scale, device=X.device, dtype=X.dtype))
            if event_bins:
                target = torch.as_tensor(self.reference_by_bin[b], device=X.device, dtype=X.dtype)
                vals.append((torch.stack(event_bins).mean(0) - target).pow(2).sum())
                weights.append(float(self.bin_weights[list(self.reference_by_bin).index(b)]))
        w = torch.as_tensor(weights, device=X.device, dtype=X.dtype)
        self.last_coverage = float(sum(weights))
        w = w / w.sum()
        return (torch.stack(vals) * w).sum()

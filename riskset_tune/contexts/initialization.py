from __future__ import annotations

import numpy as np

from ..data.types import SurvivalDataset
from .context import SyntheticContext


def initialize_context(data: SurvivalDataset, candidate_indices: np.ndarray, k: int,
                       seed: int, strategy: str = "random",
                       n_time_strata: int = 4) -> SyntheticContext:
    """Build a context from candidates; X-only Gaussian initialization is continuous-only."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(candidate_indices, dtype=int)
    if k <= 0 or k > len(idx):
        raise ValueError("context size must be positive and no larger than candidate pool")
    if n_time_strata < 1:
        raise ValueError("n_time_strata must be positive")
    if strategy == "random":
        chosen = rng.choice(idx, size=k, replace=False)
        X = data.X[chosen].copy()
    elif strategy == "stratified":
        cuts = np.quantile(data.time[idx], np.linspace(0, 1, n_time_strata + 1)[1:-1])
        strata = np.asarray(["%d:%d" % (int(data.event[i]),
                          int(np.searchsorted(cuts, data.time[i])))
                           for i in idx])
        groups = [idx[strata == key] for key in np.unique(strata)]
        quota = np.asarray([len(g) for g in groups], dtype=float) / len(idx) * k
        counts = np.floor(quota).astype(int)
        for j in np.argsort(-(quota - counts))[:k - counts.sum()]:
            counts[j] += 1
        chosen_parts = [rng.choice(g, min(c, len(g)), replace=False)
                        for g, c in zip(groups, counts) if c]
        chosen = np.concatenate(chosen_parts)
        if len(chosen) < k:
            remaining = np.setdiff1d(idx, chosen)
            chosen = np.concatenate([chosen, rng.choice(remaining, k-len(chosen), replace=False)])
        rng.shuffle(chosen)
        X = data.X[chosen].copy()
    elif strategy == "gaussian":
        chosen = rng.choice(idx, size=k, replace=False)
        X = data.X[chosen].copy()
        cols = data.continuous_mask
        mean = np.nanmean(data.X[idx][:, cols], axis=0)
        std = np.nanstd(data.X[idx][:, cols], axis=0).clip(min=1e-6)
        X[:, cols] = rng.normal(mean, std, size=(k, int(cols.sum())))
    else:
        raise ValueError("unknown initialization strategy: %s" % strategy)
    if data.event[chosen].sum() < 3:
        raise ValueError("initialized context contains fewer than three events; try another seed")
    trainable = data.continuous_mask.copy()
    result = SyntheticContext(X=X, time=data.time[chosen].copy(), event=data.event[chosen].copy(),
                              categorical_mask=data.categorical_mask.copy(), trainable_mask=trainable,
                              source_indices=chosen.copy(), metadata={"strategy": strategy, "seed": seed})
    result.validate()
    return result

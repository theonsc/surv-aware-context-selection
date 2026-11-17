from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .types import SurvivalDataset


def make_splits(data: SurvivalDataset, seed: int = 42, train_fraction: float = .6,
                validation_fraction: float = .2, query_fraction_of_train: float = .5,
                min_events: int = 3) -> dict[str, np.ndarray]:
    """Stratified 60/20/20 split, then disjoint context and query pools."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a test split")
    all_idx = np.arange(len(data))
    train_idx, held_idx = train_test_split(
        all_idx, test_size=1 - train_fraction, random_state=seed,
        stratify=data.event.astype(int),
    )
    val_share = validation_fraction / (1 - train_fraction)
    val_idx, test_idx = train_test_split(
        held_idx, test_size=1 - val_share, random_state=seed + 1,
        stratify=data.event[held_idx].astype(int),
    )
    ref_idx, query_idx = train_test_split(
        train_idx, test_size=query_fraction_of_train, random_state=seed + 2,
        stratify=data.event[train_idx].astype(int),
    )
    result = {"train": np.sort(train_idx), "reference": np.sort(ref_idx),
              "queries": np.sort(query_idx), "validation": np.sort(val_idx),
              "test": np.sort(test_idx)}
    if data.event[result["reference"]].sum() < min_events:
        raise ValueError("reference split has fewer than %d events" % min_events)
    for key in ("train", "reference", "queries", "validation", "test"):
        idx = result[key]
        if len(np.unique(data.event[idx])) < 2:
            raise ValueError("split %s must contain events and censored rows" % key)
    if np.intersect1d(result["reference"], result["queries"]).size:
        raise RuntimeError("reference and optimization queries overlap")
    if data.ids is not None:
        ref_ids, query_ids = data.ids[result["reference"]], data.ids[result["queries"]]
        if np.intersect1d(ref_ids, query_ids).size:
            raise ValueError("reference and query identifiers overlap")
    return result

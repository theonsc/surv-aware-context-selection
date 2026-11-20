import numpy as np
import pytest

from riskset_tune.data.types import SurvivalDataset
from riskset_tune.methods.knn import KNNContext


def test_neighbors_respect_categories_ties_and_event_requirement():
    data = SurvivalDataset([[0, 0], [0, 0], [0, 1], [1, 0], [2, 0]],
                           [1, 2, 3, 4, 5], [0, 0, 1, 1, 1],
                           categorical_mask=[False, True])
    adapter = type("Adapter", (), {})()
    method = KNNContext(adapter, context_size=3).fit(data, data, data)
    indices, adjusted = method.neighbors(np.array([0, 0]))
    assert adjusted
    assert set(indices) == {2, 3, 4}
    assert data.event[indices].sum() == 3
    with pytest.raises(ValueError):
        KNNContext(adapter, context_size=2).fit(data, data, data)

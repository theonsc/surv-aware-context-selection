import numpy as np
import pytest

from riskset_tune.contexts import initialize_context
from riskset_tune.data.preprocessing import TrainOnlyPreprocessor
from riskset_tune.data.types import SurvivalDataset
from riskset_tune.losses import RiskSetContrast


def test_dataset_shapes_and_masks():
    with pytest.raises(ValueError):
        SurvivalDataset(np.zeros((3, 2)), np.ones(2), np.ones(3))
    data = SurvivalDataset(np.ones((3, 2)), np.ones(3), [1, 0, 1],
                           categorical_mask=[False, True])
    assert data.continuous_mask.tolist() == [True, False]


def test_preprocessing_is_fit_on_train_and_unknown_categories_are_reserved():
    X = np.asarray([[1, 10], [3, 20], [10000, 999]], dtype=np.float32)
    prep = TrainOnlyPreprocessor().fit(X[:2], categorical_mask=[False, True])
    out = prep.transform(X)
    assert np.isclose(prep.scaler_.mean_[0], 2.0)
    assert out[2, 0] > 1000
    assert out[:, 1].tolist() == [0.0, 1.0, 2.0]


def test_raw_categories_are_mapped_from_train_levels_only():
    X = np.zeros((3, 2), dtype=np.float32)
    values = {1: np.asarray(["bronze", "silver", "test-only"], dtype=object)}
    prep = TrainOnlyPreprocessor().fit(
        X[:2], categorical_mask=[False, True],
        categorical_values={1: values[1][:2]})
    out = prep.transform(X, values)
    assert out[:, 1].tolist() == [0.0, 1.0, 2.0]


def test_context_initialization_repeats_by_seed_and_keeps_fixed_targets():
    rng = np.random.default_rng(0)
    data = SurvivalDataset(rng.normal(size=(100, 3)), np.arange(1, 101),
                           np.asarray([i % 2 for i in range(100)]),
                           categorical_mask=[False, False, True])
    a = initialize_context(data, np.arange(100), 12, seed=7, strategy="stratified")
    b = initialize_context(data, np.arange(100), 12, seed=7, strategy="stratified")
    np.testing.assert_array_equal(a.source_indices, b.source_indices)
    np.testing.assert_array_equal(a.time, b.time)
    assert a.trainable_mask.tolist() == [True, True, False]


def test_risk_contrast_handles_tied_event_times_and_has_valid_bins():
    X = np.asarray([[0., 0.], [1., 2.], [2., 1.], [3., 3.], [4., 0.]])
    time = np.asarray([1., 1., 2., 2., 2.])
    event = np.asarray([1, 1, 0, 1, 0], dtype=bool)
    term = RiskSetContrast(time, event, X, np.asarray([True, False]), n_bins=4)
    assert term.reference_contrasts
    assert np.isclose(term.bin_weights.sum(), 1)


def test_splits_are_disjoint_and_reproducible():
    pytest.importorskip("sklearn")
    from riskset_tune.data.splits import make_splits
    data = SurvivalDataset(np.arange(600, dtype=float).reshape(200, 3),
                           np.arange(1, 201), np.asarray([i % 2 for i in range(200)]),
                           ids=np.arange(200))
    a = make_splits(data, seed=42)
    b = make_splits(data, seed=42)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not np.intersect1d(a["reference"], a["queries"]).size
    assert not np.intersect1d(a["train"], a["test"]).size


def test_censored_loss_remains_finite_and_differentiable():
    torch = pytest.importorskip("torch")
    from riskset_tune.losses.survival import native_censored_nll

    class Distribution:
        borders = torch.tensor([-3., -1., 1., 3.])

        def __call__(self, logits, target):
            return torch.nn.functional.cross_entropy(logits.reshape(-1, 3),
                                                      torch.ones(target.numel(), dtype=torch.long),
                                                      reduction="none").reshape(target.shape)

    logits = torch.randn(1, 3, 3, requires_grad=True)
    loss = native_censored_nll(logits, torch.tensor([[0., .4, -.5]]),
                               torch.tensor([[True, False, True]]), Distribution(),
                               torch.ones(1, 3))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_risk_penalty_backpropagates_only_through_continuous_context():
    torch = pytest.importorskip("torch")
    X = np.asarray([[0., 1.], [1., 2.], [3., 3.], [4., 4.], [5., 0.]], dtype=np.float32)
    time = np.arange(1, 6, dtype=float)
    event = np.asarray([1, 1, 0, 1, 0], dtype=bool)
    term = RiskSetContrast(time, event, X, np.asarray([True, False]), n_bins=2)
    learned = torch.tensor(X, requires_grad=True)
    value = term(learned, time, event)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(learned.grad).all()
    assert torch.count_nonzero(learned.grad[:, 1]) == 0

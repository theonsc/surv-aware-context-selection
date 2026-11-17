import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.integration


@pytest.fixture
def fitted_adapter():
    source = os.environ.get("SURVPFN_SOURCE")
    checkpoint = os.environ.get("SURVPFN_CHECKPOINT")
    if not source or not checkpoint or not Path(checkpoint).is_file():
        pytest.skip("requires local official SurvPFN source and checkpoint")
    from riskset_tune.models import SurvPFNAdapter
    adapter = SurvPFNAdapter(checkpoint, upstream_source=source, device="cpu")
    x = np.random.default_rng(1).normal(size=(12, 3)).astype(np.float32)
    time = np.arange(1, 13, dtype=np.float32)
    event = np.asarray([i % 3 != 0 for i in range(12)])
    q = np.ones((4, 3), dtype=np.float32)
    adapter.fit_context(x, time, event)
    return adapter, x, time, event, q


def test_differentiable_forward_matches_public_logits_and_freezes_backbone(fitted_adapter):
    torch = pytest.importorskip("torch")
    adapter, x, time, event, q = fitted_adapter
    context = torch.tensor(x, requires_grad=True)
    actual = adapter.forward_logits(context, time, event, q)
    expected = adapter.estimator.predict_logits(q)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected.detach().cpu().numpy(),
                               rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert all(parameter.grad is None for parameter in adapter.model.parameters())


def test_survival_predictions_match_public_api(fitted_adapter):
    adapter, _, _, _, q = fitted_adapter
    actual = adapter.predict(q)
    official = adapter.estimator.predict_survival_function(q, return_array=True)
    np.testing.assert_allclose(actual["survival"], official, rtol=1e-5, atol=1e-6)


def test_zero_risk_weight_recovers_tunetables_training_trajectory(fitted_adapter):
    from riskset_tune.data.types import SurvivalDataset
    from riskset_tune.methods import RiskSetTune, TuneTables
    adapter, _, _, _, _ = fitted_adapter
    rng = np.random.default_rng(3)
    X = rng.normal(size=(48, 3)).astype(np.float32)
    time = np.arange(1, 49, dtype=np.float32)
    event = np.asarray([i % 2 == 0 for i in range(48)])
    data = SurvivalDataset(X, time, event)
    reference, queries, validation = data.subset(np.arange(24)), data.subset(np.arange(24, 36)), data.subset(np.arange(36, 48))
    common = dict(context_size=8, seed=11, initialization="stratified", steps=2,
                  batch_size=4, validation_every=1, learning_rate=1e-4)
    first = TuneTables(adapter, **common).fit(reference, queries, validation, risk_reference_data=data)
    # Reload a fresh frozen model through the same local artifacts to eliminate estimator state.
    from riskset_tune.models import SurvPFNAdapter
    twin = SurvPFNAdapter(os.environ["SURVPFN_CHECKPOINT"], upstream_source=os.environ["SURVPFN_SOURCE"], device="cpu")
    second = RiskSetTune(twin, lambda_risk=0.0, **common).fit(reference, queries, validation,
                                                               risk_reference_data=data)
    np.testing.assert_array_equal(first.context.X.detach().cpu().numpy(),
                                  second.context.X.detach().cpu().numpy())
    assert first.history == second.history

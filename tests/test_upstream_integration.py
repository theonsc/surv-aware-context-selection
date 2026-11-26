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


def test_differentiable_survival_matches_official_grid_and_tails(fitted_adapter):
    import torch
    from SurvPFN.survival import survival_probs_from_logits
    adapter, _, _, _, q = fitted_adapter
    logits = adapter.estimator.predict_logits(q).detach().requires_grad_(True)
    borders = adapter.distribution.borders
    adapter.target_transform.mean_ = torch.tensor(10.)
    adapter.target_transform.std_ = torch.tensor(.5)
    normalized = torch.stack([borders[0] - 1, borders[0], (borders[0] + borders[1]) / 2,
                              borders[len(borders)//2], borders[-1], borders[-1] + 1])
    times = adapter.target_transform.inverse(normalized)
    actual = adapter.survival_tensor(logits, times)
    expected = survival_probs_from_logits(logits.detach(), adapter.target_transform.transform(times), adapter.distribution)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("method_name", ["RiskSetScoreTune", "RiskSetListwiseTune"])
def test_new_variants_reach_context_with_frozen_backbone_and_fixed_categories(fitted_adapter, method_name):
    import torch
    from riskset_tune.data.types import SurvivalDataset
    from riskset_tune import methods
    adapter, _, _, _, _ = fitted_adapter
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.normal(size=(84, 2)), rng.integers(0, 2, size=84)]).astype(np.float32)
    data = SurvivalDataset(X, np.arange(1, 85, dtype=np.float32), np.arange(84) % 3 != 0,
                           categorical_mask=[False, False, True])
    reference, queries, validation = data.subset(np.arange(42)), data.subset(np.arange(42, 63)), data.subset(np.arange(63, 84))
    common = dict(context_size=16, seed=21, initialization="random", steps=3,
                  batch_size=12, validation_every=1, diagnostics_every=1, context_space="input")
    model = getattr(methods, method_name)(adapter, **common)
    model.fit(reference, queries, validation, risk_reference_data=data)
    initial = reference.X[model.context.source_indices]
    np.testing.assert_array_equal(model.context.X.detach().numpy()[:, 2], initial[:, 2])
    assert model._auxiliary_state["alpha"] > 0
    assert any(r.get("weighted_penalty_grad_norm", 0) > 0 for r in model.history)
    assert all(p.grad is None for p in adapter.model.parameters())
    assert all(np.isfinite(r["train_loss"]) for r in model.history)


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
                  batch_size=4, validation_every=1, learning_rate=1e-4, context_space="input")
    first = TuneTables(adapter, **common).fit(reference, queries, validation, risk_reference_data=data)
    # Reload a fresh frozen model through the same local artifacts to eliminate estimator state.
    from riskset_tune.models import SurvPFNAdapter
    twin = SurvPFNAdapter(os.environ["SURVPFN_CHECKPOINT"], upstream_source=os.environ["SURVPFN_SOURCE"], device="cpu")
    second = RiskSetTune(twin, lambda_risk=0.0, **common).fit(reference, queries, validation,
                                                               risk_reference_data=data)
    np.testing.assert_array_equal(first.context.X.detach().cpu().numpy(),
                                  second.context.X.detach().cpu().numpy())
    assert first.history == second.history


@pytest.mark.parametrize("method_name", ["TuneTables", "RiskSetListwiseTune"])
def test_embedding_prompt_save_and_resume(fitted_adapter, method_name, tmp_path):
    import torch
    from riskset_tune.data.types import SurvivalDataset
    from riskset_tune import methods
    adapter, _, _, _, _ = fitted_adapter
    rng = np.random.default_rng(8)
    X = np.column_stack([rng.normal(size=(84, 2)), rng.integers(0, 2, size=84)]).astype(np.float32)
    data = SurvivalDataset(X, np.arange(1, 85, dtype=np.float32), np.arange(84) % 3 != 0,
                           categorical_mask=[False, False, True])
    reference, queries, validation = data.subset(np.arange(42)), data.subset(np.arange(42, 63)), data.subset(np.arange(63, 84))
    cls = getattr(methods, method_name)
    options = dict(context_size=16, seed=21, batch_size=12, validation_every=1, diagnostics_every=1)
    method = cls(adapter, steps=3, **options).fit(reference, queries, validation, risk_reference_data=data)
    assert method.prompt.shape == (16, 3, adapter.config.embedding_dim)
    prompt = method.prompt.detach().clone().requires_grad_()
    logits = adapter.forward_logits(method.context.X, method.context.time, method.context.event,
                                    queries.X, prompt=prompt)
    logits.square().mean().backward()
    assert torch.isfinite(prompt.grad).all()
    assert prompt.grad[:, 2].abs().sum() > 0
    assert all(p.grad is None for p in adapter.model.parameters())
    expected = method.predict(validation.X)
    method.save(tmp_path / "context.pt")
    loaded = cls(adapter, steps=3, **options).load(tmp_path / "context.pt")
    actual = loaded.predict(validation.X)
    for key in ("risk", "survival", "times"):
        np.testing.assert_allclose(expected[key], actual[key], rtol=1e-6, atol=1e-6)
    state = tmp_path / "state.pt"
    cls(adapter, steps=2, **options).fit(reference, queries, validation, risk_reference_data=data, state_path=state)
    resumed = cls(adapter, steps=3, **options).fit(reference, queries, validation, risk_reference_data=data,
                                                    state_path=state, resume=True)
    torch.testing.assert_close(method.prompt, resumed.prompt)


def test_listwise_zero_strength_matches_embedding_tunetables(fitted_adapter):
    import torch
    from riskset_tune.data.types import SurvivalDataset
    from riskset_tune.methods import TuneTables, RiskSetListwiseTune
    adapter, _, _, _, _ = fitted_adapter
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=(84, 2)), rng.integers(0, 2, size=84)]).astype(np.float32)
    data = SurvivalDataset(X, np.arange(1, 85, dtype=np.float32), np.arange(84) % 3 != 0,
                           categorical_mask=[False, False, True])
    ref, q, val = data.subset(np.arange(42)), data.subset(np.arange(42, 63)), data.subset(np.arange(63, 84))
    options = dict(context_size=16, seed=21, steps=3, batch_size=12,
                   validation_every=1, auxiliary_strength=0)
    a = TuneTables(adapter, **options).fit(ref, q, val, risk_reference_data=data)
    b = RiskSetListwiseTune(adapter, **options).fit(ref, q, val, risk_reference_data=data)
    torch.testing.assert_close(a.prompt, b.prompt)
    assert a.history == b.history

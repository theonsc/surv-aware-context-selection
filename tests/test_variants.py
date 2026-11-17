import numpy as np
import pytest
import torch

from riskset_tune.contexts import initialize_context
from riskset_tune.data.types import SurvivalDataset
from riskset_tune.losses.variants import RiskSetScore, listwise_risk_loss
from riskset_tune.losses import RiskSetContrast
from riskset_tune.methods import RiskSetListwiseTune, RiskSetScoreTune, TuneTables


def dataset():
    rng = np.random.default_rng(9)
    n = 90
    x = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.integers(0, 2, n)]).astype('float32')
    time = np.repeat(np.arange(1, 46), 2).astype('float32')
    event = np.arange(n) % 3 != 0
    return SurvivalDataset(x, time, event, categorical_mask=np.array([False, False, True]))


def test_score_targets_match_manual_tied_risk_sets_and_row_order():
    data = dataset()
    context = initialize_context(data, np.arange(len(data)), 24, 7, 'random', 4)
    term = RiskSetScore(data, context)
    z = term.encoder.transform(data.X)
    for m, beta in enumerate(term.probes):
        eta = z @ beta
        bins = np.searchsorted(term.edges[1:-1], data.time, side='right')
        for bidx, b in enumerate(term.bins):
            samples = []
            for i in np.flatnonzero(data.event & (bins == b)):
                risk = data.time >= data.time[i]
                w = np.exp(eta[risk] - eta[risk].max())
                samples.append(z[i, :2] - np.average(z[risk, :2], axis=0, weights=w))
            np.testing.assert_allclose(term.targets[m, bidx], np.mean(samples, axis=0), rtol=1e-6, atol=1e-6)
    order = np.random.default_rng(4).permutation(len(data))
    shuffled = RiskSetScore(data.subset(order), context)
    np.testing.assert_allclose(term.targets, shuffled.targets, rtol=1e-5, atol=1e-6)
    x = torch.tensor(context.X, requires_grad=True)
    term(x).backward()
    assert torch.isfinite(x.grad).all()
    assert term.coverage > 0


def test_score_zero_probe_matches_original_contrast_and_empty_bins_are_recorded():
    data = dataset()
    context = initialize_context(data, np.arange(len(data)), 24, 7, 'random', 4)
    context.event[:] = False
    context.event[np.argmin(context.time)] = True
    score = RiskSetScore(data, context)
    assert 0 < score.coverage < 1
    score.probes = score.probes[:1]
    score.targets = score.targets[:1]
    original = RiskSetContrast(data.time, data.event, data.X, context.trainable_mask, 4)
    x = torch.tensor(context.X)
    torch.testing.assert_close(score(x) * 2, original(x, context.time, context.event), rtol=2e-5, atol=1e-6)
    context.event[:] = False
    with pytest.raises(ValueError, match='no event'):
        RiskSetScore(data, context)


class _Transform:
    def __init__(self):
        self.mean_ = torch.tensor(0.)
        self.std_ = torch.tensor(1.)
    def transform(self, t):
        return (t - self.mean_) / self.std_


class _Adapter:
    _torch = torch
    device = torch.device('cpu')
    def __init__(self):
        self.target_transform = _Transform()
    def fit_context(self, x, time, event):
        self.time_context = torch.as_tensor(time, dtype=torch.float32)
        self.event_context = torch.as_tensor(event, dtype=torch.bool)
        self.target_transform.mean_ = self.time_context.mean()
        self.target_transform.std_ = self.time_context.std().clamp_min(1e-6)
        self.y_context = self.target_transform.transform(self.time_context)
    def forward_logits(self, x, time, event, q):
        return x[:, 0].mean() * q[:, 0] + x[:, 1].mean() * q[:, 1]
    def loss(self, logits, time, event, weights):
        return ((logits - torch.as_tensor(time) / 60).square() * torch.as_tensor(weights)).mean()
    def survival_tensor(self, logits, times):
        return torch.sigmoid(-logits[:, None] * (1 + times[None, :] / 100))


def test_listwise_manual_comparisons_censoring_ties_and_gradient():
    logits = torch.tensor([0., 0., 2., -1.], requires_grad=True)
    loss, stats = listwise_risk_loss(_Adapter(), logits, [1, 2, 1, 3], [True, False, False, False])
    assert stats == {'valid_comparisons': 2, 'valid_anchors': 1}
    s = _Adapter().survival_tensor(logits, torch.tensor([1.])).flatten()
    r = -torch.log(s)
    expected = torch.logsumexp(r[[0, 1, 3]], 0) - r[0]
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0 and logits.grad[2] == 0
    empty, stats = listwise_risk_loss(_Adapter(), logits, [1, 2, 1, 3], [False] * 4)
    assert empty == 0 and stats['valid_anchors'] == 0


def _fit(cls, steps, path=None, resume=False, strength=.1, diagnostics=2):
    data = dataset()
    reference = data.subset(np.arange(50))
    queries = data.subset(np.arange(50, 70))
    validation = data.subset(np.arange(70, 90))
    obj = cls(_Adapter(), context_size=20, seed=17, initialization='random', steps=steps,
              batch_size=12, validation_every=2, diagnostics_every=diagnostics,
              auxiliary_strength=strength)
    obj.fit(reference, queries, validation, risk_reference_data=data, state_path=path, resume=resume)
    return obj


def test_zero_strength_is_exact_baseline_for_both_variants():
    baseline = _fit(TuneTables, 6)
    for variant in (RiskSetScoreTune, RiskSetListwiseTune):
        result = _fit(variant, 6, strength=0)
        torch.testing.assert_close(result.context.X, baseline.context.X, rtol=0, atol=0)
        assert result.history == baseline.history


def test_variants_resume_exactly_and_instrumentation_is_inert(tmp_path):
    for variant in (RiskSetScoreTune, RiskSetListwiseTune):
        full = _fit(variant, 6)
        path = tmp_path / (variant.name + '.pt')
        _fit(variant, 4, path)
        resumed = _fit(variant, 6, path, resume=True)
        torch.testing.assert_close(full.context.X, resumed.context.X, rtol=0, atol=0)
        assert full.history == resumed.history
        no_diag = _fit(variant, 6, diagnostics=0)
        torch.testing.assert_close(full.context.X, no_diag.context.X, rtol=0, atol=0)
        assert full._auxiliary_state['alpha'] > 0
        c = full._auxiliary_state
        ratios = [c['alpha'] * r['aux_grad_norm'] / r['nll_grad_norm'] for r in c['batches']]
        assert np.median(ratios) == pytest.approx(.1)


def test_calibration_uses_an_independent_query_generator(monkeypatch):
    original = np.random.default_rng
    calls = []
    class Recorder:
        def __init__(self, generator):
            self.generator = generator
        def choice(self, *args, **kwargs):
            result = self.generator.choice(*args, **kwargs)
            calls.append(result.copy())
            return result
        def __getattr__(self, name):
            return getattr(self.generator, name)
    def factory(seed=None):
        result = original(seed)
        return Recorder(result) if isinstance(seed, int) and seed == 118 else result
    monkeypatch.setattr(np.random, 'default_rng', factory)
    _fit(TuneTables, 6)
    baseline = calls[:]
    calls.clear()
    _fit(RiskSetListwiseTune, 6)
    assert len(baseline) == len(calls) == 6
    for a, b in zip(baseline, calls):
        np.testing.assert_array_equal(a, b)

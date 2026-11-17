import numpy as np
import torch

from riskset_tune.data.types import SurvivalDataset
from riskset_tune.methods import RiskSetTune, TuneTables


class _Transform:
    def __init__(self):
        self.mean_ = torch.tensor(0.0)
        self.std_ = torch.tensor(1.0)

    def transform(self, values):
        return (values - self.mean_) / self.std_


class _ContractAdapter:
    _torch = torch
    device = torch.device("cpu")

    def __init__(self):
        self.target_transform = _Transform()

    def fit_context(self, x, time, event):
        self.time_context = torch.as_tensor(time, dtype=torch.float32)
        self.event_context = torch.as_tensor(event, dtype=torch.bool)
        self.target_transform.mean_ = self.time_context.mean()
        self.target_transform.std_ = self.time_context.std().clamp_min(1e-6)
        self.y_context = self.target_transform.transform(self.time_context)

    def forward_logits(self, context_x, context_time, context_event, query_x):
        return context_x[:, 0].mean().expand(len(query_x))

    def loss(self, logits, query_time, query_event, ipcw):
        target = torch.as_tensor(query_time, dtype=torch.float32)
        weights = torch.as_tensor(ipcw, dtype=torch.float32)
        return (((logits - target) ** 2) * weights).mean()


def _data():
    rng = np.random.default_rng(23)
    n = 80
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 3, size=n)]).astype(np.float32)
    time = np.arange(1, n + 1, dtype=np.float32)
    event = np.arange(n) % 3 != 0
    data = SurvivalDataset(x, time, event, categorical_mask=np.array([False, True]))
    return data.subset(np.arange(40)), data.subset(np.arange(40, 60)), data.subset(np.arange(60, 80))


def _method(steps, state_path=None, resume=False, diagnostics_every=2):
    reference, queries, validation = _data()
    method = TuneTables(_ContractAdapter(), context_size=12, seed=17,
                        initialization="random", steps=steps, batch_size=8,
                        learning_rate=1e-3, validation_every=2, diagnostics_every=diagnostics_every)
    method.fit(reference, queries, validation, state_path=state_path, resume=resume)
    return method


def test_optimizer_checkpoint_resume_matches_uninterrupted_trajectory(tmp_path):
    full = _method(steps=10)
    state = tmp_path / "training.pt"
    _method(steps=4, state_path=state)
    resumed = _method(steps=10, state_path=state, resume=True)

    torch.testing.assert_close(full.context.X, resumed.context.X, rtol=0, atol=0)
    assert full.history == resumed.history
    payload = torch.load(state, map_location="cpu", weights_only=False)
    assert payload["step"] == 10
    assert "optimizer" in payload and "numpy_rng" in payload and "torch_rng" in payload
    assert "target_mean" in payload and "target_std" in payload


def test_initial_context_is_candidate_and_categories_stay_fixed():
    method = _method(steps=4)
    assert method.history[0]["step"] == 0
    assert method.best_validation_loss <= method.history[0]["validation_loss"]
    np.testing.assert_array_equal(method.context.trainable_mask, [True, False])


def test_risk_gradient_instrumentation_does_not_change_updates():
    reference, queries, validation = _data()
    options = dict(context_size=12, seed=21, initialization="random", steps=4,
                   batch_size=8, learning_rate=1e-3, validation_every=2,
                   lambda_risk=0.1, risk_bins=3)
    instrumented = RiskSetTune(_ContractAdapter(), diagnostics_every=1, **options)
    control = RiskSetTune(_ContractAdapter(), diagnostics_every=0, **options)
    instrumented.fit(reference, queries, validation, risk_reference_data=reference)
    control.fit(reference, queries, validation, risk_reference_data=reference)
    torch.testing.assert_close(instrumented.context.X, control.context.X, rtol=0, atol=0)
    control_losses = [{k: row[k] for k in ("step", "train_nll", "risk_penalty_raw",
                                           "risk_penalty_weighted", "train_loss", "validation_loss")
                       if k in row} for row in control.history]
    instrumented_losses = [{k: row[k] for k in ("step", "train_nll", "risk_penalty_raw",
                                                 "risk_penalty_weighted", "train_loss", "validation_loss")
                            if k in row} for row in instrumented.history]
    assert control_losses == instrumented_losses
    assert "gradient_cosine" in instrumented.history[0] or "gradient_cosine" in instrumented.history[1]

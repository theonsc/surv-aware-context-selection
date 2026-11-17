"""Train-only risk-set score matching and within-batch ranking objectives."""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis


class RiskSetScore:
    """Match binned Cox score contributions at several fixed probes.

    The fitted encoder and reference targets never depend on validation/test
    rows. Time, event and categorical values of the context remain fixed.
    """

    def __init__(self, reference, context, n_bins=4):
        self.cont = np.flatnonzero(context.trainable_mask)
        self.cat = np.flatnonzero(context.categorical_mask)
        if not len(self.cont):
            raise ValueError("score matching needs a continuous feature")
        transformers = [("continuous", StandardScaler(), self.cont)]
        if len(self.cat):
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat))
        self.encoder = ColumnTransformer(transformers, remainder="drop")
        z = np.asarray(self.encoder.fit_transform(reference.X), dtype=np.float64)
        self.mean = np.asarray(self.encoder.named_transformers_["continuous"].mean_, dtype=np.float64)
        self.scale = np.maximum(np.asarray(self.encoder.named_transformers_["continuous"].scale_, dtype=np.float64), 1e-6)
        self.categories = ([np.asarray(c, dtype=np.float64) for c in
                            self.encoder.named_transformers_["categorical"].categories_]
                           if len(self.cat) else [])
        y = np.empty(len(reference), dtype=[("event", "?"), ("time", "<f8")])
        y["event"], y["time"] = reference.event, reference.time
        coef = np.asarray(CoxPHSurvivalAnalysis(alpha=.1).fit(z, y).coef_, dtype=np.float64)
        probes = [np.zeros_like(coef), .5 * coef, coef, 2 * coef]
        for j in range(len(self.cont)):
            axis = np.zeros_like(coef)
            axis[j] = 1.
            probes.extend([axis, -axis])
        self.probes = np.stack(probes)
        times = np.asarray(reference.time, dtype=np.float64)
        events = np.asarray(reference.event, dtype=bool)
        event_times = times[events]
        if not len(event_times):
            raise ValueError("score reference has no events")
        edges = np.unique(np.quantile(event_times, np.linspace(0, 1, n_bins + 1)))
        if len(edges) == 1:
            edges = np.array([edges[0], np.nextafter(edges[0], np.inf)])
        self.edges = edges
        ref_bins = np.searchsorted(edges[1:-1], event_times, side="right")
        ctx_events = np.flatnonzero(context.event)
        ctx_bins = np.searchsorted(edges[1:-1], np.asarray(context.time)[ctx_events], side="right")
        self.bins = [int(b) for b in np.unique(ref_bins) if b in set(ctx_bins.tolist())]
        if not self.bins:
            raise ValueError("score context has no event in a reference bin")
        all_weights = {int(b): float(np.mean(ref_bins == b)) for b in np.unique(ref_bins)}
        self.coverage = sum(all_weights[b] for b in self.bins)
        self.weights = np.asarray([all_weights[b] / self.coverage for b in self.bins])

        # Sorted suffix sums give exactly the >= risk sets, including all ties.
        order = np.argsort(times, kind="stable")
        sorted_t, sorted_z = times[order], z[order]
        event_idx = np.flatnonzero(events)
        risk_start = np.searchsorted(sorted_t, times[event_idx], side="left")
        targets = []
        for beta in self.probes:
            eta = sorted_z @ beta
            weights = np.exp(eta - eta.max())
            denom = np.cumsum(weights[::-1])[::-1]
            numer = np.cumsum((weights[:, None] * sorted_z[:, :len(self.cont)])[::-1], axis=0)[::-1]
            if np.any(denom[risk_start] <= 0) or not np.isfinite(numer).all():
                raise FloatingPointError("non-finite risk-set weights in reference score")
            contrast = z[event_idx, :len(self.cont)] - numer[risk_start] / denom[risk_start, None]
            targets.append(np.stack([contrast[ref_bins == b].mean(0) for b in self.bins]))
        self.targets = np.stack(targets)  # probes, bins, continuous features
        self.context_event_indices = ctx_events
        self.context_bins = ctx_bins
        context_time = np.asarray(context.time)
        self.risk_mask = context_time[None, :] >= context_time[ctx_events, None]
        self.fingerprint = hashlib.sha256(pickle.dumps(self.state_dict(), protocol=4)).hexdigest()

    def state_dict(self):
        return {"mean": self.mean, "scale": self.scale, "categories": self.categories,
                "probes": self.probes, "edges": self.edges, "bins": self.bins,
                "weights": self.weights, "coverage": self.coverage, "targets": self.targets,
                "continuous_indices": self.cont, "categorical_indices": self.cat,
                "risk_mask": self.risk_mask, "event_indices": self.context_event_indices,
                "event_bins": self.context_bins}

    def __call__(self, X):
        device, dtype = X.device, X.dtype
        def tensor(x):
            return torch.as_tensor(x, device=device, dtype=dtype)
        continuous = (X[:, self.cont] - tensor(self.mean)) / tensor(self.scale)
        encoded = [continuous]
        for col, categories in zip(self.cat, self.categories):
            encoded.append((X[:, int(col), None] == tensor(categories)[None, :]).to(dtype))
        z = torch.cat(encoded, dim=1)
        probes = tensor(self.probes)
        scores = probes @ z.T
        risk = torch.as_tensor(self.risk_mask, device=device)
        # (probes, events, context), with exact risk set membership.
        logits = scores[:, None, :].expand(-1, len(self.context_event_indices), -1)
        weights = torch.softmax(logits.masked_fill(~risk[None, :, :], -torch.inf), dim=-1)
        means = torch.einsum("mek,kp->mep", weights, continuous)
        contrasts = continuous[torch.as_tensor(self.context_event_indices, device=device)][None, :, :] - means
        by_bin = torch.stack([contrasts[:, torch.as_tensor(self.context_bins == b, device=device), :].mean(1)
                              for b in self.bins], dim=1)
        squared = (by_bin - tensor(self.targets)).square().mean(dim=(0, 2))
        return (squared * tensor(self.weights)).sum()


def listwise_risk_loss(adapter, logits, time, event):
    """Within-batch ordering surrogate based on cumulative predicted hazard."""
    device = logits.device
    time = torch.as_tensor(time, dtype=logits.dtype, device=device)
    event = torch.as_tensor(event, dtype=torch.bool, device=device)
    anchors = torch.nonzero(event, as_tuple=False).flatten()
    if not len(anchors):
        return logits.sum() * 0, {"valid_comparisons": 0, "valid_anchors": 0}
    later = time[None, :] > time[anchors, None]
    valid = later.any(dim=1)
    anchors, later = anchors[valid], later[valid]
    if not len(anchors):
        return logits.sum() * 0, {"valid_comparisons": 0, "valid_anchors": 0}
    survival = adapter.survival_tensor(logits, time[anchors]).clamp(1e-6, 1 - 1e-6)
    risk = -torch.log(survival).T  # anchors, batch
    mask = later | torch.nn.functional.one_hot(anchors, num_classes=len(time)).bool()
    group = risk.masked_fill(~mask, -torch.inf)
    loss = (torch.logsumexp(group, dim=1) - risk.gather(1, anchors[:, None]).squeeze(1)).mean()
    return loss, {"valid_comparisons": int(later.sum().item()), "valid_anchors": int(len(anchors))}

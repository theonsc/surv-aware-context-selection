from __future__ import annotations

import numpy as np


def ipcw_weights(reference_time, reference_event, query_time, query_event, eps=0.05):
    """Native SurvPFN censoring-KM weights, estimated using real training rows."""
    if not 0 < eps <= 1:
        raise ValueError("IPCW floor must be in (0, 1]")
    rt = np.asarray(reference_time, dtype=float)
    re = np.asarray(reference_event, dtype=bool)
    qt = np.asarray(query_time, dtype=float)
    qe = np.asarray(query_event, dtype=bool)
    order = np.argsort(rt, kind="stable")
    times = rt[order]
    censored = (~re[order]).astype(float)
    unique = np.unique(times)
    surv, g, start = 1.0, [], 0
    for t in unique:
        end = np.searchsorted(times, t, side="right")
        risk = len(times) - start
        dc = censored[start:end].sum()
        if risk and dc:
            surv *= 1.0 - dc / risk
        g.append(max(surv, eps))
        start = end
    g = np.asarray(g)
    weights = np.ones(len(qt), dtype=np.float32)
    for i in np.flatnonzero(qe):
        pos = np.searchsorted(unique, qt[i], side="right") - 1
        if pos >= 0:
            weights[i] = 1.0 / g[pos]
    return weights


def native_censored_nll(logits, y_target, event, distribution, weights):
    """IPCW event NLL plus right-censored -log survival, matching SurvPFN native loss."""
    import torch

    ev = event.to(dtype=logits.dtype)
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    y_target = y_target.to(device=logits.device, dtype=logits.dtype)
    event_nll = distribution(logits, y_target)
    probs = torch.softmax(logits, dim=-1)
    right_edges = distribution.borders[1:].to(logits.device).view(1, 1, -1)
    cdf = (probs * (right_edges <= y_target.unsqueeze(-1)).to(logits.dtype)).sum(-1)
    censored_nll = -torch.log1p(-cdf.clamp(max=.9999) + 1e-7)
    return (ev * weights * event_nll + (1 - ev) * censored_nll).mean()

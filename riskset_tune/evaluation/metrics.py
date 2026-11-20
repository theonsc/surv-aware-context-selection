from __future__ import annotations

import numpy as np


def evaluate_predictions(train_time, train_event, test_time, test_event,
                         risk, survival_times, survival, n_grid: int = 50,
                         lower_quantile: float = .1, horizon_quantile: float = .9):
    """IPCW C-index and IBS, with censoring weights and grid derived from train."""
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv

    tr_t, tr_e = np.asarray(train_time, float), np.asarray(train_event, bool)
    te_t, te_e = np.asarray(test_time, float), np.asarray(test_event, bool)
    train_y = Surv.from_arrays(event=tr_e, time=tr_t)
    if not 0 <= lower_quantile < horizon_quantile < 1:
        raise ValueError("evaluation quantiles must satisfy 0 <= lower < horizon < 1")
    if n_grid < 2:
        raise ValueError("n_grid must contain at least two time points")
    positive_train_time = tr_t[tr_t > 0]
    if len(positive_train_time) == 0:
        return {"c_index_ipcw": float("nan"), "ibs": float("nan"), "tau": None,
                "c_index_reason": "training follow-up has no usable IPCW horizon",
                "ibs_reason": "training follow-up has no non-degenerate IBS grid"}
    max_train = float(np.max(tr_t))
    tau = min(float(np.quantile(tr_t, horizon_quantile)), np.nextafter(max_train, -np.inf))
    lower = float(np.quantile(positive_train_time, lower_quantile))
    ibs_upper = np.nextafter(tau, -np.inf)
    if tau <= lower or ibs_upper <= lower:
        return {"c_index_ipcw": float("nan"), "ibs": float("nan"), "tau": tau,
                "c_index_reason": "training follow-up has no usable IPCW horizon",
                "ibs_reason": "training follow-up has no non-degenerate IBS grid"}
    # Administrative censoring at a training-derived horizon keeps the test
    # observations inside the support of the training censoring distribution.
    eval_t = np.minimum(te_t, tau)
    eval_e = te_e & (te_t <= tau)
    test_y = Surv.from_arrays(event=eval_e, time=eval_t)
    try:
        cindex = float(concordance_index_ipcw(train_y, test_y, np.asarray(risk), tau=tau)[0])
        cindex_reason = None
    except (ValueError, ZeroDivisionError) as exc:
        cindex, cindex_reason = float("nan"), str(exc)
    grid = np.linspace(lower, ibs_upper, n_grid)
    st = np.asarray(survival_times, dtype=float)
    sp = np.asarray(survival, dtype=float)
    if sp.ndim != 2 or sp.shape[0] != len(te_t):
        raise ValueError("survival predictions must have shape (n_test, n_times)")
    if st.ndim == 1:
        st = np.broadcast_to(st, sp.shape)
    if st.shape != sp.shape:
        raise ValueError("survival times must be shared or match the survival prediction shape")
    estimate = np.vstack([np.interp(grid, times[np.argsort(times)], row[np.argsort(times)])
                          for times, row in zip(st, sp)])
    if not (eval_t.min() <= grid.min() and grid.max() <= eval_t.max()):
        ibs, ibs_reason = float("nan"), "training-derived IBS grid is outside the open test follow-up interval"
    else:
        try:
            ibs = float(integrated_brier_score(train_y, test_y, estimate, grid))
            ibs_reason = None
        except (ValueError, ZeroDivisionError) as exc:
            ibs, ibs_reason = float("nan"), str(exc)
    return {"c_index_ipcw": cindex, "ibs": ibs, "tau": tau,
            "c_index_reason": cindex_reason, "ibs_reason": ibs_reason}

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np


class SurvPFNAdapter:
    """Thin bridge to the official SurvPFN source and checkpoint.

    No code or model artifact is fetched on import. Set ``SURVPFN_SOURCE`` to a
    checkout of the pinned upstream source before constructing this adapter.
    """
    UPSTREAM_REVISION = "4f26fe31efd811f546a6a3a12acd28293828ca7e"
    DEFAULT_HF_REPO = "samuelboehm/SurvPFN"
    DEFAULT_HF_FILE = "survpfn_nr.pt"

    def __init__(self, checkpoint: str | Path, device: str = "cpu",
                 upstream_source: str | Path | None = None, categorical_mask=None):
        self.checkpoint = Path(checkpoint)
        self.device_name = device
        self.upstream_source = Path(upstream_source or os.environ.get("SURVPFN_SOURCE", ""))
        self.categorical_mask = None if categorical_mask is None else np.asarray(categorical_mask, dtype=bool)
        if not self.checkpoint.is_file():
            raise FileNotFoundError("SurvPFN checkpoint is missing: %s" % self.checkpoint)
        if not self.upstream_source.is_dir():
            raise FileNotFoundError("Set SURVPFN_SOURCE to a local checkout of SurvPFN at %s" % self.UPSTREAM_REVISION)
        sys.path.insert(0, str(self.upstream_source / "src"))
        # SurvPFN needs only tfmplayground.model. Its package __init__ imports
        # unrelated training/inference utilities (and h5py); expose the same
        # pinned source package path without executing those optional imports.
        existing_package = sys.modules.get("tfmplayground")
        if existing_package is not None and hasattr(existing_package, "__path__"):
            package_paths = list(existing_package.__path__)
        else:
            package_spec = importlib.util.find_spec("tfmplayground")
            if package_spec is None or package_spec.submodule_search_locations is None:
                raise ImportError("install the pinned TFM-Playground source before loading SurvPFN")
            package_paths = list(package_spec.submodule_search_locations)
        lightweight_package = types.ModuleType("tfmplayground")
        lightweight_package.__path__ = package_paths
        lightweight_package.__package__ = "tfmplayground"
        sys.modules["tfmplayground"] = lightweight_package
        self._torch = importlib.import_module("torch")
        self._load_artifact = importlib.import_module("SurvPFN._artifact").load_artifact
        self._model_cls = importlib.import_module("SurvPFN.model").SurvPFNNet
        self._estimator_cls = importlib.import_module("SurvPFN.survival").SurvPFN
        self._target_transform_cls = importlib.import_module("SurvPFN.transforms").TargetTransform
        self._dist_cls = importlib.import_module("pfns.bar_distribution").FullSupportBarDistribution
        self.device = self._resolve_device(device)
        config, state_dict, edges, _ = self._load_artifact(self.checkpoint)
        self.config = config
        self.model = self._model_cls(
            embedding_size=config.embedding_dim,
            num_attention_heads=config.n_attention_heads,
            mlp_hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            num_outputs=config.n_buckets,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.distribution = self._dist_cls(borders=edges.to(self.device))
        self.is_fitted = False

    def _resolve_device(self, device):
        torch = self._torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable; compatibility is not assumed")
        return resolved

    def fit_context(self, X, time, event):
        torch = self._torch
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)
        time = np.asarray(time, dtype=np.float32)
        event = np.asarray(event, dtype=bool)
        if X.ndim != 2 or len(X) != len(time) or len(event) != len(time):
            raise ValueError("context X, time and event dimensions disagree")
        if event.sum() < 3:
            raise ValueError("SurvPFN requires at least three events to normalize context times")
        self.categorical_mask = (np.zeros(X.shape[1], dtype=bool) if self.categorical_mask is None
                                 else self.categorical_mask)
        if len(self.categorical_mask) != X.shape[1]:
            raise ValueError("categorical mask width does not match X")
        self.X_context = torch.as_tensor(X, device=self.device)
        self.time_context = torch.as_tensor(time, device=self.device)
        self.event_context = torch.as_tensor(event, device=self.device)
        self.target_transform = self._target_transform_cls().fit(self.time_context, self.event_context)
        self.target_transform.mean_ = self.target_transform.mean_.to(self.device)
        self.target_transform.std_ = self.target_transform.std_.to(self.device)
        self.y_context = self.target_transform.transform(self.time_context)
        self.cat_mask = torch.zeros(X.shape[1] + 1, dtype=torch.bool, device=self.device)
        self.cat_mask[:-1] = torch.as_tensor(self.categorical_mask, device=self.device)
        self.cat_mask[-1] = True
        self.estimator = self._estimator_cls.from_live_model(
            model=self.model, config=self.config,
            bucket_edges=self.distribution.borders, device=self.device,
            categorical_features_indices=np.flatnonzero(self.categorical_mask).tolist() or None)
        y_structured = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
        y_structured["event"], y_structured["time"] = event, time
        self.estimator.fit(X, y_structured)
        self.is_fitted = True
        return self

    def forward_logits(self, context_X, context_time=None, context_event=None, X_query=None):
        """Differentiable forward; gradients reach context_X, never backbone weights."""
        context_object = False
        if X_query is None:
            if context_time is None:
                raise ValueError("X_query is required")
            X_query = context_time
            if all(hasattr(context_X, key) for key in ("X", "time", "event")):
                context_time, context_event = context_X.time, context_X.event
                context_X = context_X.X
                context_object = True
            elif self.is_fitted:
                context_time, context_event = self.time_context, self.event_context
            else:
                raise ValueError("fit_context first, or pass context time and event")
        if not self.is_fitted or context_object:
            fit_X = (context_X.detach().cpu().numpy() if self._torch.is_tensor(context_X)
                     else context_X)
            self.fit_context(fit_X, context_time, context_event)
        torch = self._torch
        cx = context_X if torch.is_tensor(context_X) else torch.as_tensor(context_X, dtype=torch.float32, device=self.device)
        qx = X_query if torch.is_tensor(X_query) else torch.as_tensor(X_query, dtype=torch.float32, device=self.device)
        cx = cx.to(device=self.device, dtype=torch.float32)
        qx = qx.to(device=self.device, dtype=torch.float32)
        if cx.ndim != 2 or qx.ndim != 2 or cx.shape[1] != qx.shape[1]:
            raise ValueError("context and query must be 2D with equal feature counts")
        x = torch.cat([cx, qx], dim=0).unsqueeze(0)
        event_flag = torch.ones((1, len(cx) + len(qx), 1), dtype=x.dtype, device=self.device)
        if not self.config.is_ablation:
            event_flag[0, :len(cx), 0] = self.event_context.to(dtype=x.dtype)
        x = torch.cat([x, event_flag], dim=-1)
        y = self.target_transform.transform(self.time_context).unsqueeze(0)
        return self.model((x, y), single_eval_pos=len(cx), cat_mask=self.cat_mask)[0]

    def loss(self, logits, query_time, query_event, ipcw):
        from ..losses.survival import native_censored_nll
        torch = self._torch
        t = torch.as_tensor(query_time, dtype=torch.float32, device=self.device)
        e = torch.as_tensor(query_event, dtype=torch.bool, device=self.device)
        w = torch.as_tensor(ipcw, dtype=torch.float32, device=self.device)
        y = self.target_transform.transform(t).unsqueeze(0)
        return native_censored_nll(logits.unsqueeze(0), y, e.unsqueeze(0), self.distribution, w.unsqueeze(0))

    def survival_tensor(self, logits, times):
        """Differentiable equivalent of the official survival-grid decoder.

        ``times`` are in the original time scale.  The upstream decoder uses
        half of the endpoint bucket mass inside the grid and constant tails.
        """
        torch = self._torch
        times = torch.as_tensor(times, dtype=logits.dtype, device=logits.device).reshape(-1)
        t = self.target_transform.transform(times)
        borders = self.distribution.borders.to(device=logits.device, dtype=logits.dtype)
        widths = self.distribution.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        probs = torch.softmax(logits, dim=-1)
        factors = torch.ones_like(borders[:-1])
        factors[0] = factors[-1] = .5
        inside = probs * factors
        cumulative = inside.cumsum(-1)
        left = torch.cat([torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], dim=-1)
        clipped = t.clamp(borders[0], borders[-1])
        index = (torch.searchsorted(borders, clipped) - 1).clamp(0, len(widths) - 1)
        fraction = ((clipped - borders[index]) / (widths[index] + 1e-10)).clamp(0, 1)
        cdf = left[:, index] + inside[:, index] * fraction.unsqueeze(0)
        cdf = torch.where((t < borders[0]).unsqueeze(0), torch.zeros_like(cdf), cdf)
        cdf = torch.where((t > borders[-1]).unsqueeze(0), cumulative[:, -1:], cdf)
        return (1 - cdf).clamp(0, 1)

    def predict(self, X_query):
        if not self.is_fitted:
            raise RuntimeError("fit_context must be called before predict")
        logits = self.estimator.predict_logits(X_query)
        sf_times, sf = self.estimator._survival_grid(logits)
        risk = self.estimator.predict(X_query)
        return {"risk": np.asarray(risk), "times": sf_times, "survival": sf}

    def predict_logits(self, X_query):
        if not self.is_fitted:
            raise RuntimeError("fit_context must be called before predict")
        return self.estimator.predict_logits(X_query)

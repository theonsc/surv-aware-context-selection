from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Callable, Sequence
import os
import hashlib
import json

import numpy as np

from ..contexts import initialize_context
from ..data.types import SurvivalDataset
from ..losses import RiskSetContrast, ipcw_weights
from ..models import SurvPFNAdapter


class ContextMethod:
    name = "base"

    def __init__(self, adapter: SurvPFNAdapter, context_size: int = 64, seed: int = 42,
                 initialization: str = "random", steps: int = 200,
                 batch_size: int = 32, learning_rate: float = 1e-3,
                 validation_every: int = 10, patience: int | None = None,
                 lambda_risk: float = 0.0, risk_bins: int = 4,
                 device: str = "cpu",
                 loss_components: Sequence[Callable] | None = None,
                 n_time_strata: int = 4, censoring_weight_floor: float = 0.05,
                 diagnostics_every: int = 100, deadline_epoch: float | None = None,
                 deadline_control_path: str | Path | None = None,
                 auxiliary_strength: float = .1, context_space: str | None = None):
        self.context_space = context_space or ("embedding" if self.name in {"tunetables", "riskset-listwise"} else "input")
        if self.context_space not in {"input", "embedding"}:
            raise ValueError("context_space must be input or embedding")
        if self.context_space == "embedding" and (self.name not in {"tunetables", "riskset-listwise"} or initialization == "gaussian" or loss_components):
            raise ValueError("embedding mode supports TuneTables and Listwise with random or stratified initialization")
        self.prompt = None
        self.adapter, self.k, self.seed = adapter, context_size, seed
        self.initialization, self.steps = initialization, steps
        self.batch_size, self.learning_rate = batch_size, learning_rate
        self.validation_every, self.patience = validation_every, patience
        self.lambda_risk, self.risk_bins, self.device = lambda_risk, risk_bins, device
        self.loss_components = tuple(loss_components or ())
        self.n_time_strata = n_time_strata
        self.censoring_weight_floor = censoring_weight_floor
        self.diagnostics_every = int(diagnostics_every)
        self.deadline_epoch = deadline_epoch
        self.deadline_control_path = Path(deadline_control_path) if deadline_control_path else None
        self.auxiliary_strength = float(auxiliary_strength)
        self._auxiliary_state = None
        if self.auxiliary_strength < 0:
            raise ValueError("auxiliary_strength must be non-negative")
        if any(not callable(component) for component in self.loss_components):
            raise TypeError("each additional loss component must be callable")
        if self.steps < 1 or self.batch_size < 1 or self.validation_every < 1:
            raise ValueError("steps, batch_size and validation_every must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.history = []
        self.context = None

    def fit(self, reference_data: SurvivalDataset, optimization_queries: SurvivalDataset,
            validation_data: SurvivalDataset, *, risk_reference_data: SurvivalDataset | None = None,
            state_path: str | Path | None = None, resume: bool = False):
        torch = self.adapter._torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        rng = np.random.default_rng(self.seed + 101)
        strategy = "random" if self.name == "random" else self.initialization
        initial = initialize_context(reference_data, np.arange(len(reference_data)), self.k,
                                     self.seed, strategy, self.n_time_strata)
        self.context = initial
        self.diagnostics = {"n_optimized_features": int(initial.trainable_mask.sum()),
                            "optimized_feature_indices": np.flatnonzero(initial.trainable_mask).tolist()}
        if self.context_space == "embedding":
            self.diagnostics["n_optimized_features"] = initial.X.shape[1]
            self.diagnostics["optimized_feature_indices"] = list(range(initial.X.shape[1]))
        self.adapter.categorical_mask = initial.categorical_mask
        self.adapter.fit_context(initial.X, initial.time, initial.event)
        x_fixed = torch.as_tensor(initial.X.copy(), dtype=torch.float32, device=self.adapter.device)
        mask = torch.as_tensor(initial.trainable_mask, dtype=torch.bool, device=self.adapter.device)
        if self.context_space == "embedding":
            x_fixed = torch.empty((self.k, initial.X.shape[1], self.adapter.config.embedding_dim),
                                  dtype=torch.float32, device=self.adapter.device).uniform_(-1, 1)
            mask = torch.ones_like(x_fixed, dtype=torch.bool)
        x_parameter = torch.nn.Parameter(x_fixed.clone())
        x_trainable = bool(mask.any())
        optimizer = torch.optim.Adam([x_parameter], lr=self.learning_rate) if x_trainable else None
        query_t = np.asarray(optimization_queries.time)
        query_e = np.asarray(optimization_queries.event)
        if len(query_t) == 0 or len(validation_data) == 0:
            raise ValueError("optimization and validation query sets must be non-empty")
        risk_data = risk_reference_data or reference_data
        signature_hash = hashlib.sha256()
        for dataset in (reference_data, optimization_queries, validation_data, risk_data):
            for array in (dataset.X, dataset.time, dataset.event, dataset.categorical_mask):
                signature_hash.update(np.ascontiguousarray(array).tobytes())
        signature = {"data_sha256": signature_hash.hexdigest(), "lambda_risk": float(self.lambda_risk),
                     "learning_rate": float(self.learning_rate), "risk_bins": int(self.risk_bins),
                     "batch_size": int(self.batch_size), "validation_every": int(self.validation_every),
                     "censoring_weight_floor": float(self.censoring_weight_floor),
                     "initialization": self.initialization, "n_time_strata": int(self.n_time_strata),
                     "device": str(self.adapter.device)}
        if self.context_space == "embedding":
            signature["context_space"] = "embedding"
        if self.name in {"riskset-score", "riskset-listwise"}:
            signature.update(objective_version=2, objective=self.name,
                             auxiliary_strength=self.auxiliary_strength)
        query_w = ipcw_weights(risk_data.time, risk_data.event, query_t, query_e,
                               eps=self.censoring_weight_floor)
        val_w = ipcw_weights(risk_data.time, risk_data.event,
                             validation_data.time, validation_data.event,
                             eps=self.censoring_weight_floor)
        components = list(self.loss_components)
        components.extend(self._make_loss_components(risk_data, initial))
        best_loss, best_x, stale, start_step = float("inf"), x_fixed.detach().clone(), 0, 0
        state_path = Path(state_path) if state_path is not None else None
        if resume:
            if state_path is None or not state_path.is_file():
                raise FileNotFoundError("resume requested but the training state does not exist")
            state = torch.load(state_path, map_location=self.adapter.device, weights_only=False)
            if (state.get("method") != self.name or state.get("seed") != self.seed or state.get("k") != self.k
                    or state.get("signature") != signature):
                raise ValueError("training checkpoint does not match this method, data, seed, K or objective")
            x_parameter.data.copy_(state["x_parameter"])
            optimizer.load_state_dict(state["optimizer"])
            start_step, best_loss, best_x, stale = state["step"], state["best_loss"], state["best_x"], state["stale"]
            if (not np.array_equal(state["context_time"], initial.time)
                    or not np.array_equal(state["context_event"], initial.event)
                    or not np.array_equal(state["categorical_mask"], initial.categorical_mask)
                    or not np.array_equal(state["trainable_mask"], initial.trainable_mask)):
                raise ValueError("training checkpoint labels do not match deterministic initialization")
            self.adapter.target_transform.mean_ = state["target_mean"].to(self.adapter.device)
            self.adapter.target_transform.std_ = state["target_std"].to(self.adapter.device)
            self.adapter.y_context = self.adapter.target_transform.transform(self.adapter.time_context)
            self.history = state["history"]
            rng.bit_generator.state = state["numpy_rng"]
            torch.random.set_rng_state(state["torch_rng"].cpu())
            if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.start_step = start_step
        start = time.perf_counter()
        if self.name == "random" or not x_trainable:
            self.best_validation_loss = self._validation(
                x_fixed, validation_data, val_w)
            self.optimization_seconds = 0.0
            return self
        elif not resume:
            with torch.no_grad():
                initial_val = self._validation(x_fixed, validation_data, val_w)
            self.history.append({"step": 0, "train_nll": None, "risk_penalty_raw": 0.0,
                                 "risk_penalty_weighted": 0.0, "train_loss": initial_val,
                                 "validation_loss": initial_val, "is_initial_context": True})
            best_loss, best_x = initial_val, x_fixed.detach().clone()
        risk_parts = self._make_named_loss_components(risk_data, initial)
        if self.name in {"riskset-score", "riskset-listwise"} and self.auxiliary_strength:
            self._prepare_auxiliary(risk_data, initial)
            fingerprint = self._auxiliary_fingerprint()
            if resume:
                self._auxiliary_state = state.get("auxiliary_state")
                if not self._auxiliary_state or self._auxiliary_state.get("fingerprint") != fingerprint:
                    raise ValueError("auxiliary checkpoint does not match the fitted objective")
            else:
                self._auxiliary_state = self._calibrate_auxiliary(
                    x_parameter, x_fixed, mask, initial, optimization_queries, query_w, fingerprint)
            self.diagnostics["auxiliary_calibration"] = self._auxiliary_state
            self.diagnostics["risk_bin_coverage"] = getattr(getattr(self, "_score_term", None), "coverage", None)
        if state_path is not None and not resume:
            self._save_training_state(state_path, torch, optimizer, x_parameter, rng,
                                      start_step, best_loss, best_x, stale, signature)
        self.budget_exhausted = False
        for step in range(start_step, self.steps):
            deadline = self._current_deadline()
            if deadline is not None and time.time() >= deadline:
                self.budget_exhausted = True
                if state_path is not None:
                    self._save_training_state(state_path, torch, optimizer, x_parameter, rng,
                                              step, best_loss, best_x, stale, signature)
                break
            batch = rng.choice(len(optimization_queries),
                               size=min(self.batch_size, len(optimization_queries)), replace=False)
            optimizer.zero_grad(set_to_none=True)
            x_effective = torch.where(mask, x_parameter, x_fixed)
            logits = self._forward(x_effective, initial.time, initial.event,
                                                  torch.as_tensor(optimization_queries.X[batch], device=self.adapter.device))
            nll = self.adapter.loss(logits, query_t[batch], query_e[batch], query_w[batch])
            component_losses = [component(x_effective, initial) for component in components]
            raw_risk = [fn(x_effective, initial) for _, _, fn in risk_parts]
            weighted_risk = [weight * value for (_, weight, _), value in zip(risk_parts, raw_risk)]
            aux_stats = {}
            if self._auxiliary_state is not None:
                aux, aux_stats = self._auxiliary_loss(x_effective, logits, query_t[batch], query_e[batch])
                raw_risk.append(aux)
                weighted_risk.append(self._auxiliary_state["alpha"] * aux)
            loss = nll + sum(component_losses, nll.new_zeros(())) + sum(weighted_risk, nll.new_zeros(()))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite optimization loss at step %d" % step)
            row = {"step": step + 1, "train_nll": float(nll.detach().cpu()),
                   "risk_penalty_raw": float(sum((v.detach() for v in raw_risk), nll.new_zeros(())).cpu()),
                   "risk_penalty_weighted": float(sum((v.detach() for v in weighted_risk), nll.new_zeros(())).cpu()),
                   "train_loss": float(loss.detach().cpu())}
            row.update(aux_stats)
            if self._auxiliary_state is not None:
                row["auxiliary_alpha"] = self._auxiliary_state["alpha"]
                if getattr(self, "_score_term", None) is not None:
                    row["risk_bin_coverage"] = self._score_term.coverage
            if risk_parts:
                coverages = [getattr(getattr(fn, "__self__", None), "last_coverage", None)
                             for _, _, fn in risk_parts]
                coverages = [value for value in coverages if value is not None]
                if coverages:
                    row["risk_bin_coverage"] = float(np.mean(coverages))
                elif getattr(self, "_legacy_risk_term", None) is not None:
                    row["risk_bin_coverage"] = self._legacy_risk_term.last_coverage
            if self.diagnostics_every and (step + 1) % self.diagnostics_every == 0:
                for j in (np.flatnonzero(initial.trainable_mask) if self.context_space == "input" else []):
                    scale = float(np.nanstd(risk_data.X[:, j]))
                    scale = max(scale, 1e-6)
                    row[f"context_shift_z_feature_{j}"] = float(((x_parameter[:, j] - x_fixed[:, j]) / scale).detach().abs().mean().cpu())
                if weighted_risk:
                    nll_grad = torch.autograd.grad(nll, x_parameter, retain_graph=True, allow_unused=True)[0]
                    penalty_grad = torch.autograd.grad(sum(weighted_risk, nll.new_zeros(())), x_parameter,
                                                       retain_graph=True, allow_unused=True)[0]
                    if nll_grad is not None and penalty_grad is not None:
                        selected = mask.expand_as(nll_grad)
                        ng, pg = nll_grad[selected], penalty_grad[selected]
                        n_norm, p_norm = ng.norm(), pg.norm()
                        row.update(nll_grad_norm=float(n_norm.detach().cpu()),
                                   weighted_penalty_grad_norm=float(p_norm.detach().cpu()),
                                   penalty_to_nll_grad_ratio=float((p_norm / n_norm.clamp_min(1e-12)).detach().cpu()),
                                   gradient_cosine=float((torch.sum(ng * pg) / (n_norm * p_norm).clamp_min(1e-12)).detach().cpu()))
            loss.backward()
            if x_parameter.grad is None or not torch.isfinite(x_parameter.grad).all():
                raise FloatingPointError("missing or invalid gradient for synthetic context X*")
            optimizer.step()
            if (step + 1) % self.validation_every == 0 or step + 1 == self.steps:
                x_effective = torch.where(mask, x_parameter, x_fixed)
                val_loss = self._validation(x_effective.detach(), validation_data, val_w)
                row["validation_loss"] = val_loss
                if val_loss < best_loss:
                    best_loss, best_x, stale = val_loss, x_effective.detach().clone(), 0
                else:
                    stale += 1
                self.history.append(row)
                if self.patience is not None and stale >= self.patience:
                    break
            else:
                self.history.append(row)
            if state_path is not None and ((step + 1) % self.validation_every == 0 or step + 1 == self.steps):
                self._save_training_state(state_path, torch, optimizer, x_parameter, rng,
                                          step + 1, best_loss, best_x, stale, signature)
        if self.context_space == "embedding":
            self.prompt = best_x.detach()
        else:
            self.context.X = best_x.detach()
        self.best_validation_loss = best_loss
        self.optimization_seconds = time.perf_counter() - start
        return self

    def _save_training_state(self, path, torch, optimizer, x_parameter, rng,
                             step, best_loss, best_x, stale, signature):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        state = {"method": self.name, "seed": self.seed, "k": self.k, "signature": signature,
                 "step": int(step), "x_parameter": x_parameter.detach().cpu(),
                 "context_time": self.context.time.copy(), "context_event": self.context.event.copy(),
                 "target_mean": self.adapter.target_transform.mean_.detach().cpu(),
                 "target_std": self.adapter.target_transform.std_.detach().cpu(),
                 "categorical_mask": self.context.categorical_mask.copy(),
                 "trainable_mask": self.context.trainable_mask.copy(),
                 "optimizer": optimizer.state_dict(), "best_loss": float(best_loss),
                 "best_x": best_x.detach().cpu(), "stale": int(stale), "history": self.history,
                 "numpy_rng": rng.bit_generator.state, "torch_rng": torch.random.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        if self.name in {"riskset-score", "riskset-listwise"}:
            state.update(state_version=2, auxiliary_state=self._auxiliary_state)
        torch.save(state, temporary)
        os.replace(temporary, path)

    def _current_deadline(self):
        if self.deadline_control_path is None:
            return self.deadline_epoch
        with self.deadline_control_path.open(encoding="utf-8") as stream:
            return float(json.load(stream)["deadline_epoch"])

    def _prepare_auxiliary(self, reference_data, context):
        pass

    def _auxiliary_fingerprint(self):
        return self.name + ":v2"

    def _auxiliary_loss(self, x, logits, time, event):
        raise NotImplementedError

    def _calibrate_auxiliary(self, parameter, fixed, mask, context, queries, query_w, fingerprint):
        torch = self.adapter._torch
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.k, 701]))
        records = []
        times, events = np.asarray(queries.time), np.asarray(queries.event)
        for _ in range(20):
            batch = rng.choice(len(queries), size=min(self.batch_size, len(queries)), replace=False)
            t, e = times[batch], events[batch]
            valid = e & (t[:, None] < t[None, :]).any(axis=1)
            if not valid.any():
                continue
            x = torch.where(mask, parameter, fixed)
            logits = self._forward(x, context.time, context.event,
                torch.as_tensor(queries.X[batch], dtype=torch.float32, device=self.adapter.device))
            nll = self.adapter.loss(logits, t, e, query_w[batch])
            auxiliary, _ = self._auxiliary_loss(x, logits, t, e)
            ng = torch.autograd.grad(nll, parameter, retain_graph=True)[0]
            ag = torch.autograd.grad(auxiliary, parameter)[0]
            selected = mask.expand_as(ng)
            n_norm, a_norm = float(ng[selected].norm().detach()), float(ag[selected].norm().detach())
            if not np.isfinite([n_norm, a_norm]).all() or min(n_norm, a_norm) <= 1e-12:
                continue
            records.append({"indices": batch.tolist(), "nll_grad_norm": n_norm,
                            "aux_grad_norm": a_norm})
            if len(records) == 5:
                break
        if len(records) != 5:
            raise ValueError("auxiliary calibration requires five valid batches within 20 draws")
        alpha = self.auxiliary_strength * float(np.median(
            [r["nll_grad_norm"] / r["aux_grad_norm"] for r in records]))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("invalid calibrated auxiliary strength")
        state = {"fingerprint": fingerprint, "alpha": alpha, "batches": records,
                 "target_gradient_ratio": self.auxiliary_strength}
        if getattr(self, "_score_term", None) is not None:
            state["score_state"] = self._score_term.state_dict()
        return state

    def _make_named_loss_components(self, reference_data, context):
        return []

    def _make_loss_components(self, reference_data, context):
        """Return differentiable scalar additions; subclasses can extend the objective."""
        return []

    def _forward(self, x, time, event, queries):
        if self.context_space == "embedding":
            return self.adapter.forward_logits(self.context.X, time, event, queries, prompt=x)
        return self.adapter.forward_logits(x, time, event, queries)

    def _validation(self, x, validation, weights):
        torch = self.adapter._torch
        with torch.no_grad():
            logits = self._forward(
                x, self.context.time, self.context.event,
                torch.as_tensor(validation.X, dtype=torch.float32, device=self.adapter.device))
            return float(self.adapter.loss(logits, validation.time, validation.event, weights).cpu())

    def predict(self, X_query):
        start = time.perf_counter()
        # Update estimator support to the final (best validation) optimized context.
        self.adapter.fit_context(self.context.X.detach().cpu().numpy() if hasattr(self.context.X, "detach") else self.context.X,
                                 self.context.time, self.context.event)
        if self.context_space == "embedding":
            result = self.adapter.predict_prompt(self.context.X, self.context.time, self.context.event,
                                                 self.prompt, X_query)
        else:
            result = self.adapter.predict(X_query)
        self.inference_seconds = time.perf_counter() - start
        return result

    def save(self, path):
        torch = self.adapter._torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"method": self.name, "context_space": self.context_space,
                    "prompt": self.prompt.detach().cpu() if self.prompt is not None else None, "context_X": self.context.X.detach().cpu() if hasattr(self.context.X, "detach") else self.context.X,
                    "context_time": self.context.time, "context_event": self.context.event,
                    "categorical_mask": self.context.categorical_mask,
                    "trainable_mask": self.context.trainable_mask, "metadata": self.context.metadata,
                    "source_indices": self.context.source_indices,
                    "history": self.history, "diagnostics": getattr(self, "diagnostics", {})}, temporary)
        os.replace(temporary, path)

    def load(self, path):
        payload = self.adapter._torch.load(path, map_location=self.adapter.device, weights_only=False)
        if payload["method"] != self.name:
            raise ValueError("saved method %s cannot be loaded by %s" % (payload["method"], self.name))
        self.context_space = payload.get("context_space", "input")
        self.prompt = payload.get("prompt")
        if self.prompt is not None:
            self.prompt = self.prompt.to(self.adapter.device)
        from ..contexts import SyntheticContext
        self.context = SyntheticContext(payload["context_X"], payload["context_time"],
                                        payload["context_event"], payload["categorical_mask"],
                                        payload["trainable_mask"],
                                        source_indices=payload.get("source_indices"),
                                        metadata=payload["metadata"])
        self.history = payload["history"]
        self.adapter.categorical_mask = self.context.categorical_mask
        self.adapter.fit_context(self.context.X, self.context.time, self.context.event)
        return self


class RandomContext(ContextMethod):
    name = "random"


class TuneTables(ContextMethod):
    name = "tunetables"


class RiskSetTune(ContextMethod):
    name = "riskset-tune"

    def _make_loss_components(self, reference_data, context):
        return []

    def _make_named_loss_components(self, reference_data, context):
        if self.lambda_risk == 0:
            return []
        term = RiskSetContrast(reference_data.time, reference_data.event,
                               reference_data.X, context.trainable_mask, self.risk_bins)
        self._legacy_risk_term = term
        return [("riskset_contrast", self.lambda_risk,
                 lambda X, ctx: term(X, ctx.time, ctx.event))]


class RiskSetScoreTune(ContextMethod):
    name = "riskset-score"

    def _prepare_auxiliary(self, reference_data, context):
        from ..losses.variants import RiskSetScore
        self._score_term = RiskSetScore(reference_data, context, self.risk_bins)

    def _auxiliary_fingerprint(self):
        return self.name + ":v2:" + self._score_term.fingerprint

    def _auxiliary_loss(self, x, logits, time, event):
        return self._score_term(x), {}


class RiskSetListwiseTune(ContextMethod):
    name = "riskset-listwise"

    def _auxiliary_loss(self, x, logits, time, event):
        from ..losses.variants import listwise_risk_loss
        return listwise_risk_loss(self.adapter, logits, time, event)

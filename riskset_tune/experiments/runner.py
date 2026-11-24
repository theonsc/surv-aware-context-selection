from __future__ import annotations

import hashlib
import json
import math
import platform
import os
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from ..data.loaders import list_eligible_survset, load_finsurvival_dataset, load_survset_dataset
from ..data.preprocessing import TrainOnlyPreprocessor
from ..data.splits import make_splits
from ..data.types import SurvivalDataset
from ..evaluation import evaluate_predictions, save_results, plot_results
from ..methods import RandomContext, RiskSetTune, TuneTables, KNNContext, RiskSetListwiseTune
from ..models import SurvPFNAdapter


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_method_names(methods, previous_methods=()):
    legacy_run = "riskset-listwise" in previous_methods
    resolved = [
        ("riskset-listwise" if legacy_run else "NLL_order")
        if name in {"riskset-listwise", "NLL_order"} else name
        for name in methods
    ]
    if len(set(resolved)) != len(resolved):
        raise ValueError("method configuration contains duplicate methods or aliases")
    return resolved


def _transform_dataset(data, train_idx, pca_components=None):
    train_categories = {j: vals[train_idx] for j, vals in data.categorical_values.items()}
    categorical_count = int(data.categorical_mask.sum())
    if pca_components is not None:
        available = 10 - categorical_count
        if available < 1:
            raise ValueError("SurvPFN supports at most 10 features and requires a continuous feature here")
        pca_components = min(int(pca_components), available)
    pre = TrainOnlyPreprocessor(n_components=pca_components).fit(
        data.X[train_idx], data.categorical_mask, train_categories)
    X = pre.transform(data.X, data.categorical_values)
    n_cont = (pre.pca_.n_components_ if pre.pca_ is not None else int(data.continuous_mask.sum()))
    cat_mask = np.asarray([False] * n_cont + [True] * int(data.categorical_mask.sum()))
    if X.shape[1] < 2 or X.shape[1] > 10:
        raise ValueError("preprocessed feature count must be between 2 and 10 for this SurvPFN checkpoint")
    return SurvivalDataset(X, data.time, data.event, ids=data.ids,
                           metadata={**data.metadata, "preprocessing": "fit on training rows"},
                           categorical_mask=cat_mask), pre


def _indices_dataset(data, indices, split_name):
    return data.subset(np.asarray(indices, dtype=int), name=split_name)


def prepare_survset(config, seed, max_k):
    data_cfg = config["data"]
    name = data_cfg.get("name", "auto")
    if name == "auto":
        min_rows = max(data_cfg.get("min_rows", 500), int(math.ceil(max_k / .3)))
        candidates = list_eligible_survset(min_rows=min_rows,
                                           min_features=data_cfg.get("min_features", 2),
                                           max_features=data_cfg.get("max_features", 10),
                                           min_events=data_cfg.get("min_events", 10),
                                           max_rows=data_cfg.get("max_rows"), first_only=True)
        if not candidates:
            raise ValueError("no eligible SurvSet dataset found; see NEXT_STEPS.md")
        name = candidates[0]
    raw = load_survset_dataset(name)
    try:
        splits = make_splits(raw, seed=seed, **data_cfg.get("split", {}))
    except ValueError as exc:
        raise ValueError("SurvSet dataset %s is ineligible for seed %s: %s" % (name, seed, exc)) from exc
    transformed, _ = _transform_dataset(raw, splits["train"], data_cfg.get("pca_components"))
    for key, value in splits.items():
        transformed.splits[key] = np.asarray(value)
    return {"dataset": name,
            "reference": _indices_dataset(transformed, splits["reference"], "reference_candidates"),
            "queries": _indices_dataset(transformed, splits["queries"], "optimization_queries"),
            "validation": _indices_dataset(transformed, splits["validation"], "validation"),
            "test": _indices_dataset(transformed, splits["test"], "test"),
            "risk_reference": _indices_dataset(transformed, splits["train"], "real_training"),
            "train": _indices_dataset(transformed, splits["train"], "real_training"),
            "split_indices": splits}


def prepare_finsurvival(config, seed, max_k):
    data_cfg = config["data"]
    train_raw, test_raw = load_finsurvival_dataset(
        data_cfg["root"], data_cfg["index_event"], data_cfg["outcome_event"],
        format=data_cfg.get("format", "rds"), columns=data_cfg.get("columns"))
    if test_raw is None:
        raise ValueError("FinSurvival requires its official local test split")
    all_train = np.arange(len(train_raw))
    train_idx, val_idx = train_test_split(
        all_train, test_size=data_cfg.get("validation_fraction", .2), random_state=seed,
        stratify=train_raw.event.astype(int))
    ref_idx, query_idx = train_test_split(
        train_idx, test_size=.5, random_state=seed + 2,
        stratify=train_raw.event[train_idx].astype(int))
    transformed_train, pre = _transform_dataset(train_raw, train_idx, data_cfg.get("pca_components", 10))
    transformed_test = SurvivalDataset(pre.transform(test_raw.X, test_raw.categorical_values),
                                       test_raw.time, test_raw.event,
                                       ids=test_raw.ids, metadata=test_raw.metadata,
                                       categorical_mask=transformed_train.categorical_mask)
    return {"dataset": train_raw.metadata["dataset"],
            "reference": _indices_dataset(transformed_train, ref_idx, "reference_candidates"),
            "queries": _indices_dataset(transformed_train, query_idx, "optimization_queries"),
            "validation": _indices_dataset(transformed_train, val_idx, "validation"),
            "test": transformed_test,
            "risk_reference": _indices_dataset(transformed_train, train_idx, "real_training"),
            "train": _indices_dataset(transformed_train, train_idx, "real_training"),
            "split_indices": {"train": train_idx, "reference": ref_idx,
                              "queries": query_idx, "validation": val_idx,
                              "test": np.arange(len(test_raw))}}


def _cox_diagnostic(bundle, config, tau_config):
    """Fit a full-training, L2-regularized Cox model as a diagnostic baseline."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    train, test = bundle["train"], bundle["test"]
    cat = np.flatnonzero(train.categorical_mask)
    cont = np.flatnonzero(~train.categorical_mask)
    transformers = []
    if len(cont):
        transformers.append(("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cont))
    if len(cat):
        transformers.append(("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat))
    encoder = ColumnTransformer(transformers, remainder="drop")
    x_train = encoder.fit_transform(train.X)
    x_test = encoder.transform(test.X)
    y_train = np.empty(len(train), dtype=[("event", "?"), ("time", "<f8")])
    y_train["event"], y_train["time"] = train.event, train.time
    model = CoxPHSurvivalAnalysis(alpha=0.1).fit(x_train, y_train)
    funcs = model.predict_survival_function(x_test)
    times = np.asarray(model.unique_times_, dtype=float)
    survival = np.vstack([fn(times) for fn in funcs])
    risk = np.asarray(model.predict(x_test))
    metrics = evaluate_predictions(
        train.time, train.event, test.time, test.event, risk, times, survival,
        n_grid=config.get("evaluation", {}).get("time_grid_points", 50),
        lower_quantile=config.get("evaluation", {}).get("lower_time_quantile", .1),
        horizon_quantile=config.get("evaluation", {}).get("horizon_quantile", .9))
    return metrics, risk


def run_experiment(config, checkpoint, output_dir):
    config = {**config, "training": {**config["training"]},
              "experiment": {**config["experiment"]}}
    config["training"].setdefault("context_space", "embedding")
    model_cfg, train_cfg, exp_cfg = config["model"], config["training"], config["experiment"]
    methods = {"knn": KNNContext, "random": RandomContext, "tunetables": TuneTables,
               "riskset-tune": RiskSetTune, "riskset-listwise": RiskSetListwiseTune,
               "NLL_order": RiskSetListwiseTune}
    rows = []
    split_records = []
    split_arrays = {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous_config = out / "resolved_config.yaml"
    if previous_config.exists():
        previous = load_config(previous_config)
        old_mode = previous.get("training", {}).get("context_space", "input")
        new_mode = config.get("training", {}).get("context_space", "embedding")
        if old_mode != new_mode:
            raise ValueError("context space differs from existing run; use a new output directory")
    else:
        previous = {}
    config["experiment"]["methods"] = resolve_method_names(
        config["experiment"]["methods"], previous.get("experiment", {}).get("methods", []))
    for seed in exp_cfg["seeds"]:
        for k in exp_cfg["context_sizes"]:
            bundle = (prepare_survset(config, seed, max(exp_cfg["context_sizes"]))
                      if config["data"]["benchmark"] == "survset"
                      else prepare_finsurvival(config, seed, max(exp_cfg["context_sizes"])))
            split_summary = {"dataset": bundle["dataset"], "seed": seed, "K": k,
                             "splits": {}}
            for split_name, split_value in bundle["split_indices"].items():
                array = np.asarray(split_value, dtype=np.int64)
                key = "seed_%s_K_%s_%s" % (seed, k, split_name)
                split_arrays[key] = array
                split_summary["splits"][split_name] = {"key": key, "n": int(len(array))}
            split_records.append(split_summary)
            torch_device = model_cfg.get("device", "cpu")
            adapter = SurvPFNAdapter(checkpoint, device=torch_device,
                                     categorical_mask=bundle["reference"].categorical_mask)
            for name in exp_cfg["methods"]:
                method_cls = methods[name]
                lambdas = (train_cfg.get("risk_lambdas", [train_cfg["lambda_risk"]])
                           if name == "riskset-tune" else [0.0])
                for lambda_risk in lambdas:
                    instance = method_cls(
                        adapter=adapter, context_size=k, seed=seed,
                        initialization=train_cfg.get("initialization", "stratified"),
                        context_space=train_cfg.get("context_space") if name in {"tunetables", "riskset-listwise", "NLL_order"} else None,
                        steps=train_cfg["steps"], batch_size=train_cfg["batch_size"],
                        learning_rate=train_cfg["learning_rate"],
                        validation_every=train_cfg["validation_every"],
                        patience=train_cfg.get("early_stopping_patience"),
                        lambda_risk=(lambda_risk if name == "riskset-tune" else 0.0),
                        risk_bins=train_cfg["risk_bins"], device=torch_device,
                        auxiliary_strength=train_cfg.get("auxiliary_strength", .1),
                        n_time_strata=train_cfg.get("n_time_strata", 4),
                        censoring_weight_floor=train_cfg.get("censoring_weight_floor", .05),
                        diagnostics_every=train_cfg.get("diagnostics_every", 100),
                        deadline_epoch=train_cfg.get("deadline_epoch"))
                    state_path = None
                    if train_cfg.get("state_dir") and name not in {"random", "knn"}:
                        lambda_label = ("lambda_%s" % str(lambda_risk).replace(".", "p")
                                        if name == "riskset-tune" else "lambda_0")
                        state_path = (Path(train_cfg["state_dir"]) / bundle["dataset"] /
                                      str(seed) / str(k) / name / lambda_label / "training.pt")
                    instance.fit(bundle["reference"], bundle["queries"], bundle["validation"],
                                 risk_reference_data=bundle["risk_reference"],
                                 state_path=state_path, resume=bool(state_path and state_path.is_file()))
                    prediction = instance.predict(bundle["test"].X)
                    metrics = evaluate_predictions(
                        bundle["train"].time, bundle["train"].event,
                        bundle["test"].time, bundle["test"].event,
                        prediction["risk"], prediction["times"], prediction["survival"],
                        n_grid=config.get("evaluation", {}).get("time_grid_points", 50),
                        lower_quantile=config.get("evaluation", {}).get("lower_time_quantile", .1),
                        horizon_quantile=config.get("evaluation", {}).get("horizon_quantile", .9))
                    memory_bytes = None
                    try:
                        import psutil
                        memory_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
                    except ImportError:
                        pass
                    record = {"dataset": bundle["dataset"], "K": k, "seed": seed,
                              "method": name, "context_space": instance.context_space, "lambda_risk": lambda_risk if name == "riskset-tune" else 0.0,
                              "steps_requested": train_cfg["steps"], "completed_steps": max(
                                  (row["step"] for row in instance.history), default=0),
                              "optimized_features": getattr(instance, "diagnostics", {}).get("n_optimized_features"),
                              "budget_exhausted": getattr(instance, "budget_exhausted", False),
                              "C-index IPCW": metrics["c_index_ipcw"],
                              "IBS": metrics["ibs"],
                              "optimization_seconds": instance.optimization_seconds,
                              "inference_seconds": instance.inference_seconds,
                              "memory_bytes": memory_bytes,
                              "metric_reasons": {"c_index": metrics["c_index_reason"],
                                                 "ibs": metrics["ibs_reason"]},
                              "best_validation_loss": instance.best_validation_loss}
                    rows.append(record)
                    artifact_dir = out / "contexts" / bundle["dataset"] / str(seed) / str(k)
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    suffix = ("_lambda_%s" % str(lambda_risk).replace(".", "p")
                              if name == "riskset-tune" else "")
                    instance.save(artifact_dir / (name.replace("-", "_") + suffix + ".pt"))
            if config.get("evaluation", {}).get("cox_baseline", False):
                try:
                    cox_metrics, _ = _cox_diagnostic(bundle, config, config.get("evaluation", {}))
                    rows.append({"dataset": bundle["dataset"], "K": k, "seed": seed,
                                 "method": "coxph_l2", "lambda_risk": 0.0,
                                 "steps_requested": 0, "completed_steps": 0,
                                 "optimized_features": len(bundle["train"].continuous_mask.nonzero()[0]),
                                 "C-index IPCW": cox_metrics["c_index_ipcw"], "IBS": cox_metrics["ibs"],
                                 "optimization_seconds": 0.0, "inference_seconds": 0.0,
                                 "memory_bytes": None,
                                 "metric_reasons": {"c_index": cox_metrics["c_index_reason"],
                                                    "ibs": cox_metrics["ibs_reason"]},
                                 "best_validation_loss": None, "reference_rows": len(bundle["train"])})
                except Exception as exc:
                    rows.append({"dataset": bundle["dataset"], "K": k, "seed": seed,
                                 "method": "coxph_l2", "lambda_risk": 0.0,
                                 "steps_requested": 0, "completed_steps": 0,
                                 "C-index IPCW": float("nan"), "IBS": float("nan"),
                                 "metric_reasons": {"cox": repr(exc)}})
    frame = save_results(rows, out)
    np.savez_compressed(out / "splits.npz", **split_arrays)
    resolved_path = out / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    versions = {}
    for distribution in ("riskset-tune", "numpy", "torch", "scikit-learn",
                         "scikit-survival", "pandas", "survset", "pfns"):
        try:
            versions[distribution] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            continue
    (out / "run_metadata.json").write_text(json.dumps({
        "python": platform.python_version(), "config": config,
        "checkpoint": str(checkpoint), "upstream_revision": SurvPFNAdapter.UPSTREAM_REVISION,
        "checkpoint_sha256": digest.hexdigest(), "package_versions": versions,
        "splits": split_records, "results": rows,
    }, indent=2, default=str), encoding="utf-8")
    try:
      if config.get("evaluation", {}).get("make_plots", True):
        mpl_dir = out / ".matplotlib"
        mpl_dir.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
        os.environ.setdefault("MPLBACKEND", "Agg")
        plot_results(frame, out / "c_index_by_method.png")
    except ImportError:
        pass  # plotting is optional; CSV/Markdown results remain available
    return rows

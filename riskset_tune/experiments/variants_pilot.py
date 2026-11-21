"""Independent, resumable staged comparison of the two risk-set objectives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .extended import eligibility
from .runner import _cox_diagnostic, load_config, prepare_survset
from ..evaluation import evaluate_predictions
from ..methods import RandomContext, RiskSetListwiseTune, RiskSetScoreTune, RiskSetTune, TuneTables, KNNContext
from ..models import SurvPFNAdapter

METHODS = {"knn": KNNContext, "tunetables": TuneTables, "riskset-tune": RiskSetTune,
           "riskset-score": RiskSetScoreTune, "riskset-listwise": RiskSetListwiseTune,
           "random": RandomContext}
FIXED = {"random", "knn", "coxph_l2"}
LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def atomic_csv(frame, path):
    path = Path(path)
    with LOCK:
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)


def deadline(control):
    return float(json.loads(Path(control).read_text(encoding="utf-8"))["deadline_epoch"])


def read_results(output):
    rows = []
    for path in sorted(Path(output).glob("stage_*/*/results_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") == "complete":
                rows.append(row)
        except (OSError, json.JSONDecodeError):
            continue
    return pd.DataFrame(rows)


def aggregate(output):
    """Only include completed methods paired at identical dataset/K/seed/stage."""
    output = Path(output)
    with LOCK:
        rows = read_results(output)
        if rows.empty:
            return rows, pd.DataFrame()
        atomic_csv(rows, output / "all_stage_results.csv")
        keys = ["dataset", "K", "seed", "stage"]
        base = rows[rows.method.eq("tunetables")]
        paired = []
        for method in ("riskset-tune", "riskset-score", "riskset-listwise", "random", "knn", "coxph_l2"):
            variant = rows[rows.method.eq(method)]
            merged = variant.merge(base, on=keys, suffixes=("_variant", "_baseline"))
            for r in merged.to_dict("records"):
                item = {k: r[k] for k in keys}
                item["method"] = method
                for metric, label in (("ibs", "ibs"), ("c_index_ipcw", "c_index")):
                    v, b = float(r[metric + "_variant"]), float(r[metric + "_baseline"])
                    item[label + "_baseline"] = b
                    item[label + "_variant"] = v
                    item[label + "_delta"] = v - b
                    item[label + "_relative_pct"] = 100 * (v - b) / abs(b) if b else np.nan
                paired.append(item)
        pairs = pd.DataFrame(paired)
        atomic_csv(pairs, output / "paired_comparisons.csv")
        if not pairs.empty:
            summary = []
            for group_keys, group in pairs.groupby(["stage", "method", "dataset", "K"], dropna=False):
                ibs, ci = group.ibs_delta.dropna(), group.c_index_delta.dropna()
                summary.append(dict(zip(["stage", "method", "dataset", "K"], group_keys)) | {
                    "n": len(group), "ibs_mean_delta": ibs.mean(), "ibs_median_delta": ibs.median(),
                    "ibs_wins": int((ibs < 0).sum()), "c_index_mean_delta": ci.mean(),
                    "c_index_median_delta": ci.median(), "c_index_wins": int((ci > 0).sum())})
            atomic_csv(pd.DataFrame(summary), output / "paired_by_dataset_K.csv")
            stages = set(int(x) for x in rows.stage.unique())
            common_rows = []
            if {200, 1000, 2000}.issubset(stages):
                for _, group in pairs.groupby(["dataset", "K", "seed", "method"]):
                    if {200, 1000, 2000}.issubset(set(group.stage)):
                        common_rows.extend(group.to_dict("records"))
            atomic_csv(pd.DataFrame(common_rows, columns=pairs.columns), output / "common_stage_trajectories.csv")
        return rows, pairs


def estimate_progress(output, config, jobs, completed, control):
    """Project remaining work from elapsed fit times, never test metrics."""
    rows = read_results(output)
    if rows.empty:
        return None
    stages = config["experiment"]["stage_steps"]
    previous = {s: (stages[i-1] if i else 0) for i, s in enumerate(stages)}
    optimized = [m for m in config["experiment"]["methods"] if m not in FIXED]
    rates = {}
    for _, row in rows[rows.method.isin(optimized)].iterrows():
        steps = int(row.completed_steps) - int(row.get("start_step", previous[int(row.stage)]))
        if steps > 0 and np.isfinite(row.optimization_seconds):
            rates.setdefault((int(row.K), row.method), []).append(float(row.optimization_seconds) / steps)
    medians = {key: float(np.median(values)) for key, values in rates.items()}
    if not all(any((k, m) in medians for k in config["experiment"]["context_sizes"]) for m in optimized):
        return None
    fixed_seconds = {m: float(rows[rows.method.eq(m)].wall_seconds.median())
                     for m in FIXED if rows.method.eq(m).any()}
    method_overhead = float((rows.wall_seconds - rows.optimization_seconds.fillna(0)).clip(lower=0).median())
    available = max(0., deadline(control) - time.time())
    projected_total, projected_jobs, used_extrapolation = 0., 0, False
    for stage, seed, k, dataset in jobs:
        if (stage, seed, k, dataset) in completed:
            continue
        job_dir = Path(output) / f"stage_{stage}" / f"{dataset}_seed_{seed}_K_{k}"
        cost = 0.
        for method in config["experiment"]["methods"]:
            if (job_dir / f"results_{method}.json").exists():
                continue
            if method in FIXED:
                cost += fixed_seconds.get(method, 1.)
                continue
            rate = medians.get((k, method))
            if rate is None:
                other = 64 if k == 32 else 32
                rate = medians[(other, method)] * (1 / 1.35 if k == 32 else 1.35)
                used_extrapolation = True
            cost += rate * (stage - previous[stage]) + method_overhead
        projected_total += cost
        if projected_total <= available:
            projected_jobs += 1
    return {"remaining_budget_hours": available / 3600,
            "projected_remaining_hours": projected_total / 3600,
            "estimated_additional_jobs_within_budget": projected_jobs,
            "completed_jobs": len(completed),
            "rates_seconds_per_step": {f"K{k}:{m}": value for (k, m), value in medians.items()},
            "uses_prior_K_ratio_1_35": used_extrapolation,
            "note": "Approximation from observed wall times; future coverage is not guaranteed."}


def report(output, status, label):
    output = Path(output)
    with LOCK:
        rows, pairs = aggregate(output)
        lines = [f"# RiskSet variantes — {label}", "", f"Atualizado: {now()}",
                 f"Estado: {status.get('state')}; combinações completas: {len(status.get('completed_jobs', []))}.",
                 "IBS é principal; diferenças negativas favorecem a variante. C-index maior é melhor.",
                 "Cada linha comparada usa o mesmo dataset, K, seed e marco. Seeds não são independentes.",
                 "Os datasets são de desenvolvimento; Cox usa todo o treino real.", ""]
        if status.get("coverage_estimate"):
            estimate = status["coverage_estimate"]
            lines += [f"Estimativa de trabalho restante: {estimate['projected_remaining_hours']:.2f} h; "
                      f"orçamento restante: {estimate['remaining_budget_hours']:.2f} h; "
                      f"jobs adicionais estimados: {estimate['estimated_additional_jobs_within_budget']}.",
                      "A projeção usa tempos anteriores e será atualizada; não é garantia de cobertura.", ""]
        if not rows.empty:
            table = rows.groupby(["stage", "method"])[["ibs", "c_index_ipcw"]].agg(["mean", "count"])
            lines.extend(["## Cobertura e métricas", "", table.round(6).to_markdown(), ""])
        if not pairs.empty:
            table = pairs.groupby(["stage", "method"]).agg(
                n=("ibs_delta", "count"), ibs_mean=("ibs_delta", "mean"),
                ibs_median=("ibs_delta", "median"),
                ibs_mean_relative_pct=("ibs_relative_pct", "mean"),
                ibs_wins=("ibs_delta", lambda x: int((x < 0).sum())),
                c_index_mean=("c_index_delta", "mean"),
                c_index_median=("c_index_delta", "median"),
                c_index_mean_relative_pct=("c_index_relative_pct", "mean"))
            lines.extend(["## Diferenças pareadas contra TuneTables", "", table.round(6).to_markdown(), "",
                          "Detalhes absolutos e relativos por dataset e K em `paired_comparisons.csv` e `paired_by_dataset_K.csv`.", ""])
            dataset_means = pairs.groupby(["stage", "method", "dataset"], as_index=False).ibs_delta.mean()
            dataset_wins = dataset_means.groupby(["stage", "method"]).agg(
                datasets=("dataset", "size"), ibs_dataset_wins=("ibs_delta", lambda x: int((x < 0).sum())))
            lines += ["## Vitórias por dataset em IBS", "", dataset_wins.to_markdown(), ""]
        lines.extend([f"Job atual: `{status.get('current_job')}`", "",
                      "Resultados parciais não representam confirmação de eficácia.", ""])
        temp = output / (label + "_report.md.tmp")
        temp.write_text("\n".join(lines), encoding="utf-8")
        os.replace(temp, output / (label + "_report.md"))

        # Each context is saved atomically. Keep only its latest completed stage.
        histories = {}
        for path in sorted(output.glob("stage_*/*/contexts/*.pt")):
            stage = int(path.parts[-4].split("_", 1)[1])
            key = (path.parts[-3], path.stem)
            if key in histories and histories[key][0] >= stage:
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            histories[key] = (stage, [{"job": key[0], "method": payload["method"], "stage": stage, **r}
                                      for r in payload.get("history", [])])
        diagnostic_rows = [r for _, items in histories.values() for r in items]
        if diagnostic_rows:
            frame = pd.DataFrame(diagnostic_rows)
            atomic_csv(frame, output / "training_diagnostics.csv")
            try:
                os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
                os.environ.setdefault("MPLBACKEND", "Agg")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                for column, filename, ylabel in (("validation_loss", "validation_curves.png", "NLL de validação"),
                                                  ("penalty_to_nll_grad_ratio", "gradient_ratios.png", "Razão de normas")):
                    if column not in frame:
                        continue
                    fig, ax = plt.subplots(figsize=(9, 5))
                    for method, group in frame.dropna(subset=[column]).groupby("method"):
                        curve = group.groupby("step")[column].mean()
                        ax.plot(curve.index, curve.values, label=method)
                    ax.set(xlabel="Passos", ylabel=ylabel, title="Diagnóstico exploratório; cobertura variável")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(output / filename, dpi=130)
                    plt.close(fig)
            except ImportError:
                pass


def metadata_file(output, config, checkpoint):
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    versions = {}
    for name in ("riskset-tune", "numpy", "torch", "scikit-learn", "scikit-survival", "pandas", "survset", "pfns"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    previous = output / "run_metadata.json"
    mode = config["training"].get("context_space", "embedding")
    if previous.exists():
        prior_config = json.loads(previous.read_text()).get("config", {})
        prior_mode = prior_config.get("training", {}).get("context_space", "input")
        if prior_mode != mode:
            raise ValueError("context space differs from existing run; use a new output directory")
    atomic_json(output / "run_metadata.json", {"python": platform.python_version(), "config": config,
                "checkpoint": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
                "survpfn_revision": SurvPFNAdapter.UPSTREAM_REVISION, "versions": versions})
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run_job(config, checkpoint, output, control, stage, seed, k, dataset, status, status_path):
    name = f"{dataset}_seed_{seed}_K_{k}"
    job = output / f"stage_{stage}" / name
    job.mkdir(parents=True, exist_ok=True)
    local = {**config, "data": {**config["data"], "name": dataset}}
    bundle = prepare_survset(local, seed, max(config["experiment"]["context_sizes"]))
    splits_path = job / "splits.npz"
    if not splits_path.exists():
        np.savez_compressed(splits_path, **bundle["split_indices"])
    adapter = None
    methods = config["experiment"]["methods"]
    train = config["training"]
    for method in methods:
        result_path = job / f"results_{method}.json"
        if result_path.exists():
            continue
        if time.time() >= deadline(control):
            return False
        status.update(current_method=method, current_job=[stage, seed, k, dataset], updated_utc=now(),
                      elapsed_seconds=time.time() - status["started_epoch"])
        atomic_json(status_path, status)
        started = time.perf_counter()
        try:
            if method == "coxph_l2":
                metrics, _ = _cox_diagnostic(bundle, config, {})
                record = {"c_index_ipcw": metrics["c_index_ipcw"], "ibs": metrics["ibs"],
                          "metric_reasons": {"c_index": metrics["c_index_reason"], "ibs": metrics["ibs_reason"]},
                          "completed_steps": 0, "best_validation_loss": None, "reference_rows": len(bundle["train"]),
                          "optimization_seconds": time.perf_counter() - started}
            else:
                if adapter is None:
                    adapter = SurvPFNAdapter(checkpoint, device="cpu",
                                             categorical_mask=bundle["reference"].categorical_mask)
                instance = METHODS[method](adapter, context_size=k, seed=seed,
                    initialization=train["initialization"], context_space=train.get("context_space") if method in {"tunetables", "riskset-listwise"} else None, steps=stage, batch_size=train["batch_size"],
                    learning_rate=train["learning_rate"], validation_every=train["validation_every"],
                    patience=None, lambda_risk=train["lambda_risk"] if method == "riskset-tune" else 0.,
                    auxiliary_strength=train["auxiliary_strength"], risk_bins=train["risk_bins"],
                    n_time_strata=train["n_time_strata"], censoring_weight_floor=train["censoring_weight_floor"],
                    diagnostics_every=train["diagnostics_every"], deadline_control_path=control)
                state_path = output / "state" / dataset / str(seed) / str(k) / method / "training.pt"
                if method in {"random", "knn"}:
                    state_path = None
                instance.fit(bundle["reference"], bundle["queries"], bundle["validation"],
                             risk_reference_data=bundle["risk_reference"], state_path=state_path,
                             resume=bool(state_path and state_path.exists()))
                completed_steps = max((r["step"] for r in instance.history), default=0)
                if method not in {"random", "knn"} and completed_steps < stage:
                    return False
                prediction = instance.predict(bundle["test"].X)
                ev = config["evaluation"]
                metrics = evaluate_predictions(bundle["train"].time, bundle["train"].event,
                    bundle["test"].time, bundle["test"].event, prediction["risk"], prediction["times"],
                    prediction["survival"], n_grid=ev["time_grid_points"],
                    lower_quantile=ev["lower_time_quantile"], horizon_quantile=ev["horizon_quantile"])
                context_path = job / "contexts" / (method + ".pt")
                instance.save(context_path)
                state = getattr(instance, "_auxiliary_state", None)
                record = {"c_index_ipcw": metrics["c_index_ipcw"], "ibs": metrics["ibs"],
                          "metric_reasons": {"c_index": metrics["c_index_reason"], "ibs": metrics["ibs_reason"]},
                          "completed_steps": completed_steps, "best_validation_loss": instance.best_validation_loss,
                          "start_step": instance.start_step,
                          "optimization_seconds": instance.optimization_seconds,
                          "inference_seconds": instance.inference_seconds,
                          "auxiliary_alpha": state["alpha"] if state else None,
                          "risk_bin_coverage": instance.diagnostics.get("risk_bin_coverage"),
                          "adjusted_queries": instance.diagnostics.get("adjusted_queries"),
                          "context_path": str(context_path.relative_to(output)),
                          "state_path": str(state_path.relative_to(output)) if state_path else None}
            record.update(context_space=(instance.context_space if method not in {"coxph_l2"} else "input"),
                          status="complete", dataset=dataset, seed=seed, K=k, stage=stage,
                          method=method, finished_utc=now(), wall_seconds=time.perf_counter() - started)
            try:
                import psutil
                record["memory_bytes"] = int(psutil.Process(os.getpid()).memory_info().rss)
            except ImportError:
                record["memory_bytes"] = None
            atomic_json(result_path, record)
            print(f"{now()} completed stage={stage} seed={seed} K={k} dataset={dataset} method={method}", flush=True)
        except Exception as exc:
            failure = {"status": "failed", "dataset": dataset, "seed": seed, "K": k, "stage": stage,
                       "method": method, "error": repr(exc), "finished_utc": now()}
            atomic_json(result_path, failure)
            status.setdefault("failures", []).append(failure)
            atomic_json(status_path, status)
            print(f"{now()} failed {failure}", flush=True)
    return all(json.loads((job / f"results_{method}.json").read_text()).get("status") == "complete"
               for method in methods)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path, control, lock_path = output / "status.json", output / "control.json", output / "driver.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        old_pid = int(lock_path.read_text())
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            lock_path.unlink()
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        else:
            raise RuntimeError(f"pilot already running with PID {old_pid}")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    config = load_config(args.config)
    config["training"].setdefault("context_space", "embedding")
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    status = {**prior, "state": "running", "started_epoch": prior.get("started_epoch", time.time()),
              "started_utc": prior.get("started_utc", now()), "updated_utc": now(),
              "completed_jobs": prior.get("completed_jobs", []), "failures": prior.get("failures", [])}
    if not control.exists():
        atomic_json(control, {"deadline_epoch": args.deadline_epoch, "updated_utc": now()})
    atomic_json(status_path, status)
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; latest checkpoint is available")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    done = threading.Event()
    def timed_report():
        delay = max(0., status["started_epoch"] + 3 * 3600 - time.time())
        if not done.wait(delay):
            report(output, dict(status), "partial_3h")
            status["partial_report_written_utc"] = now()
            atomic_json(status_path, status)
    timer = threading.Thread(target=timed_report, daemon=True)
    timer.start()
    def heartbeat():
        while not done.wait(10):
            with LOCK:
                status.update(updated_utc=now(), elapsed_seconds=time.time() - status["started_epoch"],
                              deadline_epoch=deadline(control))
                atomic_json(status_path, status)
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        metadata_file(output, config, args.checkpoint)
        eligible = eligibility(config, output)
        if eligible != config["experiment"]["datasets"]:
            raise ValueError("one or more configured datasets failed preflight; see eligibility.json")
        completed = {tuple(x) for x in status["completed_jobs"]}
        all_jobs = [(stage, seed, k, dataset) for stage in config["experiment"]["stage_steps"]
                    for seed in config["experiment"]["seeds"] for k in config["experiment"]["context_sizes"]
                    for dataset in eligible]
        for job in all_jobs:
            if job in completed:
                continue
            if time.time() >= deadline(control):
                break
            success = run_job(config, Path(args.checkpoint), output, control, *job, status, status_path)
            if success:
                completed.add(job)
            status.update(completed_jobs=[list(x) for x in sorted(completed)], updated_utc=now(),
                          elapsed_seconds=time.time() - status["started_epoch"])
            aggregate(output)
            status["coverage_estimate"] = estimate_progress(output, config, all_jobs, completed, control)
            atomic_json(status_path, status)
        state = ("complete_with_failures" if status["failures"] else "complete") if len(completed) == len(all_jobs) else (
            "budget_exhausted" if time.time() >= deadline(control) else "complete_with_failures")
        status.update(state=state, finished_utc=now(), pending_jobs=len(all_jobs)-len(completed),
                      elapsed_seconds=time.time()-status["started_epoch"], current_job=None, current_method=None)
        done.set()
        timer.join(timeout=1)
        monitor.join(timeout=1)
        report(output, status, "final")
        atomic_json(status_path, status)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), updated_utc=now(),
                      elapsed_seconds=time.time()-status["started_epoch"])
        atomic_json(status_path, status)
        raise
    finally:
        done.set()
        timer.join(timeout=1)
        monitor.join(timeout=1)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

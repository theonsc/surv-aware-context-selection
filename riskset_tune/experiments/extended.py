"""Resumable staged SurvPFN pilot with a timed partial report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import pandas as pd
import yaml

from riskset_tune.experiments.runner import load_config, prepare_survset, run_experiment
from riskset_tune.contexts import initialize_context


AGGREGATE_LOCK = threading.RLock()
STATUS_LOCK = threading.RLock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    with STATUS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def read_rows(output: Path):
    frames = []
    for path in sorted(output.glob("stage_*/*/results.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            frame["stage"] = int(path.parent.parent.name.split("_", 1)[1])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_aggregates(output: Path):
    rows = read_rows(output)
    if rows.empty:
        return rows
    rows.to_csv(output / "all_stage_results.csv", index=False)
    # Paired comparisons are grouped at identical dataset, K, seed and stage.
    comparison_rows = []
    base_keys = ["dataset", "K", "seed", "stage"]
    for keys, group in rows.groupby(base_keys, dropna=False):
        indexed = {(str(r["method"]), float(r["lambda_risk"])): r
                   for r in group.to_dict("records")}
        random = indexed.get(("random", 0.0))
        tune = indexed.get(("tunetables", 0.0))
        requested_steps = int(keys[3])
        if tune is None or int(tune["completed_steps"]) < requested_steps:
            continue
        for lam in (0.01, 0.1, 1.0):
            risk = indexed.get(("riskset-tune", lam))
            if risk is None or int(risk["completed_steps"]) < requested_steps:
                continue
            for baseline in (random, tune):
                if baseline is None:
                    continue
                comparison_rows.append({**dict(zip(base_keys, keys)),
                                        "lambda_risk": lam, "baseline": baseline["method"],
                                        "C-index difference": None})
                item = comparison_rows[-1]
                item["C-index difference"] = float(risk["C-index IPCW"] - baseline["C-index IPCW"])
                item["C-index relative %"] = 100 * item["C-index difference"] / abs(float(baseline["C-index IPCW"]))
                item["IBS difference"] = float(risk["IBS"] - baseline["IBS"])
                item["IBS relative %"] = 100 * item["IBS difference"] / abs(float(baseline["IBS"]))
    pd.DataFrame(comparison_rows).to_csv(output / "paired_comparisons.csv", index=False)

    # Select lambda only by validation NLL; test metrics are attached after selection.
    selected = rows[rows.method.eq("riskset-tune") &
                    (rows.completed_steps >= rows.stage)].copy()
    selected = selected.sort_values(["dataset", "K", "seed", "stage", "best_validation_loss", "lambda_risk"])
    selected = selected.groupby(base_keys, as_index=False, dropna=False).first()
    selected.to_csv(output / "validation_selected_lambda.csv", index=False)
    selected_comparisons = []
    for method in ("random", "tunetables"):
        base = rows[rows.method.eq(method)].copy()
        if method == "tunetables":
            base = base[base.completed_steps >= base.stage]
        merged = selected.merge(base, on=base_keys, suffixes=("_selected", "_baseline"))
        for record in merged.to_dict("records"):
            item = {key: record[key] for key in base_keys}
            item.update(baseline=method, lambda_risk=record["lambda_risk_selected"],
                        selected_validation_nll=record["best_validation_loss_selected"],
                        C_index_difference=float(record["C-index IPCW_selected"] - record["C-index IPCW_baseline"]),
                        IBS_difference=float(record["IBS_selected"] - record["IBS_baseline"]))
            selected_comparisons.append(item)
    pd.DataFrame(selected_comparisons).to_csv(output / "paired_selected_lambda.csv", index=False)
    rows[rows.method.eq("coxph_l2")].to_csv(output / "cox_diagnostic.csv", index=False)
    return rows


def write_aggregates(output: Path):
    with AGGREGATE_LOCK:
        return _write_aggregates(output)


def write_report(output: Path, status: dict, label: str):
    rows = write_aggregates(output)
    report = output / ("partial_report_3h.md" if label == "partial_3h" else "final_report.md")
    lines = [f"# Piloto ampliado — {label}", "", f"Estado registrado em UTC: {utc_now()}",
             f"Tempo decorrido: {status.get('elapsed_seconds', 0) / 3600:.2f} horas",
             f"Combinações completas: {len(status.get('completed_jobs', []))}", ""]
    if rows.empty:
        lines += ["Ainda não há combinações finalizadas para resumir.", ""]
    else:
        complete_rows = rows[rows.method.isin(["random", "coxph_l2"]) |
                             (rows.completed_steps >= rows.stage)]
        lines += ["## Métricas médias por dataset, K, etapa, método e λ", "",
                  complete_rows.groupby(["dataset", "K", "stage", "method", "lambda_risk"], dropna=False)
                  [["C-index IPCW", "IBS", "best_validation_loss", "completed_steps"]]
                  .mean(numeric_only=True).round(6).to_markdown(), "",
                  "## Comparações pareadas por seed", "",
                  "Ver `paired_comparisons.csv` para cada λ e `paired_selected_lambda.csv` para λ escolhido pela validação; Δ C-index positivo e Δ IBS negativo favorecem RiskSet-Tune.",
                  "O λ selecionado usa exclusivamente NLL de validação; ver `validation_selected_lambda.csv`.", ""]
    active_states = []
    import torch
    for state_path in sorted((output / "state").glob("**/training.pt"),
                             key=lambda item: item.stat().st_mtime, reverse=True)[:12]:
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            active_states.append({"fit": str(state_path.relative_to(output)),
                                  "step": state.get("step"),
                                  "last_validation_nll": state.get("best_loss")})
        except (OSError, RuntimeError, EOFError):
            continue
    if status.get("current_job") or active_states:
        lines += ["## Progresso do ajuste em andamento", "",
                  f"Combinação atual: `{status.get('current_job', 'sem ajuste ativo')}`", "",
                  "Últimos checkpoints de treino (o passo registrado pode estar até 49 passos atrás):", "",
                  pd.DataFrame(active_states).to_markdown(index=False) if active_states else "Nenhum checkpoint de ajuste salvo ainda.", ""]
    lines += ["Os seeds compartilham indivíduos entre divisões; não são observações independentes.",
              "Os valores são exploratórios e cada base tem seu próprio split de teste.", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    if not rows.empty:
        history_by_fit = {}
        for path in sorted(output.glob("stage_*/*/contexts/**/*.pt")):
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
            relative = path.relative_to(output)
            stage = int(relative.parts[0].split("_", 1)[1])
            fit_key = (relative.parts[1], path.stem)
            if fit_key in history_by_fit and history_by_fit[fit_key][0] >= stage:
                continue
            history_frames = []
            artifact_name = path.stem
            lambda_value = None
            if "_lambda_" in artifact_name:
                lambda_value = artifact_name.rsplit("_lambda_", 1)[-1].replace("p", ".")
            curve_label = (f"riskset-tune λ={lambda_value}" if lambda_value is not None
                           else payload["method"])
            for item in payload.get("history", []):
                history_frames.append({"artifact": str(relative), "stage": stage,
                                       "method": payload["method"], "lambda_risk": lambda_value,
                                       "curve_label": curve_label, **item})
            history_by_fit[fit_key] = (stage, history_frames)
        history_frames = [row for _, fit_rows in history_by_fit.values() for row in fit_rows]
        if history_frames:
            histories = pd.DataFrame(history_frames)
            histories.to_csv(output / "training_diagnostics.csv", index=False)
            if "validation_loss" in histories:
                os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
                os.environ.setdefault("MPLBACKEND", "Agg")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                val = histories.dropna(subset=["validation_loss"])
                fig, ax = plt.subplots(figsize=(9, 5))
                for method, grp in val.groupby("curve_label"):
                    curve = grp.groupby("step")["validation_loss"].mean()
                    ax.plot(curve.index, curve.values, label=method)
                ax.set(xlabel="Passos", ylabel="NLL de validação", title="Curvas de validação (execuções concluídas)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output / "validation_curves.png", dpi=140)
                plt.close(fig)


def eligibility(config, output):
    records, eligible = [], []
    for name in config["experiment"].get("datasets", []):
        per_seed = []
        try:
            for seed in config["experiment"]["seeds"]:
                local = {**config, "data": {**config["data"], "name": name}}
                bundle = prepare_survset(local, seed, max(config["experiment"]["context_sizes"]))
                checks = {"seed": seed, "n_train": len(bundle["train"]),
                          "reference_rows": len(bundle["reference"]),
                          "events_context": int(bundle["reference"].event.sum()),
                          "continuous_features": int(bundle["reference"].continuous_mask.sum()),
                          "categorical_features": int(bundle["reference"].categorical_mask.sum()),
                          "train_events": int(bundle["train"].event.sum()),
                          "train_censored": int((~bundle["train"].event).sum()),
                          "validation_events": int(bundle["validation"].event.sum()),
                          "validation_censored": int((~bundle["validation"].event).sum()),
                          "test_events": int(bundle["test"].event.sum()),
                          "test_censored": int((~bundle["test"].event).sum())}
                context_checks = []
                strategy = config["training"].get("initialization", "random")
                for k in config["experiment"]["context_sizes"]:
                    context = initialize_context(bundle["reference"],
                                                 np.arange(len(bundle["reference"])),
                                                 k, seed, strategy,
                                                 config["training"].get("n_time_strata", 4))
                    context_checks.append({"K": k, "events": int(context.event.sum()),
                                           "continuous_features": int(context.trainable_mask.sum())})
                checks["contexts_by_K"] = context_checks
                if checks["reference_rows"] < max(config["experiment"]["context_sizes"]):
                    raise ValueError("reference pool is too small for K=64")
                if checks["events_context"] < 3 or any(c["events"] < 3 for c in context_checks) or min(checks[k] for k in
                   ("train_events", "train_censored", "validation_events", "validation_censored",
                    "test_events", "test_censored")) < 1:
                    raise ValueError("split lacks required events or censoring")
                if checks["continuous_features"] < 1:
                    raise ValueError("no continuous feature available for context optimization")
                per_seed.append(checks)
            eligible.append(name)
            records.append({"dataset": name, "eligible": True, "seeds": per_seed})
        except Exception as exc:
            records.append({"dataset": name, "eligible": False, "reason": repr(exc)})
    atomic_json(output / "eligibility.json", records)
    return eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-hours", type=float)
    parser.add_argument("--partial-hours", type=float)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    if prior.get("state") == "complete":
        print("extended pilot already complete", flush=True)
        return
    lock_path = output / "driver.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            pid = int(lock_path.read_text())
            os.kill(pid, 0)
        except ProcessLookupError:
            lock_path.unlink(missing_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except (ValueError, FileNotFoundError):
            lock_path.unlink(missing_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        else:
            raise RuntimeError(f"another extended pilot owns {lock_path}")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    config = load_config(args.config)
    budget_hours = args.budget_hours if args.budget_hours is not None else config["experiment"].get("budget_hours", 10)
    partial_hours = args.partial_hours if args.partial_hours is not None else config["experiment"].get("partial_report_hours", 3)
    config["training"]["state_dir"] = str(output / "state")
    started_epoch = prior.get("started_epoch", time.time())
    deadline = started_epoch + budget_hours * 3600
    config["training"]["deadline_epoch"] = deadline
    config["evaluation"]["make_plots"] = False
    config_path = output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    eligible = prior.get("eligible_datasets") or eligibility(config, output)
    completed = set(tuple(x) for x in prior.get("completed_jobs", []))
    status = {**prior, "state": "running", "started_epoch": started_epoch,
              "started_utc": prior.get("started_utc", utc_now()),
              "target_hours": budget_hours, "partial_after_hours": partial_hours,
              "eligible_datasets": eligible, "completed_jobs": [list(x) for x in completed],
              "updated_utc": utc_now()}
    atomic_json(status_path, status)
    report_triggered = threading.Event()

    def stop_cleanly(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}; saving at the latest 50-step checkpoint")

    signal.signal(signal.SIGTERM, stop_cleanly)
    signal.signal(signal.SIGINT, stop_cleanly)

    def report_timer():
        delay = max(0.0, started_epoch + partial_hours * 3600 - time.time())
        if report_triggered.wait(delay):
            return
        status["elapsed_seconds"] = time.time() - started_epoch
        write_report(output, status, "partial_3h")
        status["partial_report_written_utc"] = utc_now()
        status["updated_utc"] = utc_now()
        atomic_json(status_path, status)

    timer = threading.Thread(target=report_timer, name="partial-report", daemon=True)
    timer.start()
    stages = config["experiment"].get("stage_steps", [200, 1000, 2000, 5000])
    job_count = len(stages) * len(config["experiment"]["seeds"]) * len(config["experiment"]["context_sizes"]) * len(eligible)
    try:
        for stage in stages:
            for seed in config["experiment"]["seeds"]:
                for k in config["experiment"]["context_sizes"]:
                    for dataset in eligible:
                        job = (stage, seed, k, dataset)
                        if job in completed:
                            continue
                        if time.time() >= deadline:
                            break
                        local = yaml.safe_load(yaml.safe_dump(config))
                        local["data"]["name"] = dataset
                        local["experiment"]["seeds"] = [seed]
                        local["experiment"]["context_sizes"] = [k]
                        local["training"]["steps"] = stage
                        job_out = output / f"stage_{stage}" / f"{dataset}_seed_{seed}_K_{k}"
                        status.update(current_job=list(job), elapsed_seconds=time.time() - started_epoch,
                                      updated_utc=utc_now())
                        atomic_json(status_path, status)
                        print(f"{utc_now()} start stage={stage} dataset={dataset} seed={seed} K={k}", flush=True)
                        run_experiment(local, args.checkpoint, job_out)
                        job_rows = pd.read_csv(job_out / "results.csv")
                        optimized = job_rows[job_rows.method.isin(["tunetables", "riskset-tune"])]
                        if len(optimized) and (optimized.completed_steps >= stage).all():
                            completed.add(job)
                        status["completed_jobs"] = [list(x) for x in sorted(completed)]
                        status.update(elapsed_seconds=time.time() - started_epoch, updated_utc=utc_now())
                        write_aggregates(output)
                        atomic_json(status_path, status)
                        print(f"{utc_now()} complete stage={stage} dataset={dataset} seed={seed} K={k}", flush=True)
                    if time.time() >= deadline:
                        break
                if time.time() >= deadline:
                    break
            if time.time() >= deadline:
                break
        status["state"] = "complete" if len(completed) == job_count else "budget_exhausted"
        status["pending_jobs"] = job_count - len(completed)
        status.pop("current_job", None)
        status["finished_utc"] = utc_now()
        status["elapsed_seconds"] = time.time() - started_epoch
        report_triggered.set()
        timer.join(timeout=1)
        write_report(output, status, "final")
        atomic_json(status_path, status)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), updated_utc=utc_now(),
                      elapsed_seconds=time.time() - started_epoch,
                      completed_jobs=[list(x) for x in sorted(completed)])
        atomic_json(status_path, status)
        raise
    finally:
        report_triggered.set()
        timer.join(timeout=1)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULT_COLUMNS = ["dataset", "K", "seed", "method", "C-index IPCW", "IBS",
                  "optimization_seconds", "inference_seconds", "memory_bytes"]


def results_table(rows):
    frame = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in frame:
            frame[col] = pd.NA
    return frame[RESULT_COLUMNS + [c for c in frame if c not in RESULT_COLUMNS]]


def save_results(rows, output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = results_table(rows)
    csv_tmp, md_tmp = out / "results.csv.tmp", out / "results.md.tmp"
    frame.to_csv(csv_tmp, index=False)
    md_tmp.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    csv_tmp.replace(out / "results.csv")
    md_tmp.replace(out / "results.md")
    keys = ["dataset", "K", "seed"]
    if {"dataset", "K", "seed", "method"}.issubset(frame.columns):
        # A λ sweep has repeated RiskSet-Tune rows for one seed/K. Do not let
        # pivot_table silently choose one λ and present it as the paired result.
        method_counts = frame.groupby(keys + ["method"], dropna=False).size()
        if len(method_counts) and int(method_counts.max()) > 1:
            if "lambda_risk" not in frame:
                return frame
            paired_keys = keys + ["lambda_risk"]
        else:
            paired_keys = keys
        for metric in ("C-index IPCW", "IBS"):
            if metric in frame:
                paired = frame.pivot_table(index=paired_keys, columns="method", values=metric, aggfunc="first")
                if "tunetables" in paired and "riskset-tune" in paired:
                    paired["riskset_minus_tunetables"] = paired["riskset-tune"] - paired["tunetables"]
                if "random" in paired and "tunetables" in paired:
                    paired["tunetables_minus_random"] = paired["tunetables"] - paired["random"]
                paired.to_csv(out / (metric.lower().replace(" ", "_") + "_paired.csv"))
    return frame


def plot_results(frame: pd.DataFrame, output_path: str | Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frame.boxplot(column="C-index IPCW", by="method", ax=ax)
    ax.set_title("Paired experiment results")
    ax.set_ylabel("C-index IPCW")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

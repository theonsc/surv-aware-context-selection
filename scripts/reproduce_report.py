# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = ["matplotlib==3.9.4", "numpy==2.0.2"]
# ///
"""Generate the report's two tables and four figures from benchmark results."""

import argparse
import csv
import json
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

BENCHMARKS = {"survset": "SurvSet", "finsurvival": "FinSurvival"}
KS = (32, 64, 128)
SEEDS = (100, 101, 102)
METHODS = (
    ("random", ""), ("kmeans", ""), ("knn", ""), ("tunetables", ""),
    ("riskset-listwise", "0.1"), ("riskset-listwise", "0.2"),
)
LABELS = ("Aleatório", "k-means", "kNN", "Prompt tuning",
          "Prompt tuning + ord. (α = 0,1)", "Prompt tuning + ord. (α = 0,2)")
METRICS = {"ibs": ("ibs", 60), "c_index_ipcw": ("c-index", 25)}
T_CRITICAL_DF2 = 4.3026527299


def report_cases(run_dir):
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    groups = {
        "survset": [("survset", name) for name in config["survset"]],
        "finsurvival": [
            ("finsurvival", f"{start}_{end}".replace(" ", "_"))
            for start, end in config["finsurvival"]
        ],
    }
    if any(len(cases) != 6 or len(set(cases)) != 6 for cases in groups.values()):
        raise ValueError("Expected six datasets for each benchmark")
    return groups


def dataset_size(job, benchmark):
    if benchmark == "survset":
        splits = ("train", "validation", "test")
        return sum(len(np.load(job / f"{split}_rows.npy", mmap_mode="r", allow_pickle=False))
                   for split in splits)
    audit = json.loads((job / "audit.json").read_text())
    return audit["train"]["joined"] + audit["test"]["joined"]


def read_sizes(run_dir, groups):
    sizes = {}
    for group, cases in groups.items():
        for case in cases:
            counts = {dataset_size(run_dir.joinpath(*case, str(seed)), group) for seed in SEEDS}
            if len(counts) != 1 or min(counts) < 1:
                raise ValueError(f"Invalid or inconsistent size across seeds: {case}")
            sizes[case] = counts.pop()
        cases.sort(key=lambda case: (sizes[case], case[1]))
    return sizes, groups


def read_results(path, cases):
    expected = set(product(cases, KS, SEEDS, METHODS))
    selected = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            method = row["method"]
            method = "riskset-listwise" if method == "NLL_order" else method
            method = (method, row["listwise_alpha"])
            if method not in METHODS:
                continue
            stage = 0 if method in METHODS[:3] else 2000
            if int(row["stage"]) != stage:
                continue
            key = ((row["benchmark"], row["dataset"]), int(row["K"]), int(row["seed"]), method)
            if key not in expected or key in selected or row["status"] != "complete":
                raise ValueError(f"Unexpected, duplicate or incomplete result: {key}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if any(not isfinite(v) or not 0 <= v <= 1 for v in values.values()):
                raise ValueError(f"Invalid metric: {key}")
            if method == METHODS[0] and any(v == 0 for v in values.values()):
                raise ValueError(f"Zero random baseline makes relative change undefined: {key}")
            selected[key] = values
    missing = expected - selected.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} results, including {min(missing)}")
    return selected


def benchmark_rows(records, groups):
    for group, cases in groups.items():
        columns = list(product(METRICS, KS))
        summaries = {}
        for method, (metric, k) in product(METHODS, columns):
            by_seed = [mean(records[case, k, seed, method][metric] for case in cases)
                       for seed in SEEDS]
            summaries[method, metric, k] = mean(by_seed), T_CRITICAL_DF2 * stdev(by_seed) / len(SEEDS)**.5
        for method, label in zip(METHODS, LABELS):
            for metric, k in columns:
                avg, margin = summaries[method, metric, k]
                choose = min if metric == "ibs" else max
                best = choose(METHODS, key=lambda m: summaries[m, metric, k][0])
                yield BENCHMARKS[group], label, metric, k, f"{avg:.4f}", f"{margin:.4f}", method == best


def compression_rows(records, sizes, groups):
    for group, cases in groups.items():
        for case in cases:
            tuning = mean(records[case, 32, seed, METHODS[3]]["ibs"] for seed in SEEDS)
            candidates = [mean(records[case, 128, seed, method]["ibs"] for seed in SEEDS)
                          for method in METHODS[:3]]
            best = min(range(len(candidates)), key=candidates.__getitem__)
            yield (BENCHMARKS[group], case[1], sizes[case], f"{tuning:.4f}",
                   LABELS[best], f"{candidates[best]:.4f}", tuning < candidates[best])


def write_table(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def relative_change(records, case, k, seed, method, metric):
    base = records[case, k, seed, METHODS[0]][metric]
    return 100 * (records[case, k, seed, method][metric] - base) / base


def percent(value, suffix=""):
    return f"{value:+.1f}".replace("-", "−").replace(".", ",") + suffix


def plot_benchmarks(records, groups, output):
    labels = [label.replace(" (α", "\n(α") for label in LABELS]
    ink = '#192c3e'
    columns = [0, 1, 2, 3.2, 4.2, 5.2]
    for metric, (name, limit) in METRICS.items():
        cmap = LinearSegmentedColormap.from_list('efeito', ['#096268', '#f4f3f0', '#963923'] if metric == 'ibs' else ['#963923', '#f4f3f0', '#096268'])
        fig = plt.figure(figsize=(6.4, 2.5), facecolor='white')
        ax = fig.add_axes([.01, .19, .98, .79], xlim=(-2.05, 5.74), ylim=(5.55, -1.9))
        ax.axis('off')
        norm = Normalize(-limit, limit)
        for x, title in [(1, 'SurvSet'), (4.2, 'FinSurvival')]:
            ax.text(x, -1.55, title, ha='center', va='center', fontsize=12, weight='bold', color=ink)
            ax.hlines(-1.24, x-1.48, x+1.48, color='#a5b3be', lw=.7)
        for x, k in zip(columns, [32,64,128]*2):
            ax.text(x, -.87, f'K = {k}', ha='center', va='center', fontsize=10, color=ink)
        ax.text(-2, -.87, 'Método', va='center', fontsize=10, weight='bold', color=ink)
        for y, (method, label) in enumerate(zip(METHODS, labels)):
            ax.text(-.55, y, label, ha='right', va='center', fontsize=7.2, color=ink, linespacing=.95)
            for x, group, k in zip(columns, ['survset']*3+['finsurvival']*3, [32,64,128]*2):
                value = mean(relative_change(records, case, k, seed, method, metric)
                             for case in groups[group] for seed in SEEDS)
                color = cmap(norm(value))
                ax.add_patch(Rectangle((x-.48,y-.47), .96,.94, facecolor=color))
                label_value = ('0,0%' if method == METHODS[0] else percent(value))
                ax.text(x, y, label_value, ha='center', va='center', fontsize=10, color='#101b25')
        cax = fig.add_axes([.41,.105,.42,.035])
        ticks = [-limit,-limit/2,0,limit/2,limit]
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=ticks)
        bar.set_ticklabels([f'{t:g}%'.replace('.',',').replace('-','−') for t in ticks])
        bar.ax.tick_params(length=2,pad=2,labelsize=8,colors=ink)
        bar.outline.set_visible(False)
        fig.savefig(output / f'{name}-versus-aleatorio-por-colecao.png',dpi=450)
        plt.close(fig)


def plot_datasets(records, sizes, groups, output):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': '#1d3043',
        'xtick.color': '#30485d', 'ytick.color': '#30485d', 'figure.facecolor': 'white',
    })
    norm = TwoSlopeNorm(vmin=-180, vcenter=0, vmax=180)

    def task_label(case):
        name = case[1].replace('_Account_Liquidated', '_Liquidated').replace('_', ' →\n')
        return name + '\n' + f'{sizes[case]:,}'.replace(',', '.')

    for metric, (name, _) in METRICS.items():
        cmap = LinearSegmentedColormap.from_list('reference', ['#126669', '#f6f5f2', '#a0412e'] if metric == 'ibs' else ['#a0412e', '#f6f5f2', '#126669'])
        fig, axes = plt.subplots(2, 1, figsize=(5.5, 7.23))
        fig.subplots_adjust(left=.298, right=.995, top=.956, bottom=.139, hspace=.38)
        for (group, cases), ax in zip(groups.items(), axes):
            values = np.array([
                [mean(relative_change(records, case, k, seed, method, metric)
                      for seed in SEEDS) for case in cases]
                for k, method in product(KS, METHODS)
            ])
            data = np.concatenate([values, values.mean(axis=1, keepdims=True)], axis=1)
            heat = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
            ax.set_title(BENCHMARKS[group],
                         loc='left', fontsize=9.4, weight='bold', pad=5)
            ax.set_yticks(range(18), list(LABELS) * 3)
            ax.set_xticks(range(7), [task_label(c) for c in cases] + ['Média'])
            ax.tick_params(axis='y', length=0, labelsize=5.8, pad=3.7)
            ax.tick_params(axis='x', length=0, labelsize=6.8, pad=4)
            ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 18, 1), minor=True)
            ax.tick_params(which='minor', length=0)
            ax.grid(which='minor', color='white', linewidth=.65)
            ax.axvline(5.5, color='white', linewidth=2.3)
            for y in (5.5, 11.5):
                ax.axhline(y, color='white', linewidth=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for i, k in enumerate(KS):
                ax.text(-.42, i * 6 + 2.5, f'K = {k}', transform=ax.get_yaxis_transform(),
                        fontsize=8.1, fontweight='bold', ha='left', va='center')
            for row, col in product(range(18), range(7)):
                value = data[row, col]
                rgb = cmap(norm(value))[:3]
                brightness = sum(w * c for w, c in zip((.2126, .7152, .0722), rgb))
                start = (row // 6) * 6
                best = value == (data[start:start + 6, col].min() if metric == 'ibs' else data[start:start + 6, col].max())
                ax.text(col, row, ('0,0%' if abs(value) < .05 else percent(value, '%')), ha='center', va='center', fontsize=7.0,
                        color='white' if brightness < .51 else '#1d3043',
                        weight='bold' if best else 'normal')
        cax = fig.add_axes([.318, .043, .65, .012])
        bar = fig.colorbar(heat, cax=cax, orientation='horizontal', ticks=[-180,-90,0,90,180],
                           format=FuncFormatter(lambda x, _: f'{x:g}%'.replace('-', '−')))
        bar.outline.set_visible(False)
        cax.tick_params(length=0, labelsize=7.2, pad=3)
        fig.savefig(output / f'{name}-versus-aleatorio-por-tarefa.png', dpi=380, facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Benchmark output directory")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN_DIR/relatorio")
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "relatorio"
    try:
        groups = report_cases(args.run_dir)
        records = read_results(args.run_dir / "results.csv", set().union(*groups.values()))
        sizes, groups = read_sizes(args.run_dir, groups)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    tables = output / "tabelas"
    figures = output / "figuras"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    write_table(tables / "ibs-c-index-por-benchmark.csv",
                ("benchmark", "metodo", "metrica", "K", "media", "margem_ic95", "melhor"),
                benchmark_rows(records, groups))
    write_table(tables / "compressao-32-vs-128.csv",
                ("benchmark", "dataset", "n", "ibs_prompt_tuning_k32", "melhor_baseline_k128",
                 "ibs_baseline_k128", "prompt_tuning_vence"),
                compression_rows(records, sizes, groups))
    plot_benchmarks(records, groups, figures)
    plot_datasets(records, sizes, groups, figures)
    print(f"{len(records)} combinações validadas, 2 tabelas e 4 figuras em {output}")


if __name__ == "__main__":
    main()

#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
mkdir -p runs
if [ "$#" -gt 1 ]; then
    printf 'Usage: %s [output-directory]\n' "$0" >&2
    exit 2
fi
if [ "$#" -eq 1 ]; then
    run_dir=$1
    if [ ! -f "$run_dir/manifest.json" ]; then
        printf 'No benchmark manifest in %s\n' "$run_dir" >&2
        exit 2
    fi
else
    run_dir=$(mktemp -d runs/full_benchmark_fixed_alpha.XXXXXX)
fi

printf 'Diretório da execução: %s\n' "$run_dir"
export SURVPFN_SOURCE="$PWD/.local/upstream/SurvPFN"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python -m riskset_tune.experiments.full_benchmark \
    --config configs/full_benchmark_fixed_alpha.yaml \
    --checkpoint artifacts/checkpoints/survpfn_nr.pt \
    --fin-root .local/finsurvival \
    --output "$run_dir" >> "$run_dir/benchmark.log" 2>&1

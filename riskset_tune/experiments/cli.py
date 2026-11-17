from __future__ import annotations

import argparse
from pathlib import Path

from .runner import load_config, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run paired RiskSet-Tune experiments")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--checkpoint", help="local SurvPFN .pt checkpoint")
    parser.add_argument("--output", default="runs/pilot")
    parser.add_argument("--list-results", action="store_true",
                        help="aggregate an existing results.csv without loading a model")
    args = parser.parse_args(argv)
    if args.list_results:
        import pandas as pd
        csv_path = Path(args.output) / "results.csv"
        if not csv_path.is_file():
            parser.error("results file does not exist: %s" % csv_path)
        print(pd.read_csv(csv_path).to_markdown(index=False))
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required for model runs; checkpoint downloads are not automatic")
    config = load_config(args.config)
    run_experiment(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

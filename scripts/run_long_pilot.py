"""Run balanced, paired pilot batches until a wall-clock budget is nearly spent."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import yaml

from riskset_tune.evaluation import plot_results, save_results


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=14400)
    parser.add_argument("--estimate-round-seconds", type=float, required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sizes = config["experiment"]["context_sizes"]
    first_seed = config["experiment"]["seeds"][0]
    started = time.monotonic()
    status = {"state": "running", "started_utc": now(), "target_seconds": args.seconds,
              "seed_start": first_seed, "context_sizes": sizes, "completed_batches": [],
              "completed_rounds": 0, "initial_estimate_round_seconds": args.estimate_round_seconds}
    write_status(output / "status.json", status)
    frames = []
    round_durations = []
    seed = first_seed
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(output / ".matplotlib")
    env["MPLBACKEND"] = "Agg"
    try:
        while True:
            remaining = args.seconds - (time.monotonic() - started)
            estimate = (sum(round_durations[-3:]) / len(round_durations[-3:])
                        if round_durations else args.estimate_round_seconds)
            if remaining < estimate * 1.03:
                break
            round_start = time.monotonic()
            for k in sizes:
                batch = output / "batches" / ("seed_%s_K_%s" % (seed, k))
                batch.mkdir(parents=True, exist_ok=False)
                config["experiment"]["seeds"] = [seed]
                config["experiment"]["context_sizes"] = [k]
                config_path = batch / "input_config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                status.update(current_seed=seed, current_K=k, updated_utc=now())
                write_status(output / "status.json", status)
                print(f"{now()} starting seed={seed} K={k}", flush=True)
                subprocess.run([sys.executable, "-m", "riskset_tune.experiments.cli",
                                "--config", str(config_path), "--checkpoint", args.checkpoint,
                                "--output", str(batch)], check=True, env=env)
                frames.append(pd.read_csv(batch / "results.csv"))
                save_results(pd.concat(frames, ignore_index=True).to_dict("records"), output)
                status["completed_batches"].append({"seed": seed, "K": k, "finished_utc": now()})
                status.update(updated_utc=now(), elapsed_seconds=time.monotonic() - started)
                write_status(output / "status.json", status)
                print(f"{now()} finished seed={seed} K={k}", flush=True)
            round_durations.append(time.monotonic() - round_start)
            status["completed_rounds"] += 1
            status["round_seconds"] = round_durations
            write_status(output / "status.json", status)
            seed += 1
        if frames:
            os.environ.update(MPLCONFIGDIR=env["MPLCONFIGDIR"], MPLBACKEND="Agg")
            plot_results(pd.concat(frames, ignore_index=True), output / "c_index_by_method.png")
        status.update(state="complete", finished_utc=now(), elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        status.update(state="failed", finished_utc=now(), elapsed_seconds=time.monotonic() - started,
                      error=repr(exc))
        raise
    finally:
        status.pop("current_seed", None)
        status.pop("current_K", None)
        write_status(output / "status.json", status)


if __name__ == "__main__":
    main()

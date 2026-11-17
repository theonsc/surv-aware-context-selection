"""Install or inspect the user LaunchAgent for the extended pilot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time
import yaml


LABEL = "com.risksettune.extended-pilot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-hours", type=float)
    parser.add_argument("--partial-hours", type=float)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output = Path(args.output).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    # Keep the venv symlink path: resolving it points at the base interpreter,
    # which drops pyvenv.cfg and cannot import the project's installed package.
    python = root / ".venv" / "bin" / "python"
    driver = (root / "scripts" / "run_extended_pilot.py").resolve()
    source = (root / ".local" / "upstream" / "SurvPFN").resolve()
    if not all(path.exists() for path in (config, checkpoint, python, driver, source)):
        raise FileNotFoundError("config, checkpoint, Python environment, driver or pinned source is missing")
    run_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    budget_hours = args.budget_hours if args.budget_hours is not None else run_config["experiment"].get("budget_hours", 10)
    partial_hours = args.partial_hours if args.partial_hours is not None else run_config["experiment"].get("partial_report_hours", 3)
    output.mkdir(parents=True, exist_ok=True)
    args_vector = ["/usr/bin/caffeinate", "-i", str(python), "-u", str(driver),
                   "--config", str(config), "--checkpoint", str(checkpoint),
                   "--output", str(output), "--budget-hours", str(budget_hours),
                   "--partial-hours", str(partial_hours)]
    plist = {"Label": LABEL, "ProgramArguments": args_vector,
             "WorkingDirectory": str(root), "RunAtLoad": True,
             "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 30,
             "EnvironmentVariables": {
                 "SURVPFN_SOURCE": str(source),
                 "UV_CACHE_DIR": str(root / ".uv-cache"),
                 "MPLCONFIGDIR": str(output / ".matplotlib"),
                 "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
             },
             "StandardOutPath": str(output / "driver.log"),
             "StandardErrorPath": str(output / "driver.err")}
    plist_path = output / "LaunchAgent.plist"
    with plist_path.open("wb") as stream:
        plistlib.dump(plist, stream, sort_keys=True)
    domain = f"gui/{os.getuid()}"
    print(f"LaunchAgent plist: {plist_path}")
    if not args.start:
        return
    existing = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"],
                              capture_output=True, text=True)
    if existing.returncode == 0:
        raise RuntimeError(f"LaunchAgent {LABEL} is already loaded")
    subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"{domain}/{LABEL}"], check=True)
    time.sleep(2)
    check = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        raise RuntimeError("LaunchAgent did not remain loaded after bootstrap")
    status_path = output / "status.json"
    deadline = time.time() + 45
    while time.time() < deadline:
        if status_path.is_file():
            import json
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") in {"running", "complete", "budget_exhausted"}:
                print(f"LaunchAgent confirmed; current state: {status['state']}")
                return
            if status.get("state") == "failed":
                raise RuntimeError("pilot reported startup failure; see " + str(output / "status.json"))
        time.sleep(1)
    raise TimeoutError("LaunchAgent is loaded but no pilot status appeared within 45 seconds")


if __name__ == "__main__":
    main()

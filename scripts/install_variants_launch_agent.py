"""Create and optionally launch the independent macOS variants pilot."""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import time
from pathlib import Path

LABEL = "com.risksettune.variants-pilot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--implementation-start', type=float, required=True)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output, config, checkpoint = Path(args.output).resolve(), Path(args.config).resolve(), Path(args.checkpoint).resolve()
    python = root / '.venv' / 'bin' / 'python'
    source = root / '.local' / 'upstream' / 'SurvPFN'
    if not all(p.exists() for p in (python, source, config, checkpoint)):
        raise FileNotFoundError('missing Python environment, upstream source, config or checkpoint')
    domain = f'gui/{os.getuid()}'
    if args.start and subprocess.run(['launchctl', 'print', f'{domain}/{LABEL}'], capture_output=True).returncode == 0:
        raise RuntimeError(f'{LABEL} is already loaded; refusing duplicate execution')
    planned_deadline = min(args.implementation_start + 12 * 3600, time.time() + 7 * 3600)
    if planned_deadline <= time.time():
        raise ValueError('the authorized 12-hour budget has already expired')
    output.mkdir(parents=True, exist_ok=True)
    control = output / 'control.json'
    if not control.exists():
        tmp = control.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({'deadline_epoch': planned_deadline,
                                  'implementation_start_epoch': args.implementation_start,
                                  'total_budget_hours': 12, 'execution_cap_hours': 7}, indent=2) + '\n')
        os.replace(tmp, control)
    else:
        planned_deadline = float(json.loads(control.read_text())["deadline_epoch"])
        if planned_deadline <= time.time():
            raise ValueError('the saved deadline has expired; update control.json before resuming')
    plist = {'Label': LABEL, 'RunAtLoad': True, 'KeepAlive': False,
             'WorkingDirectory': str(root),
             'ProgramArguments': ['/usr/bin/caffeinate', '-i', str(python), '-u', '-m',
                'riskset_tune.experiments.variants_pilot', '--config', str(config),
                '--checkpoint', str(checkpoint), '--output', str(output),
                '--deadline-epoch', repr(planned_deadline)],
             'EnvironmentVariables': {'SURVPFN_SOURCE': str(source),
                'UV_CACHE_DIR': str(root / '.uv-cache'), 'MPLCONFIGDIR': str(output / '.matplotlib'),
                'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'},
             'StandardOutPath': str(output / 'driver.log'),
             'StandardErrorPath': str(output / 'driver.err')}
    path = output / 'LaunchAgent.plist'
    with path.open('wb') as stream:
        plistlib.dump(plist, stream, sort_keys=True)
    print('LaunchAgent plist:', path)
    print('Deadline epoch:', planned_deadline)
    if not args.start:
        return
    subprocess.run(['launchctl', 'bootstrap', domain, str(path)], check=True)
    # kickstart without -k does not terminate an already-running job.
    subprocess.run(['launchctl', 'kickstart', f'{domain}/{LABEL}'], check=True)
    for _ in range(60):
        check = subprocess.run(['launchctl', 'print', f'{domain}/{LABEL}'], capture_output=True, text=True)
        status_path = output / 'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('state') == 'failed':
                raise RuntimeError(f"pilot reported startup failure: {status.get('error')}")
            if status.get('state') == 'running' and 'state = running' in check.stdout:
                print('Confirmed: LaunchAgent running and status updated')
                return
        time.sleep(1)
    raise TimeoutError('LaunchAgent did not confirm a running state within 60 seconds')


if __name__ == '__main__':
    main()

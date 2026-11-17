"""Check every planned calibration against the local official checkpoint."""
import argparse
import json
import os
import time
from pathlib import Path

from riskset_tune.experiments.runner import load_config, prepare_survset
from riskset_tune.methods import RiskSetListwiseTune, RiskSetScoreTune
from riskset_tune.models import SurvPFNAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    results = []
    for seed in config['experiment']['seeds']:
        for dataset in config['experiment']['datasets']:
            local = {**config, 'data': {**config['data'], 'name': dataset}}
            bundle = prepare_survset(local, seed, max(config['experiment']['context_sizes']))
            adapter = SurvPFNAdapter(args.checkpoint, device='cpu',
                                     categorical_mask=bundle['reference'].categorical_mask)
            for k in config['experiment']['context_sizes']:
                for cls in (RiskSetScoreTune, RiskSetListwiseTune):
                    started = time.perf_counter()
                    try:
                        method = cls(adapter, context_size=k, seed=seed, initialization='random', steps=1,
                            batch_size=32, learning_rate=.001, validation_every=1,
                            diagnostics_every=1, auxiliary_strength=.1, risk_bins=4)
                        method.fit(bundle['reference'], bundle['queries'], bundle['validation'],
                                   risk_reference_data=bundle['risk_reference'])
                        state = method._auxiliary_state
                        row = {'dataset': dataset, 'seed': seed, 'K': k, 'method': cls.name,
                               'status': 'ok', 'alpha': state['alpha'],
                               'coverage': method.diagnostics.get('risk_bin_coverage'),
                               'seconds': time.perf_counter()-started}
                    except Exception as exc:
                        row = {'dataset': dataset, 'seed': seed, 'K': k, 'method': cls.name,
                               'status': 'failed', 'reason': repr(exc),
                               'seconds': time.perf_counter()-started}
                    results.append(row)
                    path = Path(args.output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(results, indent=2) + '\n')
                    os.replace(tmp, path)
                    print(row, flush=True)
    if any(row['status'] != 'ok' for row in results):
        raise SystemExit('one or more planned calibrations failed')


if __name__ == '__main__':
    main()

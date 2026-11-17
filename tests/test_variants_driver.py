import json
import time

import pandas as pd
import pytest
import torch

from riskset_tune.experiments.variants_pilot import aggregate, atomic_json, deadline, report
from riskset_tune.methods import TuneTables
from test_variants import _Adapter, dataset


def test_aggregate_pairs_only_complete_methods_in_same_stage(tmp_path):
    for stage, method, status, ibs in [(200, 'tunetables', 'complete', .2),
                                       (200, 'riskset-score', 'complete', .19),
                                       (200, 'riskset-listwise', 'failed', .01),
                                       (1000, 'riskset-score', 'complete', .18)]:
        job = tmp_path / f'stage_{stage}' / 'data_seed_42_K_32'
        atomic_json(job / f'results_{method}.json', {'status': status, 'dataset': 'data', 'K': 32,
                    'seed': 42, 'stage': stage, 'method': method, 'ibs': ibs, 'c_index_ipcw': .6})
    rows, pairs = aggregate(tmp_path)
    assert len(rows) == 3
    assert len(pairs) == 1 and pairs.iloc[0]['method'] == 'riskset-score'
    assert pairs.iloc[0]['ibs_delta'] == pytest.approx(-.01)
    assert pd.read_csv(tmp_path / 'paired_comparisons.csv').shape[0] == 1
    report(tmp_path, {'state': 'running', 'completed_jobs': [], 'current_job': [200, 42, 32, 'data']}, 'partial_3h')
    assert 'riskset-score' in (tmp_path / 'partial_3h_report.md').read_text()


def test_deadline_file_is_live_and_stops_with_checkpoint(tmp_path):
    control = tmp_path / 'control.json'
    atomic_json(control, {'deadline_epoch': time.time() - 1})
    assert deadline(control) < time.time()
    data = dataset()
    reference, queries, validation = data.subset(range(50)), data.subset(range(50, 70)), data.subset(range(70, 90))
    path = tmp_path / 'state.pt'
    method = TuneTables(_Adapter(), context_size=20, seed=17, initialization='random', steps=6,
                        batch_size=12, validation_every=2, deadline_control_path=control)
    method.fit(reference, queries, validation, risk_reference_data=data, state_path=path)
    assert method.budget_exhausted and torch.load(path, weights_only=False)['step'] == 0
    future = time.time() + 60
    atomic_json(control, {'deadline_epoch': future})
    assert deadline(control) == future
    resumed = TuneTables(_Adapter(), context_size=20, seed=17, initialization='random', steps=6,
                         batch_size=12, validation_every=2, deadline_control_path=control)
    resumed.fit(reference, queries, validation, risk_reference_data=data, state_path=path, resume=True)
    assert not resumed.budget_exhausted
    assert torch.load(path, weights_only=False)['step'] == 6

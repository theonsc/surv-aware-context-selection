"""Compare predictive NLL and time-dependent ordering in saved contexts."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from riskset_tune.experiments.runner import prepare_survset
from riskset_tune.losses.survival import ipcw_weights
from riskset_tune.models import SurvPFNAdapter


def predictions(adapter, context, data, train):
    adapter.categorical_mask = context['categorical_mask']
    adapter.fit_context(context['context_X'], context['context_time'], context['context_event'])
    with torch.no_grad():
        logits = adapter.forward_logits(context['context_X'], context['context_time'],
                                        context['context_event'], data.X)
        weights = ipcw_weights(train.time, train.event, data.time, data.event, eps=.05)
        nll = float(adapter.loss(logits, data.time, data.event, weights))
        anchors = np.flatnonzero(data.event)
        risk = -torch.log(adapter.survival_tensor(logits, data.time[anchors]).clamp(1e-6, 1-1e-6)).cpu().numpy()
    comparable = data.time[None, :] > data.time[anchors, None]
    anchor_risk = risk[anchors, np.arange(len(anchors))][None, :]
    difference = anchor_risk - risk
    scores = (difference > 0).astype(float) + .5 * (difference == 0)
    return nll, scores.T[comparable], comparable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/riskset_variants_20261105_112723'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/checkpoints/survpfn_nr.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/tunetables_ordering_diagnostic'))
    args = parser.parse_args()
    config = yaml.safe_load((args.run / 'resolved_config.yaml').read_text())
    exp = config['experiment']
    adapter = SurvPFNAdapter(args.checkpoint)
    result_rows, pair_rows = [], []
    for dataset, seed, k in itertools.product(exp['datasets'], exp['seeds'], exp['context_sizes']):
        local = {**config, 'data': {**config['data'], 'name': dataset}}
        bundle = prepare_survset(local, seed, max(exp['context_sizes']))
        contexts = []
        for stage in exp['stage_steps']:
            job = args.run / f'stage_{stage}' / f'{dataset}_seed_{seed}_K_{k}'
            with np.load(job / 'splits.npz') as saved:
                for name, indices in bundle['split_indices'].items():
                    if not np.array_equal(saved[name], indices):
                        raise ValueError(f'Split divergente: {job} {name}')
            context = torch.load(job / 'contexts/tunetables.pt', map_location='cpu', weights_only=False)
            if any(torch.equal(context['context_X'], old['context_X']) for _, old, _ in contexts):
                continue
            measures = {}
            for split in ('validation', 'test'):
                nll, scores, comparable = predictions(adapter, context, bundle[split], bundle['train'])
                measures[split] = (nll, scores, comparable)
            history = [row['validation_loss'] for row in context['history'] if row.get('validation_loss') is not None]
            if not np.isclose(measures['validation'][0], min(history), atol=2e-5, rtol=1e-5):
                raise ValueError(f'NLL de validação divergente: {job}')
            contexts.append((stage, context, measures))
            result_rows.append(dict(dataset=dataset, seed=seed, K=k, stage=stage,
                                    nll_validation=measures['validation'][0], nll_test=measures['test'][0],
                                    concordance_test=measures['test'][1].mean() if len(measures['test'][1]) else np.nan,
                                    comparable_pairs=len(measures['test'][1])))
        for (a, _, m1), (b, _, m2) in itertools.combinations(contexts, 2):
            assert np.array_equal(m1['test'][2], m2['test'][2])
            score1, score2 = m1['test'][1], m2['test'][1]
            pair_rows.append(dict(dataset=dataset, seed=seed, K=k, stage_a=a, stage_b=b,
                                  delta_nll_validation=m2['validation'][0]-m1['validation'][0],
                                  delta_nll_test=m2['test'][0]-m1['test'][0],
                                  delta_concordance_test=score2.mean()-score1.mean() if len(score1) else np.nan,
                                  changed_pair_fraction=np.mean(score1 != score2) if len(score1) else np.nan,
                                  comparable_pairs=len(score1)))
        print(f'{dataset} seed={seed} K={k}: {len(contexts)} contextos', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results, pairs = pd.DataFrame(result_rows), pd.DataFrame(pair_rows)
    results.to_csv(args.output / 'contexts.csv', index=False)
    pairs.to_csv(args.output / 'paired_contexts.csv', index=False)
    eligible = pairs[pairs.delta_nll_validation.abs() <= .01].copy()
    eligible['similar_test'] = eligible.delta_nll_test.abs() <= .01
    eligible['different_order'] = eligible.delta_concordance_test.abs() >= .01
    eligible.to_csv(args.output / 'selected_validation.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, group in eligible.groupby('dataset'):
        ax.scatter(group.delta_nll_test, 100*group.delta_concordance_test, label=dataset, alpha=.8)
    ax.axvspan(-.01, .01, color='grey', alpha=.12)
    ax.axhline(0, color='black', linewidth=.6)
    ax.axvline(0, color='black', linewidth=.6)
    ax.set(xlabel='Diferença de NLL em teste por consulta', ylabel='Diferença de concordância temporal (pontos percentuais)', title='TuneTables adaptado: contextos com NLL de validação próxima')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / 'nll_vs_concordance.png', dpi=180)
    both = eligible[eligible.similar_test]
    summary = f"Pares distintos: {len(pairs)}\nSelecionados na validação: {len(eligible)}\nNLL também próxima em teste: {len(both)}\nDestes, diferença de concordância >= 1 ponto percentual: {both.different_order.sum()}\n"
    summary += '\nPor dataset (NLL próxima em teste, diferença >= 1 pp):\n'
    summary += both.groupby('dataset').different_order.agg(['size', 'sum']).to_string() + '\n'
    (args.output / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()

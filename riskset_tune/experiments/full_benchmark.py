"""Resumable comparison of context selection across SurvSet and FinSurvival."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib import metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from ..contexts import initialize_context
from ..data.types import SurvivalDataset
from ..evaluation import evaluate_predictions
from ..methods import KNNContext, TuneTables, RiskSetListwiseTune
from ..models import SurvPFNAdapter
from .runner import prepare_survset


def file_hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(repr((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def verify_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Reproduction inputs differ from {path}. Use a new output directory.')
    else:
        write_json(path, value)


def record_run(out, config_path, cfg, checkpoint, upstream, fin_root):
    package = Path(__file__).resolve().parents[1]
    sources = sorted(package.rglob('*.py'))
    configs = {'benchmark.yaml': Path(config_path), 'riskset_variants.yaml': Path('configs/riskset_variants.yaml')}
    manifest = {
        'config': cfg,
        'config_sha256': {name: file_hash(path) for name, path in configs.items()},
        'checkpoint_sha256': file_hash(checkpoint),
        'code_sha256': {str(p.relative_to(package)): file_hash(p) for p in sources},
        'upstream_sha256': {str(p.relative_to(upstream)): file_hash(p) for p in sorted((upstream / 'src').rglob('*.py'))},
        'fin_sha256': {str(p.relative_to(fin_root)): file_hash(p) for p in sorted(fin_root.rglob('*.rds'))},
        'fin_source_manifest': file_hash(fin_root / 'source.json') if (fin_root / 'source.json').is_file() else None,
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions())),
        'python': platform.python_version(), 'platform': platform.platform(),
        'device': 'cpu', 'torch_threads': torch.get_num_threads(),
        'torch_deterministic': torch.are_deterministic_algorithms_enabled(),
        'native_threadpools': [{k: v for k, v in pool.items() if k != 'filepath'} for pool in threadpool_info()],
    }
    path = out / 'manifest.json'
    if not path.exists() and any(out.glob('*/*/*/*/*/*.json')):
        raise ValueError('Existing results lack a manifest. Use a new output directory.')
    verify_json(path, manifest)
    snapshot = out / 'reproduction'
    for name, source in configs.items():
        snapshot.mkdir(exist_ok=True)
        shutil.copyfile(source, snapshot / name)
    for source in sources:
        dest = snapshot / 'riskset_tune' / source.relative_to(package)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)


def record_splits(job, bundle):
    splits = {name: bundle[name] for name in ('train', 'reference', 'queries', 'validation', 'test')}
    row_ids = bundle.get('split_indices', {})
    hashes = {}
    for name, data in splits.items():
        ids = row_ids[name] if name in row_ids else np.asarray(data.ids, dtype=str)
        hashes[name] = {key: array_hash(value) for key, value in
                        {'rows': ids, 'X': data.X, 'time': data.time, 'event': data.event,
                         'categorical_mask': data.categorical_mask}.items()}
        target = job / f'{name}_rows.npy'
        if target.exists() and array_hash(np.load(target, allow_pickle=False)) != hashes[name]['rows']:
            raise ValueError(f'Saved row order differs from reconstructed split {target}')
        if not target.exists():
            temp = target.with_suffix('.tmp.npy')
            np.save(temp, ids, allow_pickle=False)
            os.replace(temp, target)
    verify_json(job / 'splits.json', hashes)


def _clean(frame):
    before = len(frame)
    frame = frame.dropna(subset=['id']).drop_duplicates()
    conflicts = frame['id'].duplicated(keep=False)
    frame = frame.loc[~conflicts].copy()
    return frame, {'input_rows': before, 'conflicting_rows': int(conflicts.sum()), 'output_rows': len(frame)}


def prepare_fin(root, index, outcome, seed, test_size=2048, validation_size=1024):
    import pyreadr

    base = Path(root) / 'Survival_Data' / index
    prepared, audit = {}, {}
    for split in ('train', 'test'):
        features = pd.concat([next(iter(pyreadr.read_r(str(p)).values()))
                              for p in sorted((base / ('X_' + split)).glob('*.rds'))], ignore_index=True)
        labels = next(iter(pyreadr.read_r(str(base / outcome / ('y_' + split + '.rds'))).values()))
        features, a = _clean(features)
        labels, b = _clean(labels)
        prepared[split] = labels.merge(features, on='id', how='inner', validate='one_to_one')
        audit[split] = {'features': a, 'labels': b, 'joined': len(prepared[split])}
    train, test = prepared['train'], prepared['test']
    if set(train.id).intersection(test.id):
        raise ValueError('official train and test IDs overlap')
    test = test.iloc[train_test_split(np.arange(len(test)), train_size=test_size, random_state=701,
                                      stratify=test.status)[0]].copy()
    idx = np.arange(len(train))
    train_idx, val_idx = train_test_split(idx, test_size=validation_size, random_state=seed, stratify=train.status)
    ref_idx, query_idx = train_test_split(train_idx, test_size=.5, random_state=seed + 2,
                                          stratify=train.status.iloc[train_idx])
    excluded = {'id', 'timeDiff', 'status', 'Index Event', 'Outcome Event', 'type', 'pool', 'user', 'timestamp', 'indexEvent', 'outcomeEvent'}
    columns = [c for c in train if c in test and c not in excluded and not c.startswith('exo')]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in columns if c not in numeric]
    for frame in (train, test):
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        for c in categorical:
            frame[c] = frame[c].astype(object).where(frame[c].notna(), np.nan)
    encoder = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), numeric),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='__missing__'),
                              OneHotEncoder(handle_unknown='ignore')), categorical)], sparse_threshold=0)
    encoded = encoder.fit_transform(train.iloc[train_idx])
    pca = PCA(n_components=min(10, encoded.shape[1]), random_state=0).fit(encoded)
    def transform(frame):
        x = pca.transform(encoder.transform(frame)).astype(np.float32)
        return SurvivalDataset(x, frame.timeDiff.to_numpy(), frame.status.to_numpy(), ids=frame.id.to_numpy())
    all_train, test_data = transform(train), transform(test)
    audit['test_sample_ids_sha256'] = hashlib.sha256('\n'.join(test.id.astype(str)).encode()).hexdigest()
    return {'reference': all_train.subset(ref_idx), 'queries': all_train.subset(query_idx),
            'validation': all_train.subset(val_idx), 'train': all_train.subset(train_idx),
            'risk_reference': all_train.subset(train_idx), 'test': test_data, 'audit': audit}


def context(data, k, seed, strategy):
    # Repair the rare invalid initial draw deterministically and identically across methods.
    for offset in range(1000):
        try:
            result = initialize_context(data, np.arange(len(data)), k, seed + offset, strategy)
            result.metadata['seed_offset'] = offset
            return result
        except ValueError as exc:
            if 'fewer than three events' not in str(exc):
                raise
    raise ValueError('could not initialize context with three events')


def kmeans_context(data, k, seed):
    cat = data.categorical_mask
    parts = [data.X[:, ~cat]]
    for j in np.flatnonzero(cat):
        parts.append((data.X[:, j, None] == np.unique(data.X[:, j])[None, :]).astype(float) / np.sqrt(2))
    z = np.concatenate(parts, axis=1)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(z)
    dist = (np.square(model.cluster_centers_).sum(axis=1)[:, None]
            + np.square(z).sum(axis=1)[None, :] - 2 * model.cluster_centers_ @ z.T)
    selected = []
    for row in dist:
        selected.append(next(int(i) for i in np.argsort(row, kind='stable') if i not in selected))
    selected = np.array(selected)
    needed = max(0, 3 - int(data.event[selected].sum()))
    if needed:
        available = np.flatnonzero(data.event & ~np.isin(np.arange(len(data)), selected))
        positions = np.flatnonzero(~data.event[selected])
        for position in positions[-needed:]:
            chosen = available[np.argmin(dist[position, available])]
            selected[position] = chosen
            available = available[available != chosen]
    return selected


def batch_predict(adapter, X, batch=128):
    risks, curves, times = [], [], None
    for start in range(0, len(X), batch):
        result = adapter.predict(X[start:start + batch])
        risks.extend(result['risk'])
        curves.extend(result['survival'])
        times = result['times']
    return {'risk': np.asarray(risks), 'times': times, 'survival': np.asarray(curves)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/full_benchmark.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--benchmark', choices=['survset', 'finsurvival', 'all'], default='all')
    parser.add_argument('--dataset')
    parser.add_argument('--checkpoint', default='artifacts/checkpoints/survpfn_nr.pt')
    parser.add_argument('--fin-root', default='.local/finsurvival')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            run(args, cfg, out)


def run(args, cfg, out):
    context_selection = cfg.get('context_selection', 'all')
    if context_selection not in ('all', 'random'):
        raise ValueError('context_selection must be all or random')
    methods = [
        ('random', 'random'), ('stratified', 'stratified'), ('kmeans', None),
        ('knn', None), ('tunetables', 'random'), ('tunetables', 'stratified'),
        ('riskset-listwise', 'random'), ('riskset-listwise', 'stratified'),
    ]
    if context_selection == 'random':
        methods = [(method, init) for method, init in methods if init == 'random']

    adapter = SurvPFNAdapter(args.checkpoint, device='cpu')
    record_run(out, args.config, cfg, args.checkpoint, adapter.upstream_source, Path(args.fin_root))
    jobs = [('survset', x) for x in cfg['survset']] + [('finsurvival', x) for x in cfg['finsurvival']]
    for benchmark, name in jobs:
        label = name if isinstance(name, str) else '_'.join(name).replace(' ', '_')
        if args.benchmark not in ('all', benchmark) or (args.dataset and args.dataset != label):
            continue
        for seed in cfg['seeds']:
            if benchmark == 'survset':
                spec = yaml.safe_load(Path('configs/riskset_variants.yaml').read_text())
                spec['data']['name'] = name
                bundle = prepare_survset(spec, seed, 256)
            else:
                bundle = prepare_fin(args.fin_root, *name, seed, cfg['test_size'], cfg['validation_size'])
            job = out / benchmark / label / str(seed); job.mkdir(parents=True, exist_ok=True)
            record_splits(job, bundle)
            if 'audit' in bundle:
                (job / 'audit.json').write_text(json.dumps(bundle['audit'], indent=2))
            for k in cfg['context_sizes']:
                for method, init in methods:
                    tag = method + ('_' + init if init and method not in ('random','stratified') else '')
                    folder = job / str(k) / tag; folder.mkdir(parents=True, exist_ok=True)
                    stages = cfg['stages'] if method in ('tunetables','riskset-listwise') else [0]
                    for stage in stages:
                        result_path = folder / f'{stage}.json'
                        if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'complete':
                            continue
                        started = time.time()
                        actual_seed = seed
                        try:
                            if method in ('tunetables','riskset-listwise'):
                                cls = TuneTables if method == 'tunetables' else RiskSetListwiseTune
                                initial = context(bundle['reference'], k, seed, init)
                                ids = initial.source_indices
                                actual_seed = initial.metadata['seed']
                                instance = cls(adapter, context_size=k, seed=actual_seed, initialization=init,
                                               steps=stage, batch_size=32, learning_rate=.001, validation_every=50,
                                               context_space=cfg.get("context_space", "embedding"))
                                state = folder / 'training.pt'
                                instance.fit(bundle['reference'], bundle['queries'], bundle['validation'],
                                             risk_reference_data=bundle['train'], state_path=state, resume=state.exists())
                                if instance.history[-1]['step'] != stage:
                                    raise ValueError(f'Training state does not match requested stage {stage}')
                                instance.save(folder / f'context_{stage}.pt')
                                adapter.categorical_mask = instance.context.categorical_mask
                                adapter.fit_context(instance.context.X, instance.context.time, instance.context.event)
                                prediction = batch_predict(instance, bundle['test'].X)
                            elif method == 'knn':
                                instance = KNNContext(adapter, context_size=k, seed=seed).fit(bundle['reference'],bundle['queries'],bundle['validation'])
                                prediction = instance.predict(bundle['test'].X, record_neighbors=True)
                                ids = prediction['source_indices']
                            else:
                                if method == 'kmeans':
                                    ids = kmeans_context(bundle['reference'], k, seed)
                                    ref = bundle['reference']; x, t, e = ref.X[ids], ref.time[ids], ref.event[ids]
                                else:
                                    c = context(bundle['reference'], k, seed, init); x, t, e = c.X, c.time, c.event
                                    ids = c.source_indices
                                    actual_seed = c.metadata['seed']
                                adapter.categorical_mask = bundle['reference'].categorical_mask
                                adapter.fit_context(x, t, e)
                                prediction = batch_predict(adapter, bundle['test'].X)
                            metrics = evaluate_predictions(bundle['train'].time, bundle['train'].event,
                                bundle['test'].time, bundle['test'].event, prediction['risk'], prediction['times'], prediction['survival'])
                            selection = folder / f'selection_{stage}.npy'
                            temp = selection.with_suffix('.tmp.npy')
                            np.save(temp, ids, allow_pickle=False)
                            os.replace(temp, selection)
                            result = {'status':'complete', 'benchmark':benchmark, 'dataset':label, 'seed':seed,
                                      'K':k, 'method':method, 'initialization':init, 'stage':stage,
                                      'context_space': cfg.get('context_space', 'embedding') if method in ('tunetables', 'riskset-listwise') else 'input',
                                      'effective_seed':actual_seed, 'selection_sha256':array_hash(ids),
                                      'predictions_sha256':{name:array_hash(prediction[name])
                                                            for name in ('risk', 'times', 'survival')},
                                      'elapsed_seconds':time.time()-started, **metrics}
                        except Exception as exc:
                            result = {'status':'failed','error':repr(exc),'benchmark':benchmark,'dataset':label,
                                      'seed':seed,'K':k,'method':method,'initialization':init,'stage':stage}
                        write_json(result_path, result)
                        rows = [json.loads(path.read_text()) for path in sorted(out.glob('*/*/*/*/*/*.json'))]
                        pd.DataFrame(rows).to_csv(out / 'results.tmp', index=False)
                        os.replace(out / 'results.tmp', out / 'results.csv')
                        print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()

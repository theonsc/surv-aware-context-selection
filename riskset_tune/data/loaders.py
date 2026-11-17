from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import SurvivalDataset


def load_survset_dataset(name: str) -> SurvivalDataset:
    """Load one static SurvSet table without fitting preprocessing on all rows."""
    try:
        from SurvSet.data import SurvLoader
    except ImportError as exc:
        raise ImportError("Install riskset-tune[survset] to load SurvSet") from exc
    df = SurvLoader().load_dataset(ds_name=name)["df"].drop(columns=["time2"], errors="ignore")
    ids = df["pid"].to_numpy() if "pid" in df else np.arange(len(df))
    target_time = df["time"].to_numpy(dtype=np.float32)
    event = df["event"].to_numpy(dtype=bool)
    feature_cols = [c for c in df.columns if c.startswith("num_") or c.startswith("fac_")]
    cat_mask = np.asarray([c.startswith("fac_") for c in feature_cols], dtype=bool)
    matrix = np.empty((len(df), len(feature_cols)), dtype=np.float32)
    cat_values = {}
    for j, column in enumerate(feature_cols):
        vals = df[column].to_numpy()
        if cat_mask[j]:
            cat_values[j] = np.asarray([None if pd.isna(v) else v for v in vals], dtype=object)
            matrix[:, j] = 0.0
        else:
            matrix[:, j] = np.asarray(pd.to_numeric(vals, errors="coerce"), dtype=np.float32)
    return SurvivalDataset(matrix, target_time, event, ids=ids,
                           metadata={"dataset": name, "source": "SurvSet",
                                     "feature_names": feature_cols},
                           categorical_mask=cat_mask, categorical_values=cat_values)


def load_finsurvival_dataset(root: str | Path, index_event: str, outcome_event: str,
                             *, format: str = "csv", columns: dict | None = None,
                             preserve_official_split: bool = True) -> tuple[SurvivalDataset, SurvivalDataset | None]:
    """Read local FinSurvival train/test files. No data are downloaded."""
    root = Path(root)
    columns = columns or {"id": "id", "time": "timeDiff", "event": "status"}
    id_col, time_col, event_col = columns["id"], columns["time"], columns["event"]
    stem = "%s_%s" % (index_event, outcome_event)
    if format == "csv":
        reader = pd.read_csv
        ext = ".csv"
    elif format == "parquet":
        reader = pd.read_parquet
        ext = ".parquet"
    elif format == "rds":
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError("Install riskset-tune[finsurvival] for RDS files") from exc
        reader = lambda p: next(iter(pyreadr.read_r(str(p)).values()))
        ext = ".rds"
    else:
        raise ValueError("format must be csv, parquet or rds")
    train_path = root / (stem + "_train" + ext)
    test_path = root / (stem + "_test" + ext)
    if format == "rds" and not train_path.is_file():
        # Official DMLR layout: feature shards under X_{split} and outcomes in
        # <Index>/<Outcome>/y_<split>.rds. Join by ID as the authors do.
        base = root / "Survival_Data"
        idx_name, out_name = index_event.title(), outcome_event.title()

        def official(split):
            x_dir = base / idx_name / ("X_" + split)
            y_path = base / idx_name / out_name / ("y_" + split + ".rds")
            if not x_dir.is_dir() or not y_path.is_file():
                raise FileNotFoundError("expected FinSurvival split at %s and %s" % (x_dir, y_path))
            shards = [reader(p) for p in sorted(x_dir.glob("*.rds"))]
            if not shards:
                raise FileNotFoundError("no FinSurvival feature shards in %s" % x_dir)
            features = pd.concat(shards, ignore_index=True).dropna(subset=[id_col])
            excluded_features = {"id", "Index Event", "Outcome Event", "type", "pool",
                                 "user", "timestamp", "indexEvent", "outcomeEvent"}
            features = features.loc[:, [c for c in features
                                        if not str(c).startswith("exo") and c not in excluded_features]]
            labels = reader(y_path).dropna(subset=[id_col])
            merged = labels.merge(features, on=id_col, how="inner", suffixes=("", "_x"))
            return merged
        train_df, test_df = official("train"), official("test")
        test_path = None
    else:
        if not train_path.is_file():
            raise FileNotFoundError(train_path)
        train_df = reader(train_path)
        test_df = reader(test_path) if test_path.is_file() else None
    excluded = {id_col, time_col, event_col}
    if "features" in columns:
        feature_cols = columns["features"]
    else:
        feature_cols = [c for c in train_df.columns if c not in excluded]
    if test_df is not None:
        for col in feature_cols:
            if col not in test_df:
                test_df[col] = np.nan

    categorical_mask = np.zeros(len(feature_cols), dtype=bool)
    category_values = {}
    for j, col in enumerate(feature_cols):
        if (not pd.api.types.is_numeric_dtype(train_df[col].dtype)
                or col in columns.get("categorical_features", [])):
            categorical_mask[j] = True
            category_values[j] = {
                "train": np.asarray([None if pd.isna(v) else v for v in train_df[col]], dtype=object),
                "test": None if test_df is None else np.asarray(
                    [None if pd.isna(v) else v for v in test_df[col]], dtype=object),
            }

    def convert(frame):
        frame = frame.reindex(columns=feature_cols + [c for c in (id_col, time_col, event_col)
                                                       if c in frame.columns])
        matrix = np.empty((len(frame), len(feature_cols)), dtype=np.float32)
        for j, col in enumerate(feature_cols):
            if categorical_mask[j]:
                matrix[:, j] = 0.0
            else:
                matrix[:, j] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)
        return SurvivalDataset(matrix,
                              frame[time_col].to_numpy(dtype=np.float32),
                              frame[event_col].to_numpy(dtype=bool),
                              ids=frame[id_col].to_numpy() if id_col in frame else None,
                              metadata={"dataset": "FinSurvival:" + stem,
                                        "feature_names": list(feature_cols),
                                        "official_split": preserve_official_split},
                              categorical_mask=categorical_mask.copy(),
                              categorical_values={j: category_values[j]["train" if frame is train_df else "test"]
                                                  for j in category_values})
    return convert(train_df), None if test_df is None else convert(test_df)


def list_eligible_survset(*, min_rows: int = 500, min_features: int = 2,
                          max_features: int = 10, min_events: int = 10,
                          max_rows: int | None = None, first_only: bool = False):
    try:
        from SurvSet.data import SurvLoader
    except ImportError as exc:
        raise ImportError("Install riskset-tune[survset] to inspect SurvSet") from exc
    table = SurvLoader().df_ds.copy()
    table = table.loc[~table["is_td"].astype(bool)].copy()
    table["p"] = table["n_num"] + table["n_fac"]
    table = table.loc[(table["n"] >= min_rows) & table["p"].between(min_features, max_features)]
    if max_rows is not None:
        table = table.loc[table["n"] <= max_rows]
    candidates = []
    for name in sorted(table["ds"].tolist()):
        data = load_survset_dataset(name)
        if data.event.sum() >= min_events and (~data.event).sum() >= min_events and data.continuous_mask.any():
            candidates.append(name)
            if first_only:
                break
    return candidates

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import N_DELAY


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "total_consumption.csv"


def resolve_data_path(data_path: str | None = None) -> Path:
    path_value = data_path or os.environ.get("DATA_PATH") or str(DEFAULT_DATA_PATH)
    path = Path(path_value)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


@lru_cache(maxsize=4)
def _load_total_consumption_cached(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns or "consumption" not in df.columns:
        raise ValueError("total_consumption.csv must contain timestamp and consumption columns")

    df = df[["timestamp", "consumption"]].copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    return df


def load_total_consumption(data_path: str | None = None) -> pd.DataFrame:
    path = resolve_data_path(data_path)
    return _load_total_consumption_cached(str(path)).copy()


def get_dataset_info(data_path: str | None = None) -> dict[str, Any]:
    path = resolve_data_path(data_path)
    df = load_total_consumption(str(path))
    earliest_ts = df.index.min()
    latest_ts = df.index.max()
    earliest_date = earliest_ts.normalize()
    latest_date = latest_ts.normalize()
    latest_predictable_target = latest_date + pd.Timedelta(days=N_DELAY)

    try:
        display_path = str(path.relative_to(ROOT_DIR))
    except ValueError:
        display_path = str(path)

    return {
        "data_path": display_path,
        "row_count": int(len(df)),
        "available_actual_days": int(df.index.normalize().nunique()),
        "earliest_timestamp": earliest_ts.isoformat(),
        "latest_timestamp": latest_ts.isoformat(),
        "earliest_date": earliest_date.date().isoformat(),
        "latest_actual_date": latest_date.date().isoformat(),
        "latest_predictable_target_date": latest_predictable_target.date().isoformat(),
        "data_version": int(path.stat().st_mtime),
    }

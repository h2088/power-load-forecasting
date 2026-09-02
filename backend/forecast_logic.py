from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

for env_name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
]:
    os.environ.setdefault(env_name, "1")

warnings.filterwarnings(
    "ignore",
    message="Downcasting behavior in `replace` is deprecated",
    category=FutureWarning,
)

import numpy as np
import pandas as pd

from src.config import N_DELAY, SLOTS_PER_DAY, WINDOW_DAYS
from src.pipeline import predict, train

from backend.data_store import get_dataset_info, load_total_consumption, resolve_data_path


MODEL_NAME = "Scheme 1 Regime-aware Two-stage HistGradientBoosting"
MODEL_SUMMARY = (
    "Train on the latest 28-day window ending at D-6, predict daily level first, "
    "then predict intraday shape, with regime-aware lag features and anomaly handling."
)
MAJOR_FEATURE_GROUPS = [
    "D-6 / D-7 / D-14 / D-28 lag load values",
    "Daily level statistics and same-slot robust medians",
    "Regime flags for early_low_vol / high_zigzag / transition_drop / smooth_high_level",
    "Expected regime anchors and transition risk features",
    "Two-stage level-shape decomposition for the 96-slot curve",
]
REGIME_SEGMENTS = [
    "2026-01-01 to 2026-01-08: early_low_vol",
    "2026-01-09 to 2026-03-13: high_zigzag",
    "2026-03-14 to 2026-03-19: high_zigzag with shape shift",
    "2026-03-20 to 2026-03-23: transition_drop",
    "2026-03-24 to 2026-03-29: smooth_high_level",
]
DAILY_TOTAL_HISTORY_DAYS = 14

_PREDICTION_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_MAX_CACHE_SIZE = 32


class ForecastRequestError(ValueError):
    """Raised when the forecast request cannot be satisfied."""


def get_model_info() -> dict[str, Any]:
    dataset = get_dataset_info()
    return {
        "model_name": MODEL_NAME,
        "scheme": "方案1",
        "approach": MODEL_SUMMARY,
        "n_delay_days": N_DELAY,
        "training_window_days": WINDOW_DAYS,
        "slots_per_day": SLOTS_PER_DAY,
        "daily_total_history_days": DAILY_TOTAL_HISTORY_DAYS,
        "major_feature_groups": MAJOR_FEATURE_GROUPS,
        "regime_segments": REGIME_SEGMENTS,
        "serving_mode": (
            "On-demand retraining per requested target date using the latest "
            "available D-6 anchored history window."
        ),
        "dataset": dataset,
    }


def run_forecast(target_date: str | None = None) -> dict[str, Any]:
    dataset = get_dataset_info()
    resolved_target = _resolve_target_date(target_date, dataset)
    cache_key = (resolved_target.date().isoformat(), dataset["data_version"])
    cached = _PREDICTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    df = load_total_consumption()
    train_df, window_meta = _build_training_window(df, resolved_target)
    model, slot_means = train(train_df)
    result, _ = predict(train_df, model, slot_means, target_date=resolved_target)

    forecast_frame = _finalize_forecast_frame(result, resolved_target)
    actual_frame = _load_actual_curve(df, resolved_target)
    if not actual_frame.empty:
        forecast_frame = forecast_frame.merge(actual_frame, on="slot", how="left")
    else:
        forecast_frame["actual"] = np.nan

    summary = _build_summary(forecast_frame)
    daily_totals = _build_daily_total_context(
        df=df,
        target_date=resolved_target,
        predicted_daily_total=summary["predicted_daily_total"],
        lookback_days=DAILY_TOTAL_HISTORY_DAYS,
    )
    warnings_list = _build_warnings(window_meta, resolved_target, dataset)

    response = {
        "model": {
            "name": MODEL_NAME,
            "scheme": "方案1",
            "n_delay_days": N_DELAY,
            "training_window_days": WINDOW_DAYS,
            "data_path": dataset["data_path"],
        },
        "forecast": {
            "target_date": resolved_target.date().isoformat(),
            "cutoff_date": window_meta["cutoff_date"],
            "train_window_start": window_meta["train_window_start"],
            "train_window_end": window_meta["train_window_end"],
            "slots": _to_slot_records(forecast_frame),
            "daily_totals": daily_totals,
            "summary": summary,
            "data_quality": {
                "train_rows": window_meta["train_rows"],
                "expected_train_rows": window_meta["expected_train_rows"],
                "train_days_observed": window_meta["train_days_observed"],
                "train_coverage_ratio": window_meta["train_coverage_ratio"],
                "warnings": warnings_list,
            },
        },
    }

    if len(_PREDICTION_CACHE) >= _MAX_CACHE_SIZE:
        oldest_key = next(iter(_PREDICTION_CACHE))
        _PREDICTION_CACHE.pop(oldest_key, None)
    _PREDICTION_CACHE[cache_key] = response

    return response


def write_forecast_files(target_date: str | None = None, output_dir: str | None = None) -> dict[str, Any]:
    response = run_forecast(target_date)
    target = response["forecast"]["target_date"]
    output_root = Path(output_dir or "artifacts/forecasts").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(response["forecast"]["slots"])
    csv_path = output_root / f"forecast_{target}.csv"
    json_path = output_root / f"forecast_{target}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "target_date": target,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


def _resolve_target_date(target_date: str | None, dataset: dict[str, Any]) -> pd.Timestamp:
    latest_predictable_target = pd.Timestamp(dataset["latest_predictable_target_date"]).normalize()
    earliest_actual_date = pd.Timestamp(dataset["earliest_date"]).normalize()

    if target_date is None or str(target_date).strip() == "":
        return latest_predictable_target

    resolved = pd.Timestamp(target_date).normalize()
    min_target = earliest_actual_date + pd.Timedelta(days=N_DELAY)
    if resolved < min_target:
        raise ForecastRequestError(
            f"target_date must be on or after {min_target.date().isoformat()} "
            f"to respect the D-6 data availability rule."
        )
    if resolved > latest_predictable_target:
        raise ForecastRequestError(
            f"target_date must be on or before {latest_predictable_target.date().isoformat()} "
            "because data newer than D-6 is unavailable."
        )
    return resolved


def _build_training_window(df: pd.DataFrame, target_date: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff_date = target_date - pd.Timedelta(days=N_DELAY)
    window_start = cutoff_date - pd.Timedelta(days=WINDOW_DAYS - 1)
    train_df = df[
        (df.index >= window_start) &
        (df.index < cutoff_date + pd.Timedelta(days=1))
    ].copy()

    if train_df.empty:
        raise ForecastRequestError("No training data available for the requested target date.")

    train_rows = int(len(train_df))
    expected_rows = int(WINDOW_DAYS * SLOTS_PER_DAY)
    train_days_observed = int(train_df.index.normalize().nunique())
    coverage_ratio = round(train_rows / expected_rows, 4) if expected_rows else 0.0

    return train_df, {
        "cutoff_date": cutoff_date.date().isoformat(),
        "train_window_start": window_start.date().isoformat(),
        "train_window_end": cutoff_date.date().isoformat(),
        "train_rows": train_rows,
        "expected_train_rows": expected_rows,
        "train_days_observed": train_days_observed,
        "train_coverage_ratio": coverage_ratio,
    }


def _finalize_forecast_frame(result: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    frame = result[["slot", "predicted"]].copy()
    frame["slot"] = frame["slot"].astype(int)
    frame["date"] = target_date.normalize()
    frame["timestamp"] = frame["date"] + pd.to_timedelta(frame["slot"] * 15, unit="m")
    return frame[["date", "timestamp", "slot", "predicted"]]


def _load_actual_curve(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    actual = df[df.index.normalize() == target_date.normalize()].copy()
    if actual.empty:
        return pd.DataFrame(columns=["slot", "actual"])

    actual = actual.reset_index()
    actual["slot"] = actual["timestamp"].dt.hour * 4 + actual["timestamp"].dt.minute // 15
    return actual[["slot", "consumption"]].rename(columns={"consumption": "actual"})


def _build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    peak_row = frame.loc[frame["predicted"].idxmax()]
    valley_row = frame.loc[frame["predicted"].idxmin()]
    predicted = frame["predicted"]
    summary = {
        "predicted_daily_total": round(float(predicted.sum()), 4),
        "predicted_daily_mean": round(float(predicted.mean()), 4),
        "predicted_peak_value": round(float(peak_row["predicted"]), 4),
        "predicted_peak_slot": int(peak_row["slot"]),
        "predicted_peak_timestamp": pd.Timestamp(peak_row["timestamp"]).isoformat(),
        "predicted_valley_value": round(float(valley_row["predicted"]), 4),
        "predicted_valley_slot": int(valley_row["slot"]),
        "predicted_valley_timestamp": pd.Timestamp(valley_row["timestamp"]).isoformat(),
    }

    valid = frame.dropna(subset=["actual"])
    if not valid.empty:
        actual = valid["actual"]
        denominator = actual.replace(0, np.nan)
        mape = np.nanmean(np.abs((actual - valid["predicted"]) / denominator)) * 100
        mae = np.mean(np.abs(actual - valid["predicted"]))
        bias = np.mean(valid["predicted"] - actual)
        summary.update(
            {
                "actual_daily_total": round(float(actual.sum()), 4),
                "actual_daily_mean": round(float(actual.mean()), 4),
                "actual_mape": round(float(mape), 4),
                "actual_mae": round(float(mae), 4),
                "actual_bias": round(float(bias), 4),
            }
        )

    return summary


def _build_daily_total_context(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    predicted_daily_total: float,
    lookback_days: int,
) -> dict[str, Any]:
    daily_actuals = (
        df.assign(date=df.index.normalize())
        .groupby("date", as_index=False)
        .agg(
            actual_total=("consumption", "sum"),
            slots_observed=("consumption", "size"),
        )
        .sort_values("date")
    )

    complete_daily_actuals = daily_actuals[daily_actuals["slots_observed"] >= SLOTS_PER_DAY].copy()

    latest_complete_actual_date = complete_daily_actuals["date"].max()
    history_end_date = min(target_date - pd.Timedelta(days=1), latest_complete_actual_date)
    history_start_date = history_end_date - pd.Timedelta(days=lookback_days - 1)

    history = complete_daily_actuals[
        (complete_daily_actuals["date"] >= history_start_date) &
        (complete_daily_actuals["date"] <= history_end_date)
    ].copy()
    history["predicted_total"] = np.nan

    target_actual_row = complete_daily_actuals[complete_daily_actuals["date"] == target_date].copy()
    target_actual_total = (
        float(target_actual_row["actual_total"].iloc[0])
        if not target_actual_row.empty
        else np.nan
    )

    target_row = pd.DataFrame(
        {
            "date": [target_date.normalize()],
            "actual_total": [target_actual_total],
            "predicted_total": [predicted_daily_total],
        }
    )

    chart_frame = pd.concat([history, target_row], ignore_index=True)
    chart_frame = chart_frame.sort_values("date").reset_index(drop=True)

    bars: list[dict[str, Any]] = []
    for row in chart_frame.itertuples(index=False):
        actual_total = None if pd.isna(row.actual_total) else round(float(row.actual_total), 4)
        predicted_total_value = None if pd.isna(row.predicted_total) else round(float(row.predicted_total), 4)
        bars.append(
            {
                "date": pd.Timestamp(row.date).date().isoformat(),
                "actual_total": actual_total,
                "predicted_total": predicted_total_value,
                "is_target_date": pd.Timestamp(row.date).normalize() == target_date.normalize(),
            }
        )

    return {
        "lookback_days": lookback_days,
        "history_start_date": history_start_date.date().isoformat(),
        "history_end_date": history_end_date.date().isoformat(),
        "bars": bars,
    }


def _build_warnings(
    window_meta: dict[str, Any],
    target_date: pd.Timestamp,
    dataset: dict[str, Any],
) -> list[str]:
    warnings_list: list[str] = []
    if window_meta["train_coverage_ratio"] < 0.98:
        warnings_list.append(
            "The 28-day training window is incomplete; forecast stability may be lower than normal."
        )
    if target_date.date().isoformat() > dataset["latest_actual_date"]:
        warnings_list.append(
            "Actual values are not yet available for this target date, so only forecasted values are shown."
        )
    if pd.Timestamp("2026-03-20") <= target_date <= pd.Timestamp("2026-03-23"):
        warnings_list.append(
            "This target date falls inside the transition_drop segment, which is historically the most difficult regime."
        )
    return warnings_list


def _to_slot_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        actual_value = None if pd.isna(row.actual) else round(float(row.actual), 6)
        records.append(
            {
                "slot": int(row.slot),
                "timestamp": pd.Timestamp(row.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "predicted": round(float(row.predicted), 6),
                "actual": actual_value,
            }
        )
    return records

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import N_DELAY, SLOTS_PER_DAY, SLOTS_PER_HOUR


EARLY_LOW_VOL = "early_low_vol"
HIGH_ZIGZAG = "high_zigzag"
TRANSITION_DROP = "transition_drop"
SMOOTH_HIGH_LEVEL = "smooth_high_level"
REGIME_ORDER = [EARLY_LOW_VOL, HIGH_ZIGZAG, TRANSITION_DROP, SMOOTH_HIGH_LEVEL]
REGIME_TO_ID = {name: idx for idx, name in enumerate(REGIME_ORDER)}
LAG_DAYS_LIST = [
    (6, "lag_6d"),
    (7, "lag_7d"),
    (14, "lag_14d"),
    (28, "lag_28d"),
]
REGIME_SLOT_PREFIXES = [
    (HIGH_ZIGZAG, "zigzag_slot"),
    (SMOOTH_HIGH_LEVEL, "smooth_slot"),
    (EARLY_LOW_VOL, "low_vol_slot"),
]


def _lag_feature_names():
    return [name for _, name in LAG_DAYS_LIST]


def _feature_columns():
    lag_names = _lag_feature_names()
    cols = [
        "date",
        "slot",
        "dayofweek",
        "slot_parity",
    ]
    cols.extend(lag_names)

    lag_suffixes = [
        "is_anomaly",
        "day_mean",
        "day_std",
        "day_mad1",
        "day_lag1_ac",
        "local_mean_5",
        "local_median_5",
        "local_min_5",
        "local_max_5",
        "regime_id",
        "is_transition_drop",
        "is_smooth_high_level",
        "is_early_low_vol",
        "is_high_zigzag",
    ]
    for suffix in lag_suffixes:
        cols.extend([f"{lag_name}_{suffix}" for lag_name in lag_names])

    cols.extend([
        "same_slot_median_7d",
        "same_slot_median_28d",
        "zigzag_slot_median_7d",
        "zigzag_slot_median_28d",
        "smooth_slot_median_7d",
        "smooth_slot_median_28d",
        "low_vol_slot_median_7d",
        "low_vol_slot_median_28d",
        "daily_mean_median_7d",
        "daily_mean_median_28d",
        "daily_mean_min_7d",
        "daily_mean_min_28d",
        "daily_std_median_7d",
        "daily_std_median_28d",
        "zigzag_daily_mean_median_28d",
        "smooth_daily_mean_median_28d",
        "low_vol_daily_mean_median_28d",
        "zigzag_daily_std_median_28d",
        "smooth_daily_std_median_28d",
        "low_vol_daily_std_median_28d",
        "recent_mean_ratio_28d",
        "recent_std_ratio_28d",
        "recent_mean_delta_6_7",
        "recent_mean_delta_6_14",
        "recent_mean_delta_6_28",
        "recent_std_delta_6_14",
        "recent_local_range_6d",
        "recent_local_range_7d",
        "regime_disagreement",
        "expected_transition_risk",
        "expected_regime_confidence",
        "expected_regime_id",
        "expected_is_transition_drop",
        "expected_is_smooth_high_level",
        "expected_is_early_low_vol",
        "expected_is_high_zigzag",
        "expected_day_mean_anchor",
        "expected_day_std_anchor",
        "expected_slot_median_7d",
        "expected_slot_median_28d",
    ])
    return cols


def _level_feature_columns():
    lag_names = _lag_feature_names()
    cols = ["dayofweek"]

    for suffix in [
        "is_anomaly",
        "day_mean",
        "day_std",
        "day_mad1",
        "day_lag1_ac",
        "regime_id",
        "is_transition_drop",
        "is_smooth_high_level",
        "is_early_low_vol",
        "is_high_zigzag",
    ]:
        cols.extend([f"{lag_name}_{suffix}" for lag_name in lag_names])

    cols.extend([
        "daily_mean_median_7d",
        "daily_mean_median_28d",
        "daily_mean_min_7d",
        "daily_mean_min_28d",
        "daily_std_median_7d",
        "daily_std_median_28d",
        "zigzag_daily_mean_median_28d",
        "smooth_daily_mean_median_28d",
        "low_vol_daily_mean_median_28d",
        "zigzag_daily_std_median_28d",
        "smooth_daily_std_median_28d",
        "low_vol_daily_std_median_28d",
        "recent_mean_ratio_28d",
        "recent_std_ratio_28d",
        "recent_mean_delta_6_7",
        "recent_mean_delta_6_14",
        "recent_mean_delta_6_28",
        "recent_std_delta_6_14",
        "regime_disagreement",
        "expected_transition_risk",
        "expected_regime_confidence",
        "expected_regime_id",
        "expected_is_transition_drop",
        "expected_is_smooth_high_level",
        "expected_is_early_low_vol",
        "expected_is_high_zigzag",
        "expected_day_mean_anchor",
        "expected_day_std_anchor",
    ])
    return cols


def _shape_feature_columns():
    lag_names = _lag_feature_names()
    cols = [
        "slot",
        "slot_parity",
        "dayofweek",
    ]
    cols.extend(lag_names)

    for suffix in [
        "is_anomaly",
        "day_mean",
        "day_std",
        "day_mad1",
        "day_lag1_ac",
        "local_mean_5",
        "local_median_5",
        "local_min_5",
        "local_max_5",
        "regime_id",
        "is_transition_drop",
        "is_smooth_high_level",
        "is_early_low_vol",
        "is_high_zigzag",
    ]:
        cols.extend([f"{lag_name}_{suffix}" for lag_name in lag_names])

    cols.extend([
        "same_slot_median_7d",
        "same_slot_median_28d",
        "zigzag_slot_median_7d",
        "zigzag_slot_median_28d",
        "smooth_slot_median_7d",
        "smooth_slot_median_28d",
        "low_vol_slot_median_7d",
        "low_vol_slot_median_28d",
        "daily_mean_median_7d",
        "daily_mean_median_28d",
        "daily_std_median_7d",
        "daily_std_median_28d",
        "zigzag_daily_mean_median_28d",
        "smooth_daily_mean_median_28d",
        "low_vol_daily_mean_median_28d",
        "zigzag_daily_std_median_28d",
        "smooth_daily_std_median_28d",
        "low_vol_daily_std_median_28d",
        "recent_mean_ratio_28d",
        "recent_std_ratio_28d",
        "recent_mean_delta_6_7",
        "recent_mean_delta_6_14",
        "recent_mean_delta_6_28",
        "recent_std_delta_6_14",
        "recent_local_range_6d",
        "recent_local_range_7d",
        "regime_disagreement",
        "expected_transition_risk",
        "expected_regime_confidence",
        "expected_regime_id",
        "expected_is_transition_drop",
        "expected_is_smooth_high_level",
        "expected_is_early_low_vol",
        "expected_is_high_zigzag",
        "expected_day_mean_anchor",
        "expected_day_std_anchor",
        "expected_slot_median_7d",
        "expected_slot_median_28d",
        "lag_6d_shape_ratio",
        "lag_7d_shape_ratio",
        "lag_14d_shape_ratio",
        "lag_28d_shape_ratio",
        "same_slot_shape_ratio_7d",
        "same_slot_shape_ratio_28d",
        "expected_slot_shape_ratio_7d",
        "expected_slot_shape_ratio_28d",
    ])
    return cols


def fill_missing(df, slot_means=None, mode="train"):
    df = df.copy()
    df["slot"] = df.index.hour * SLOTS_PER_HOUR + df.index.minute // (60 // SLOTS_PER_HOUR)

    if slot_means is None:
        slot_means = df.groupby("slot")["consumption"].mean().to_dict()

    missing_mask = df["consumption"].isna()
    if missing_mask.any():
        for idx in df.index[missing_mask]:
            slot = df.at[idx, "slot"]
            week_history = df.loc[
                (df.index >= idx - pd.Timedelta(days=7)) &
                (df.index < idx) &
                (df["slot"] == slot),
                "consumption"
            ].dropna()

            if not week_history.empty:
                df.at[idx, "consumption"] = week_history.mean()
            else:
                df.at[idx, "consumption"] = slot_means[slot]

    return df, slot_means


def _build_daily_profile(df):
    rows = []

    for date, g in df.groupby("date"):
        values = g.sort_values("slot")["consumption"].reset_index(drop=True)
        mad1 = values.diff().abs().mean() if len(values) >= 2 else np.nan
        lag1_ac = values.autocorr(lag=1) if len(values) >= 3 else np.nan
        rows.append({
            "date": pd.Timestamp(date).normalize(),
            "daily_mean": values.mean(),
            "daily_std": values.std(),
            "rows": len(values),
            "daily_mad1": mad1,
            "daily_lag1_ac": lag1_ac,
        })

    profile = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return profile


def _detect_anomaly_days(profile):
    if profile.empty:
        return profile.assign(is_anomaly_day=pd.Series(dtype=bool))

    profile = profile.copy().sort_values("date").reset_index(drop=True)
    trailing_mean_median = profile["daily_mean"].shift(1).rolling(7, min_periods=5).median()
    low_mean_day = profile["daily_mean"] < (trailing_mean_median * 0.90)
    incomplete_day = (profile["rows"] < SLOTS_PER_DAY) & trailing_mean_median.notna()

    is_transition_drop = (
        low_mean_day.fillna(False)
        | incomplete_day.fillna(False)
        | (
            (profile["daily_mean"] < 25.0)
            & (profile["daily_std"] < 8.0)
            & trailing_mean_median.notna()
        )
    )
    is_smooth_high_level = (
        ~is_transition_drop
        & (profile["daily_std"] < 5.0)
        & (profile["daily_mad1"] < 5.0)
        & (profile["daily_mean"] >= 30.0)
    )
    is_early_low_vol = (
        ~is_transition_drop
        & ~is_smooth_high_level
        & (profile["daily_std"] < 5.5)
        & (profile["daily_mad1"] < 8.0)
        & (profile["daily_mean"] < 28.5)
    )
    is_high_zigzag = ~(is_transition_drop | is_smooth_high_level | is_early_low_vol)

    regime_name = np.where(
        is_transition_drop,
        TRANSITION_DROP,
        np.where(
            is_smooth_high_level,
            SMOOTH_HIGH_LEVEL,
            np.where(is_early_low_vol, EARLY_LOW_VOL, HIGH_ZIGZAG),
        ),
    )

    profile["trailing_mean_median_7d"] = trailing_mean_median
    profile["is_transition_drop"] = is_transition_drop
    profile["is_smooth_high_level"] = is_smooth_high_level
    profile["is_early_low_vol"] = is_early_low_vol
    profile["is_high_zigzag"] = is_high_zigzag
    profile["regime_name"] = regime_name
    profile["regime_id"] = pd.Series(regime_name).map(REGIME_TO_ID).astype(int)
    profile["is_anomaly_day"] = is_transition_drop
    return profile


def _prepare_daily_history(df, daily_profile):
    history = df[["date", "slot", "consumption"]].copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history = history.groupby(["date", "slot"], as_index=False)["consumption"].mean()
    history = history.merge(
        daily_profile[
            [
                "date",
                "is_anomaly_day",
                "daily_mean",
                "daily_std",
                "daily_mad1",
                "daily_lag1_ac",
                "regime_name",
                "regime_id",
                "is_transition_drop",
                "is_smooth_high_level",
                "is_early_low_vol",
                "is_high_zigzag",
            ]
        ],
        on="date",
        how="left",
    )
    return _add_local_slot_context(history)


def _add_local_slot_context(history):
    context = history[["date", "slot"]].copy()
    window_offsets = [-2, -1, 0, 1, 2]
    shift_cols = []

    for offset in window_offsets:
        shift_name = f"shift_{offset:+d}".replace("+", "p").replace("-", "m")
        shifted = history[["date", "slot", "consumption"]].copy()
        shifted["slot"] = shifted["slot"] - offset
        shifted = shifted.rename(columns={"consumption": shift_name})
        context = context.merge(shifted, on=["date", "slot"], how="left")
        shift_cols.append(shift_name)

    context["local_mean_5"] = context[shift_cols].mean(axis=1, skipna=True)
    context["local_median_5"] = context[shift_cols].median(axis=1, skipna=True)
    context["local_min_5"] = context[shift_cols].min(axis=1, skipna=True)
    context["local_max_5"] = context[shift_cols].max(axis=1, skipna=True)

    return history.merge(
        context[["date", "slot", "local_mean_5", "local_median_5", "local_min_5", "local_max_5"]],
        on=["date", "slot"],
        how="left",
    )


def _build_same_slot_stats(target_dates, history, regime_name=None, prefix="same_slot"):
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize().unique().sort_values()
    clean_history = history[~history["is_anomaly_day"].eq(True)].copy()

    if regime_name is not None:
        clean_history = clean_history[clean_history["regime_name"] == regime_name].copy()

    slot_histories = {
        slot: g.sort_values("date")
        for slot, g in clean_history.groupby("slot")
    }
    records = []

    for date in target_dates:
        for slot in range(SLOTS_PER_DAY):
            slot_history = slot_histories.get(slot)

            if slot_history is None or slot_history.empty:
                median_7d = np.nan
                median_28d = np.nan
            else:
                past = slot_history[slot_history["date"] < date]
                window_7d = past[past["date"] >= date - pd.Timedelta(days=7)]["consumption"]
                window_28d = past[past["date"] >= date - pd.Timedelta(days=28)]["consumption"]
                median_7d = window_7d.median() if len(window_7d) > 0 else np.nan
                median_28d = window_28d.median() if len(window_28d) > 0 else np.nan

            records.append({
                "date": date,
                "slot": slot,
                f"{prefix}_median_7d": median_7d,
                f"{prefix}_median_28d": median_28d,
            })

    return pd.DataFrame(records)


def _build_daily_level_stats(target_dates, daily_profile):
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize().unique().sort_values()
    clean_profile = daily_profile[~daily_profile["is_anomaly_day"].eq(True)].copy()
    records = []

    for date in target_dates:
        past = clean_profile[clean_profile["date"] < date]
        window_7d = past[past["date"] >= date - pd.Timedelta(days=7)]
        window_28d = past[past["date"] >= date - pd.Timedelta(days=28)]
        zigzag_28d = window_28d[window_28d["regime_name"] == HIGH_ZIGZAG]
        smooth_28d = window_28d[window_28d["regime_name"] == SMOOTH_HIGH_LEVEL]
        low_28d = window_28d[window_28d["regime_name"] == EARLY_LOW_VOL]

        records.append({
            "date": date,
            "daily_mean_median_7d": window_7d["daily_mean"].median() if len(window_7d) > 0 else np.nan,
            "daily_mean_median_28d": window_28d["daily_mean"].median() if len(window_28d) > 0 else np.nan,
            "daily_mean_min_7d": window_7d["daily_mean"].min() if len(window_7d) > 0 else np.nan,
            "daily_mean_min_28d": window_28d["daily_mean"].min() if len(window_28d) > 0 else np.nan,
            "daily_std_median_7d": window_7d["daily_std"].median() if len(window_7d) > 0 else np.nan,
            "daily_std_median_28d": window_28d["daily_std"].median() if len(window_28d) > 0 else np.nan,
            "zigzag_daily_mean_median_28d": zigzag_28d["daily_mean"].median() if len(zigzag_28d) > 0 else np.nan,
            "smooth_daily_mean_median_28d": smooth_28d["daily_mean"].median() if len(smooth_28d) > 0 else np.nan,
            "low_vol_daily_mean_median_28d": low_28d["daily_mean"].median() if len(low_28d) > 0 else np.nan,
            "zigzag_daily_std_median_28d": zigzag_28d["daily_std"].median() if len(zigzag_28d) > 0 else np.nan,
            "smooth_daily_std_median_28d": smooth_28d["daily_std"].median() if len(smooth_28d) > 0 else np.nan,
            "low_vol_daily_std_median_28d": low_28d["daily_std"].median() if len(low_28d) > 0 else np.nan,
        })

    return pd.DataFrame(records)


def _add_lag_features(base, history, lag_days_list):
    for lag_days, col_name in lag_days_list:
        lagged = history[
            [
                "date",
                "slot",
                "consumption",
                "is_anomaly_day",
                "daily_mean",
                "daily_std",
                "daily_mad1",
                "daily_lag1_ac",
                "local_mean_5",
                "local_median_5",
                "local_min_5",
                "local_max_5",
                "regime_id",
                "is_transition_drop",
                "is_smooth_high_level",
                "is_early_low_vol",
                "is_high_zigzag",
            ]
        ].copy()
        lagged["date"] = pd.to_datetime(lagged["date"]) + pd.Timedelta(days=lag_days)
        lagged = lagged.rename(columns={
            "consumption": f"{col_name}_raw",
            "is_anomaly_day": f"{col_name}_is_anomaly",
            "daily_mean": f"{col_name}_day_mean",
            "daily_std": f"{col_name}_day_std",
            "daily_mad1": f"{col_name}_day_mad1",
            "daily_lag1_ac": f"{col_name}_day_lag1_ac",
            "local_mean_5": f"{col_name}_local_mean_5",
            "local_median_5": f"{col_name}_local_median_5",
            "local_min_5": f"{col_name}_local_min_5",
            "local_max_5": f"{col_name}_local_max_5",
            "regime_id": f"{col_name}_regime_id",
            "is_transition_drop": f"{col_name}_is_transition_drop",
            "is_smooth_high_level": f"{col_name}_is_smooth_high_level",
            "is_early_low_vol": f"{col_name}_is_early_low_vol",
            "is_high_zigzag": f"{col_name}_is_high_zigzag",
        })
        base = base.merge(lagged, on=["date", "slot"], how="left")

    return base


def _apply_robust_lag_fallbacks(base, lag_feature_names):
    for lag_name in lag_feature_names:
        clean_candidates = []

        for other_lag_name in lag_feature_names:
            other_is_anomaly = base[f"{other_lag_name}_is_anomaly"].eq(True)
            clean_col = base[f"{other_lag_name}_raw"].where(
                ~other_is_anomaly
            )
            clean_candidates.append(clean_col.rename(other_lag_name))

        candidate_frame = pd.concat(clean_candidates, axis=1)
        fallback = candidate_frame.median(axis=1, skipna=True)
        lag_is_anomaly = base[f"{lag_name}_is_anomaly"].eq(True)
        primary_clean = base[f"{lag_name}_raw"].where(
            ~lag_is_anomaly
        )

        base[lag_name] = primary_clean.fillna(fallback).fillna(base[f"{lag_name}_raw"])
        base[f"{lag_name}_is_anomaly"] = lag_is_anomaly.astype(int)

    return base


def _coalesce_columns(base, columns):
    available = [col for col in columns if col in base.columns]
    if not available:
        return pd.Series(np.nan, index=base.index)
    return base[available].bfill(axis=1).iloc[:, 0]


def _merge_history_statistics(base, history, daily_profile):
    target_dates = base["date"].unique()

    base = base.merge(
        _build_same_slot_stats(target_dates, history),
        on=["date", "slot"],
        how="left",
    )

    for regime_name, prefix in REGIME_SLOT_PREFIXES:
        regime_stats = _build_same_slot_stats(
            target_dates,
            history,
            regime_name=regime_name,
            prefix=prefix,
        )
        base = base.merge(regime_stats, on=["date", "slot"], how="left")

    base = base.merge(
        _build_daily_level_stats(target_dates, daily_profile),
        on="date",
        how="left",
    )
    return base


def _add_regime_expectation_features(base):
    base = base.copy()

    base["recent_mean_ratio_28d"] = base["lag_6d_day_mean"] / base["daily_mean_median_28d"].replace(0, np.nan)
    base["recent_std_ratio_28d"] = base["lag_6d_day_std"] / base["daily_std_median_28d"].replace(0, np.nan)
    base["recent_mean_delta_6_7"] = base["lag_6d_day_mean"] - base["lag_7d_day_mean"]
    base["recent_mean_delta_6_14"] = base["lag_6d_day_mean"] - base["lag_14d_day_mean"]
    base["recent_mean_delta_6_28"] = base["lag_6d_day_mean"] - base["lag_28d_day_mean"]
    base["recent_std_delta_6_14"] = base["lag_6d_day_std"] - base["lag_14d_day_std"]
    base["recent_local_range_6d"] = base["lag_6d_local_max_5"] - base["lag_6d_local_min_5"]
    base["recent_local_range_7d"] = base["lag_7d_local_max_5"] - base["lag_7d_local_min_5"]

    lag_regime_cols = [f"{lag_name}_regime_id" for lag_name in _lag_feature_names()]
    lag_regimes = base[lag_regime_cols].copy()

    def _disagreement(row):
        values = row.dropna().astype(int)
        if len(values) <= 1:
            return 0.0
        return 1.0 - values.value_counts(normalize=True).iloc[0]

    base["regime_disagreement"] = lag_regimes.apply(_disagreement, axis=1)

    mean_ratio = base["recent_mean_ratio_28d"].fillna(1.0)
    std_ratio = base["recent_std_ratio_28d"].fillna(1.0)
    mean_delta_6_14 = base["recent_mean_delta_6_14"].fillna(0.0)
    mean_delta_6_28 = base["recent_mean_delta_6_28"].fillna(0.0)
    std_delta_6_14 = base["recent_std_delta_6_14"].fillna(0.0)

    transition_score = (
        0.70 * base["lag_6d_is_transition_drop"].fillna(0.0)
        + 0.45 * base["lag_7d_is_transition_drop"].fillna(0.0)
        + 0.20 * base["lag_14d_is_transition_drop"].fillna(0.0)
        + 0.35 * ((mean_ratio < 0.93) & (std_ratio < 0.80)).astype(float)
        + 0.25 * (mean_delta_6_14 < -2.0).astype(float)
        + 0.20 * (mean_delta_6_28 < -3.0).astype(float)
        + 0.15 * (std_delta_6_14 < -2.0).astype(float)
        + 0.20 * (base["regime_disagreement"] > 0.34).astype(float)
    )
    base["expected_transition_risk"] = np.clip(transition_score / 1.8, 0.0, 1.0)
    transition_dampen = 1.0 - base["expected_transition_risk"]

    smooth_score = (
        0.55 * base["lag_6d_is_smooth_high_level"].fillna(0.0)
        + 0.25 * base["lag_7d_is_smooth_high_level"].fillna(0.0)
        + 0.15 * base["lag_14d_is_smooth_high_level"].fillna(0.0)
        + 0.05 * base["lag_28d_is_smooth_high_level"].fillna(0.0)
        + 0.35 * transition_dampen * (
            (base["lag_6d_day_std"].fillna(np.inf) < 5.0)
            & (base["lag_6d_day_mean"].fillna(-np.inf) >= 30.0)
        ).astype(float)
        + 0.20 * transition_dampen * ((std_ratio < 0.70) & (mean_ratio > 0.95)).astype(float)
        - 0.40 * base["expected_transition_risk"]
    )
    early_score = (
        0.55 * base["lag_6d_is_early_low_vol"].fillna(0.0)
        + 0.25 * base["lag_7d_is_early_low_vol"].fillna(0.0)
        + 0.15 * base["lag_14d_is_early_low_vol"].fillna(0.0)
        + 0.05 * base["lag_28d_is_early_low_vol"].fillna(0.0)
        + 0.35 * transition_dampen * (
            (base["lag_6d_day_std"].fillna(np.inf) < 5.5)
            & (base["lag_6d_day_mean"].fillna(np.inf) < 28.5)
        ).astype(float)
        + 0.20 * transition_dampen * ((std_ratio < 0.80) & (mean_ratio < 1.02)).astype(float)
        - 0.30 * base["expected_transition_risk"]
    )
    zigzag_score = (
        0.55 * base["lag_6d_is_high_zigzag"].fillna(0.0)
        + 0.25 * base["lag_7d_is_high_zigzag"].fillna(0.0)
        + 0.15 * base["lag_14d_is_high_zigzag"].fillna(0.0)
        + 0.05 * base["lag_28d_is_high_zigzag"].fillna(0.0)
        + 0.25 * (
            (base["lag_6d_day_std"].fillna(-np.inf) >= 7.0)
            | (base["recent_local_range_6d"].fillna(-np.inf) > 10.0)
        ).astype(float)
        + 0.15 * (std_ratio > 0.90).astype(float)
        + 0.20 * base["expected_transition_risk"]
    )

    stable_score_frame = pd.DataFrame(
        {
            EARLY_LOW_VOL: early_score.clip(lower=0.0),
            HIGH_ZIGZAG: zigzag_score.clip(lower=0.0),
            SMOOTH_HIGH_LEVEL: smooth_score.clip(lower=0.0),
        },
        index=base.index,
    )
    expected_regime_name = stable_score_frame.idxmax(axis=1)

    base["expected_regime_confidence"] = stable_score_frame.max(axis=1)
    base["expected_regime_id"] = expected_regime_name.map(REGIME_TO_ID).astype(int)
    base["expected_is_transition_drop"] = base["expected_transition_risk"].ge(0.85).astype(int)
    base["expected_is_smooth_high_level"] = expected_regime_name.eq(SMOOTH_HIGH_LEVEL).astype(int)
    base["expected_is_early_low_vol"] = expected_regime_name.eq(EARLY_LOW_VOL).astype(int)
    base["expected_is_high_zigzag"] = expected_regime_name.eq(HIGH_ZIGZAG).astype(int)

    default_day_mean_anchor = _coalesce_columns(
        base,
        [
            "lag_6d_day_mean",
            "daily_mean_median_7d",
            "daily_mean_median_28d",
        ],
    )
    default_day_std_anchor = _coalesce_columns(
        base,
        [
            "lag_6d_day_std",
            "daily_std_median_7d",
            "daily_std_median_28d",
        ],
    )

    base["expected_day_mean_anchor"] = np.select(
        [
            base["expected_is_smooth_high_level"].eq(1),
            base["expected_is_early_low_vol"].eq(1),
            base["expected_is_high_zigzag"].eq(1),
        ],
        [
            base["smooth_daily_mean_median_28d"],
            base["low_vol_daily_mean_median_28d"],
            base["zigzag_daily_mean_median_28d"],
        ],
        default=default_day_mean_anchor,
    )
    base["expected_day_mean_anchor"] = (
        pd.Series(base["expected_day_mean_anchor"], index=base.index)
        .fillna(default_day_mean_anchor)
    )

    base["expected_day_std_anchor"] = np.select(
        [
            base["expected_is_smooth_high_level"].eq(1),
            base["expected_is_early_low_vol"].eq(1),
            base["expected_is_high_zigzag"].eq(1),
        ],
        [
            base["smooth_daily_std_median_28d"],
            base["low_vol_daily_std_median_28d"],
            base["zigzag_daily_std_median_28d"],
        ],
        default=default_day_std_anchor,
    )
    base["expected_day_std_anchor"] = (
        pd.Series(base["expected_day_std_anchor"], index=base.index)
        .fillna(default_day_std_anchor)
    )

    base["expected_slot_median_7d"] = np.select(
        [
            base["expected_is_smooth_high_level"].eq(1),
            base["expected_is_early_low_vol"].eq(1),
            base["expected_is_high_zigzag"].eq(1),
        ],
        [
            base["smooth_slot_median_7d"],
            base["low_vol_slot_median_7d"],
            base["zigzag_slot_median_7d"],
        ],
        default=base["same_slot_median_7d"],
    )
    base["expected_slot_median_7d"] = (
        pd.Series(base["expected_slot_median_7d"], index=base.index)
        .fillna(base["same_slot_median_7d"])
    )

    base["expected_slot_median_28d"] = np.select(
        [
            base["expected_is_smooth_high_level"].eq(1),
            base["expected_is_early_low_vol"].eq(1),
            base["expected_is_high_zigzag"].eq(1),
        ],
        [
            base["smooth_slot_median_28d"],
            base["low_vol_slot_median_28d"],
            base["zigzag_slot_median_28d"],
        ],
        default=base["same_slot_median_28d"],
    )
    base["expected_slot_median_28d"] = (
        pd.Series(base["expected_slot_median_28d"], index=base.index)
        .fillna(base["expected_slot_median_7d"])
        .fillna(base["same_slot_median_28d"])
    )

    return base


def _build_feature_frame(base, history, daily_profile):
    base = _add_lag_features(base, history, LAG_DAYS_LIST)
    base = _apply_robust_lag_fallbacks(base, _lag_feature_names())
    base = _merge_history_statistics(base, history, daily_profile)

    for lag_name in _lag_feature_names():
        base[f"{lag_name}_regime_id"] = base[f"{lag_name}_regime_id"].fillna(-1.0)
        for suffix in [
            "is_transition_drop",
            "is_smooth_high_level",
            "is_early_low_vol",
            "is_high_zigzag",
        ]:
            col = f"{lag_name}_{suffix}"
            base[col] = pd.to_numeric(
                base[col].replace({True: 1, False: 0}),
                errors="coerce",
            ).fillna(0).astype(int)

    base = _add_regime_expectation_features(base)
    return base


def build_train_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df.index.normalize())
    df["slot"] = df.index.hour * SLOTS_PER_HOUR + df.index.minute // (60 // SLOTS_PER_HOUR)
    df["dayofweek"] = df.index.dayofweek
    df["slot_parity"] = df["slot"] % 2

    daily_profile = _detect_anomaly_days(_build_daily_profile(df))
    history = _prepare_daily_history(df, daily_profile)

    base = df[["date", "slot", "dayofweek", "slot_parity", "consumption"]].copy()
    base = _build_feature_frame(base, history, daily_profile)
    feature_cols = _feature_columns()
    X = base[feature_cols]
    y = base["consumption"]

    return X, y, base


def build_predict_features(df, target_date=None):
    df = df.copy()
    df["date"] = pd.to_datetime(df.index.normalize())
    df["slot"] = df.index.hour * SLOTS_PER_HOUR + df.index.minute // (60 // SLOTS_PER_HOUR)
    df["slot_parity"] = df["slot"] % 2

    last_date = df["date"].max()
    if target_date is None:
        target_date = last_date + pd.Timedelta(days=N_DELAY)
    else:
        target_date = pd.Timestamp(target_date).normalize()

    daily_profile = _detect_anomaly_days(_build_daily_profile(df))
    history = _prepare_daily_history(df, daily_profile)

    base = pd.DataFrame({
        "date": [target_date] * SLOTS_PER_DAY,
        "slot": range(SLOTS_PER_DAY),
        "dayofweek": [target_date.dayofweek] * SLOTS_PER_DAY,
        "slot_parity": [slot % 2 for slot in range(SLOTS_PER_DAY)],
    })

    base = _build_feature_frame(base, history, daily_profile)
    feature_cols = _feature_columns()
    X = base[feature_cols]

    return X, base, target_date


def get_model():
    return TwoStageLoadModel()


class TwoStageLoadModel:
    def __init__(self):
        self.level_model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        )
        self.shape_model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        )
        self.level_feature_cols = _level_feature_columns()
        self.shape_feature_cols = _shape_feature_columns()

    def fit(self, X, y):
        frame = X.copy()
        frame["target"] = y.values
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

        level_frame = self._build_level_frame(frame)
        self.level_model.fit(level_frame[self.level_feature_cols], level_frame["actual_day_mean"])

        shape_frame = self._build_shape_frame(frame)
        self.shape_model.fit(shape_frame[self.shape_feature_cols], shape_frame["shape_ratio"])

        return self

    def predict(self, X):
        frame = X.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

        level_frame = self._build_level_frame(frame)
        level_predictions = level_frame[["date"]].copy()
        level_predictions["pred_day_mean_model"] = self.level_model.predict(level_frame[self.level_feature_cols])

        lag_day_mean_cols = [
            "lag_6d_day_mean",
            "lag_7d_day_mean",
            "lag_14d_day_mean",
            "lag_28d_day_mean",
        ]
        lag_day_means = level_frame[lag_day_mean_cols]
        conservative_mean = lag_day_means.min(axis=1, skipna=True)
        mean_spread = lag_day_means.std(axis=1, skipna=True).fillna(0.0)
        model_anchor = level_frame["expected_day_mean_anchor"].fillna(conservative_mean)
        model_anchor = model_anchor.fillna(level_predictions["pred_day_mean_model"])
        anchor_weight = (
            0.04
            + 0.08 * level_frame["expected_is_smooth_high_level"]
            + 0.05 * level_frame["expected_is_early_low_vol"]
            + 0.04 * level_frame["regime_disagreement"]
            + 0.04 * level_frame["expected_transition_risk"]
            - 0.02 * level_frame["expected_is_high_zigzag"]
        ).clip(0.02, 0.20)
        level_predictions["pred_day_mean"] = (
            (1.0 - anchor_weight) * level_predictions["pred_day_mean_model"]
            + anchor_weight * model_anchor
        )
        conservative_weight = (
            ((mean_spread - 1.0) / 8.0).clip(0.0, 0.35)
            + 0.04 * level_frame["expected_transition_risk"]
        ).clip(0.0, 0.40)
        conservative_anchor = conservative_mean.fillna(level_predictions["pred_day_mean"])
        level_predictions["pred_day_mean"] = (
            (1.0 - conservative_weight) * level_predictions["pred_day_mean"]
            + conservative_weight * conservative_anchor
        )

        frame = frame.merge(level_predictions[["date", "pred_day_mean"]], on="date", how="left")
        shape_frame = self._build_shape_frame(frame)
        shape_frame["shape_ratio_pred"] = self.shape_model.predict(shape_frame[self.shape_feature_cols])
        shape_frame["shape_ratio_pred"] = shape_frame["shape_ratio_pred"].clip(lower=0.35, upper=1.75)

        shape_anchor = _coalesce_columns(
            shape_frame,
            [
                "expected_slot_shape_ratio_28d",
                "expected_slot_shape_ratio_7d",
                "same_slot_shape_ratio_28d",
                "same_slot_shape_ratio_7d",
            ],
        ).fillna(1.0)
        shape_anchor_weight = (
            0.06
            + 0.08 * shape_frame["expected_is_smooth_high_level"]
            + 0.04 * shape_frame["expected_is_early_low_vol"]
            + 0.04 * shape_frame["expected_transition_risk"]
            + 0.04 * shape_frame["regime_disagreement"]
            - 0.02 * shape_frame["expected_is_high_zigzag"]
        ).clip(0.03, 0.22)
        shape_frame["shape_ratio_pred"] = (
            (1.0 - shape_anchor_weight) * shape_frame["shape_ratio_pred"]
            + shape_anchor_weight * shape_anchor
        )

        daily_shape_mean = shape_frame.groupby("date")["shape_ratio_pred"].transform("mean")
        normalized_shape = shape_frame["shape_ratio_pred"] / daily_shape_mean

        volatility_proxy = shape_frame[[
            "lag_6d_day_std",
            "lag_7d_day_std",
            "lag_14d_day_std",
            "lag_28d_day_std",
        ]].median(axis=1, skipna=True)
        volatility_proxy = volatility_proxy.fillna(shape_frame["expected_day_std_anchor"]).fillna(6.0)
        shrink = 0.45 + (volatility_proxy / 18.0)
        shrink = (
            shrink
            - 0.10 * shape_frame["expected_is_smooth_high_level"]
            - 0.05 * shape_frame["expected_is_early_low_vol"]
            - 0.05 * shape_frame["expected_transition_risk"]
            + 0.05 * shape_frame["expected_is_high_zigzag"]
        ).clip(0.30, 0.98)
        adjusted_shape = 1.0 + shrink * (normalized_shape - 1.0)

        return shape_frame["pred_day_mean"] * adjusted_shape

    def _build_level_frame(self, frame):
        grouped = frame.groupby("date", as_index=False).first()
        grouped["actual_day_mean"] = frame.groupby("date")["target"].mean().values if "target" in frame.columns else np.nan
        return grouped

    def _build_shape_frame(self, frame):
        shape = frame.copy()
        day_mean_col = "actual_day_mean" if "actual_day_mean" in shape.columns else "pred_day_mean"

        if "actual_day_mean" not in shape.columns and "target" in shape.columns:
            shape = shape.merge(
                self._build_level_frame(shape)[["date", "actual_day_mean"]],
                on="date",
                how="left",
            )
            day_mean_col = "actual_day_mean"

        for lag_name in ["lag_6d", "lag_7d", "lag_14d", "lag_28d"]:
            shape[f"{lag_name}_shape_ratio"] = shape[lag_name] / shape[f"{lag_name}_day_mean"].replace(0, np.nan)

        shape["same_slot_shape_ratio_7d"] = shape["same_slot_median_7d"] / shape["daily_mean_median_7d"].replace(0, np.nan)
        shape["same_slot_shape_ratio_28d"] = shape["same_slot_median_28d"] / shape["daily_mean_median_28d"].replace(0, np.nan)
        shape["expected_slot_shape_ratio_7d"] = shape["expected_slot_median_7d"] / shape["expected_day_mean_anchor"].replace(0, np.nan)
        shape["expected_slot_shape_ratio_28d"] = shape["expected_slot_median_28d"] / shape["expected_day_mean_anchor"].replace(0, np.nan)

        if "target" in shape.columns:
            shape["shape_ratio"] = shape["target"] / shape[day_mean_col].replace(0, np.nan)

        return shape

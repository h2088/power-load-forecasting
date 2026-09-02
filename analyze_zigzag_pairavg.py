import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import fill_missing
from src.config import N_DELAY, SLOTS_PER_DAY, SLOTS_PER_HOUR, WINDOW_DAYS
from src.pipeline import predict, train


warnings.filterwarnings("ignore")

SEGMENT_START = pd.Timestamp("2026-01-09")
SEGMENT_END = pd.Timestamp("2026-03-13")
BACKTEST_START = pd.Timestamp("2026-02-01")
BACKTEST_END = pd.Timestamp("2026-03-13")
INPUT_PATH = "data/total_consumption.csv"

DAY_SUMMARY_PATH = "zigzag_pairavg_two_stage_day_summary.csv"
SLOT_PREDICTIONS_PATH = "zigzag_pairavg_two_stage_slot_predictions.csv"
OVERALL_SUMMARY_PATH = "zigzag_pairavg_two_stage_overall_summary.csv"
DAILY_MAPE_PLOT_PATH = "zigzag_pairavg_two_stage_daily_mape.png"
EXAMPLE_PLOT_PATH = "zigzag_pairavg_two_stage_examples.png"


def _add_time_columns(df):
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame.index.normalize())
    frame["slot"] = frame.index.hour * SLOTS_PER_HOUR + frame.index.minute // (60 // SLOTS_PER_HOUR)
    return frame


def _date_mask(df, start_date, end_date):
    dates = pd.to_datetime(df.index.normalize())
    return (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))


def _complete_day_grid(df, slot_fill=None):
    if df.empty:
        return df.copy()

    frame = _add_time_columns(df)
    if slot_fill is None:
        slot_fill = frame.groupby("slot")["consumption"].mean().to_dict()
    fallback_value = frame["consumption"].mean()

    rows = []
    for date, group in frame.groupby("date"):
        full = pd.DataFrame({"slot": range(SLOTS_PER_DAY)})
        full["date"] = pd.Timestamp(date)
        full = full.merge(group[["slot", "consumption"]], on="slot", how="left")
        full["consumption"] = (
            full["consumption"]
            .fillna(full["slot"].map(slot_fill))
            .fillna(fallback_value)
        )
        full["timestamp"] = full["date"] + pd.to_timedelta(full["slot"] * 15, unit="min")
        rows.append(full[["timestamp", "consumption"]])

    completed = pd.concat(rows, ignore_index=True).set_index("timestamp").sort_index()
    return completed


def _pair_average_smooth(df, slot_fill=None):
    completed = _complete_day_grid(df, slot_fill=slot_fill)
    frame = _add_time_columns(completed)
    rows = []

    for _, group in frame.groupby("date"):
        ordered = group.sort_values("slot").copy()
        values = ordered["consumption"].to_numpy(dtype=float)
        pair_mean = values.reshape(SLOTS_PER_DAY // 2, 2).mean(axis=1)
        ordered["consumption"] = np.repeat(pair_mean, 2)
        rows.append(ordered[["consumption"]])

    return pd.concat(rows).sort_index()


def _build_residual_template(raw_df, smooth_df):
    raw_frame = _add_time_columns(_complete_day_grid(raw_df))
    smooth_frame = _add_time_columns(_complete_day_grid(smooth_df))
    smooth_frame = smooth_frame.rename(columns={"consumption": "smooth_consumption"})

    merged = raw_frame[["date", "slot", "consumption"]].merge(
        smooth_frame[["date", "slot", "smooth_consumption"]],
        on=["date", "slot"],
        how="inner",
    )
    merged["residual"] = merged["consumption"] - merged["smooth_consumption"]
    return merged.groupby("slot")["residual"].median()


def _compute_ape(actual, predicted):
    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, dtype=float)
    valid = actual.notna() & predicted.notna() & actual.ne(0)
    ape = pd.Series(np.nan, index=actual.index, dtype=float)
    ape.loc[valid] = (actual.loc[valid] - predicted.loc[valid]).abs() / actual.loc[valid] * 100.0
    return ape


def _train_window(raw_df, target_date):
    cutoff = target_date - pd.Timedelta(days=N_DELAY)
    window_start = max(cutoff - pd.Timedelta(days=WINDOW_DAYS - 1), SEGMENT_START)
    mask = _date_mask(raw_df, window_start, cutoff)
    return raw_df.loc[mask].copy()


def run_pairavg_two_stage_backtest(raw_df):
    day_records = []
    slot_records = []

    for target_date in pd.date_range(BACKTEST_START, BACKTEST_END, freq="D"):
        train_raw = _train_window(raw_df, target_date)
        if train_raw.empty:
            continue

        baseline_model, baseline_slot_means = train(train_raw)
        baseline_pred, _ = predict(train_raw, baseline_model, baseline_slot_means, target_date=target_date)
        baseline_pred = baseline_pred[["slot", "predicted"]].rename(columns={"predicted": "baseline_pred"})

        train_raw_filled, raw_slot_means = fill_missing(train_raw, mode="train")
        smooth_train = _pair_average_smooth(train_raw_filled[["consumption"]], slot_fill=raw_slot_means)

        smooth_model, smooth_slot_means = train(smooth_train)
        smooth_pred, _ = predict(smooth_train, smooth_model, smooth_slot_means, target_date=target_date)
        smooth_pred = smooth_pred[["slot", "predicted"]].rename(columns={"predicted": "smooth_pred"})

        residual_template = _build_residual_template(train_raw_filled[["consumption"]], smooth_train[["consumption"]])
        smooth_pred["template_residual"] = smooth_pred["slot"].map(residual_template)
        smooth_pred["pairavg_pred"] = smooth_pred["smooth_pred"] + smooth_pred["template_residual"]

        actual_raw = raw_df.loc[_date_mask(raw_df, target_date, target_date)].copy()
        actual_raw_frame = _add_time_columns(actual_raw)
        actual_smooth = _pair_average_smooth(actual_raw[["consumption"]], slot_fill=raw_slot_means)
        actual_smooth_frame = _add_time_columns(actual_smooth)
        actual_smooth_frame = actual_smooth_frame.rename(columns={"consumption": "smooth_actual"})

        merged = baseline_pred.merge(smooth_pred, on="slot", how="inner")
        merged = merged.merge(
            actual_raw_frame[["slot", "consumption"]],
            on="slot",
            how="left",
        ).rename(columns={"consumption": "actual_raw"})
        merged = merged.merge(
            actual_smooth_frame[["slot", "smooth_actual"]],
            on="slot",
            how="left",
        )
        merged["baseline_ape"] = _compute_ape(merged["actual_raw"], merged["baseline_pred"])
        merged["pairavg_ape"] = _compute_ape(merged["actual_raw"], merged["pairavg_pred"])
        merged["smooth_ape"] = _compute_ape(merged["smooth_actual"], merged["smooth_pred"])
        merged["date"] = target_date
        slot_records.append(
            merged[
                [
                    "date",
                    "slot",
                    "actual_raw",
                    "smooth_actual",
                    "baseline_pred",
                    "smooth_pred",
                    "template_residual",
                    "pairavg_pred",
                    "baseline_ape",
                    "pairavg_ape",
                    "smooth_ape",
                ]
            ]
        )

        day_records.append(
            {
                "date": target_date,
                "actual_mean": merged["actual_raw"].mean(),
                "smooth_actual_mean": merged["smooth_actual"].mean(),
                "baseline_pred_mean": merged["baseline_pred"].mean(),
                "smooth_pred_mean": merged["smooth_pred"].mean(),
                "pairavg_pred_mean": merged["pairavg_pred"].mean(),
                "baseline_mape": merged["baseline_ape"].mean(),
                "pairavg_mape": merged["pairavg_ape"].mean(),
                "smooth_target_mape": merged["smooth_ape"].mean(),
                "baseline_mae": (merged["actual_raw"] - merged["baseline_pred"]).abs().mean(),
                "pairavg_mae": (merged["actual_raw"] - merged["pairavg_pred"]).abs().mean(),
                "template_abs_mean": merged["template_residual"].abs().mean(),
                "mape_gain": merged["baseline_ape"].mean() - merged["pairavg_ape"].mean(),
            }
        )

    day_summary = pd.DataFrame(day_records).sort_values("date").reset_index(drop=True)
    slot_predictions = pd.concat(slot_records, ignore_index=True)
    return day_summary, slot_predictions


def _save_overall_summary(day_summary):
    summary = pd.DataFrame(
        [
            {
                "metric": "baseline_mean_mape",
                "value": day_summary["baseline_mape"].mean(),
            },
            {
                "metric": "pairavg_mean_mape",
                "value": day_summary["pairavg_mape"].mean(),
            },
            {
                "metric": "smooth_target_mean_mape",
                "value": day_summary["smooth_target_mape"].mean(),
            },
            {
                "metric": "baseline_mean_mae",
                "value": day_summary["baseline_mae"].mean(),
            },
            {
                "metric": "pairavg_mean_mae",
                "value": day_summary["pairavg_mae"].mean(),
            },
            {
                "metric": "mean_mape_gain",
                "value": day_summary["mape_gain"].mean(),
            },
            {
                "metric": "median_mape_gain",
                "value": day_summary["mape_gain"].median(),
            },
            {
                "metric": "num_improved_days",
                "value": day_summary["mape_gain"].gt(0).sum(),
            },
        ]
    )
    summary.to_csv(OVERALL_SUMMARY_PATH, index=False)
    return summary


def _plot_daily_mape(day_summary):
    plt.figure(figsize=(12, 4))
    plt.plot(day_summary["date"], day_summary["baseline_mape"], label="baseline raw mape")
    plt.plot(day_summary["date"], day_summary["pairavg_mape"], label="pairavg two-stage raw mape")
    plt.plot(day_summary["date"], day_summary["smooth_target_mape"], label="pairavg smooth-target mape")
    plt.title("Pair-Average Two-Stage Backtest MAPE")
    plt.xlabel("Date")
    plt.ylabel("MAPE (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DAILY_MAPE_PLOT_PATH, dpi=150)
    plt.close()


def _plot_examples(slot_predictions, day_summary):
    show_dates = (
        day_summary.sort_values("mape_gain", ascending=False)["date"]
        .head(4)
        .sort_values()
        .tolist()
    )
    fig, axes = plt.subplots(len(show_dates), 1, figsize=(12, 12), sharex=True)

    if len(show_dates) == 1:
        axes = [axes]

    for ax, date in zip(axes, show_dates):
        frame = slot_predictions[slot_predictions["date"] == date].sort_values("slot")
        ax.plot(frame["slot"], frame["actual_raw"], label="actual", color="#1f77b4")
        ax.plot(frame["slot"], frame["baseline_pred"], label="baseline", color="#ff7f0e")
        ax.plot(frame["slot"], frame["pairavg_pred"], label="pairavg two-stage", color="#2ca02c")
        ax.set_title(f"{pd.Timestamp(date).date()}  gain={frame['baseline_ape'].mean() - frame['pairavg_ape'].mean():.2f}pp")
        ax.grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(EXAMPLE_PLOT_PATH, dpi=150)
    plt.close()


def main():
    raw_df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"]).set_index("timestamp")
    raw_df = raw_df.loc[_date_mask(raw_df, SEGMENT_START, SEGMENT_END)].copy()

    day_summary, slot_predictions = run_pairavg_two_stage_backtest(raw_df)
    day_summary.to_csv(DAY_SUMMARY_PATH, index=False)
    slot_predictions.to_csv(SLOT_PREDICTIONS_PATH, index=False)
    overall_summary = _save_overall_summary(day_summary)
    _plot_daily_mape(day_summary)
    _plot_examples(slot_predictions, day_summary)

    print(overall_summary.to_string(index=False))
    print("\nTop improved days:")
    print(
        day_summary.sort_values("mape_gain", ascending=False)
        .head(5)[["date", "baseline_mape", "pairavg_mape", "mape_gain"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

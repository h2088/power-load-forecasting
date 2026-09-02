import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pipeline import train, predict
from src.config import N_DELAY, WINDOW_DAYS, START_DATE, END_DATE


# ─── 滚动回测 ─────────────────────────────────────────────────────────────────

def run_backtest(df):
    """
    df:   timestamp-indexed DataFrame，包含 'consumption' 列
    返回: results DataFrame，含所有目标日的预测与实际值
    """
    backtest_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    print(f"回测范围：{START_DATE} ～ {END_DATE}，共 {len(backtest_dates)} 天")
    print("回测中，请耐心等待...\n")

    for target_date in backtest_dates:
        cutoff       = target_date - pd.Timedelta(days=N_DELAY)
        window_start = cutoff - pd.Timedelta(days=WINDOW_DAYS - 1)

        df_train = df[
            (df.index.date >= window_start.date()) &
            (df.index.date <= cutoff.date())
        ]

        if len(df_train) == 0:
            print(f"  {target_date.date()}  跳过：训练数据为空")
            continue

        model, slot_means = train(df_train)
        result, _ = predict(df_train, model, slot_means, target_date=target_date)

        # 合并实际值
        actual = df[df.index.date == target_date.date()].copy()
        actual["slot"] = actual.index.hour * 4 + actual.index.minute // 15
        result = result.merge(actual[["slot", "consumption"]], on="slot", how="left")

        # 计算当日 MAPE 并打印进度
        valid = result.dropna(subset=["consumption"])
        if len(valid) > 0:
            day_mape = np.mean(np.abs((valid["consumption"] - valid["predicted"]) / valid["consumption"])) * 100
            print(f"  {target_date.date()}  MAPE: {day_mape:.2f}%")

        records.append(result)

    results = pd.concat(records, ignore_index=True)
    return results


# ─── 评估 ─────────────────────────────────────────────────────────────────────

def evaluate(results):
    """
    计算并打印整体 MAE 和 MAPE。
    返回: (mae, mape)
    """
    valid = results.dropna(subset=["consumption"])
    mae  = np.mean(np.abs(valid["consumption"] - valid["predicted"]))
    mape = np.mean(np.abs((valid["consumption"] - valid["predicted"]) / valid["consumption"])) * 100

    print(f"\n{'─' * 30}")
    print(f"MAE:  {mae:.3f} MWh")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'─' * 30}")

    return mae, mape


# ─── 画图 ─────────────────────────────────────────────────────────────────────

def plot_daily_mape(results, mae, mape, save_path="daily_mape.png"):
    """
    绘制每日 MAPE 折线图并保存。
    """
    valid = results.dropna(subset=["consumption"])
    daily_mape = (
        valid
        .groupby("date")
        .apply(lambda g: np.mean(np.abs((g["consumption"] - g["predicted"]) / g["consumption"])) * 100)
        .reset_index(name="mape")
    )

    plt.figure(figsize=(12, 4))
    plt.plot(daily_mape["date"], daily_mape["mape"])
    plt.axhline(mape, color="red", linestyle="--", label=f"mean MAPE: {mape:.2f}%")
    plt.title("Daily MAPE")
    plt.xlabel("Date")
    plt.ylabel("MAPE (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"图表已保存至 {save_path}")

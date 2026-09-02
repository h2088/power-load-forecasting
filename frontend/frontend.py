from __future__ import annotations

import io
import os
from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")

st.set_page_config(
    page_title="Power Load Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(249, 201, 95, 0.20), transparent 28%),
            radial-gradient(circle at top right, rgba(62, 148, 226, 0.16), transparent 24%),
            linear-gradient(180deg, #f6f8fb 0%, #edf2f7 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_model_info() -> dict[str, Any]:
    resp = requests.get(f"{BACKEND_URL}/api/model-info", timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_forecast(target_date: date) -> dict[str, Any]:
    resp = requests.post(
        f"{BACKEND_URL}/api/predict",
        json={"target_date": target_date.isoformat()},
        timeout=240,
    )
    resp.raise_for_status()
    return resp.json()


def build_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["predicted"],
            mode="lines",
            name="Predicted",
            line={"color": "#1565c0", "width": 3},
        )
    )

    if frame["actual"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["actual"],
                mode="lines",
                name="Actual",
                line={"color": "#ef6c00", "width": 2, "dash": "dash"},
            )
        )

    figure.update_layout(
        title="96-slot daily load curve",
        xaxis_title="Timestamp",
        yaxis_title="Consumption (MWh)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 20, "r": 20, "t": 56, "b": 20},
        height=480,
    )
    return figure


def build_daily_total_chart(daily_totals: dict[str, Any]) -> go.Figure:
    frame = pd.DataFrame(daily_totals["bars"])
    frame["date"] = pd.to_datetime(frame["date"])

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["actual_total"],
            name="Actual daily total",
            marker_color="#64b5f6",
        )
    )
    figure.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["predicted_total"],
            name="Predicted daily total",
            marker_color="#ffb74d",
        )
    )

    figure.update_layout(
        title="Daily total view: recent actual days and target-day forecast",
        xaxis_title="Date",
        yaxis_title="Daily total consumption (MWh)",
        template="plotly_white",
        barmode="group",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 20, "r": 20, "t": 56, "b": 20},
        height=420,
    )
    return figure


st.title("⚡ Power Load Forecast Dashboard")
st.caption("Production-style serving shell for scheme 1: regime-aware two-stage forecasting on total_consumption.csv")

if "forecast_response" not in st.session_state:
    st.session_state.forecast_response = None

try:
    model_info = get_model_info()
except requests.exceptions.RequestException as exc:
    st.error(f"Cannot reach backend: {exc}")
    st.stop()

dataset = model_info["dataset"]
default_target_date = pd.Timestamp(dataset["latest_predictable_target_date"]).date()
min_target_date = (pd.Timestamp(dataset["earliest_date"]) + pd.Timedelta(days=model_info["n_delay_days"])).date()
max_target_date = default_target_date

with st.sidebar:
    st.subheader("Model")
    st.write(model_info["model_name"])
    st.write(model_info["approach"])

    st.subheader("Dataset")
    st.write(f"Rows: {dataset['row_count']}")
    st.write(f"Actual data until: {dataset['latest_actual_date']}")
    st.write(f"Latest predictable target: {dataset['latest_predictable_target_date']}")
    st.write(f"Data path: `{dataset['data_path']}`")

    st.subheader("Request")
    selected_target_date = st.date_input(
        "Target date",
        value=default_target_date,
        min_value=min_target_date,
        max_value=max_target_date,
    )
    run_button = st.button("Run forecast", use_container_width=True)

    st.subheader("Feature groups")
    for item in model_info["major_feature_groups"]:
        st.write(f"- {item}")

    st.subheader("Regime segments")
    for item in model_info["regime_segments"]:
        st.write(f"- {item}")

if run_button:
    with st.spinner("Training scheme 1 and generating the 96-slot forecast..."):
        try:
            st.session_state.forecast_response = run_forecast(selected_target_date)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None:
                payload = exc.response.json()
                st.error(payload.get("error", "Backend request failed"))
            else:
                st.error("Backend request failed")
        except requests.exceptions.RequestException as exc:
            st.error(f"Backend request failed: {exc}")

response = st.session_state.forecast_response
if response is None:
    st.info("Choose a target date in the sidebar and click 'Run forecast' to generate the scheme 1 prediction.")
    st.stop()

forecast = response["forecast"]
summary = forecast["summary"]
quality = forecast["data_quality"]
frame = pd.DataFrame(forecast["slots"])
daily_totals = forecast["daily_totals"]

metric_cols = st.columns(4)
metric_cols[0].metric("Predicted total", f"{summary['predicted_daily_total']:.2f} MWh")
metric_cols[1].metric("Predicted mean", f"{summary['predicted_daily_mean']:.2f} MWh")
metric_cols[2].metric("Peak slot", f"{summary['predicted_peak_slot']}", f"{summary['predicted_peak_value']:.2f} MWh")
metric_cols[3].metric("Train coverage", f"{quality['train_coverage_ratio'] * 100:.1f}%")

if "actual_mape" in summary:
    eval_cols = st.columns(3)
    eval_cols[0].metric("Actual MAPE", f"{summary['actual_mape']:.2f}%")
    eval_cols[1].metric("Actual MAE", f"{summary['actual_mae']:.2f} MWh")
    eval_cols[2].metric("Bias", f"{summary['actual_bias']:.2f} MWh")

for warning_message in quality["warnings"]:
    st.warning(warning_message)

st.plotly_chart(build_chart(frame), use_container_width=True)
st.caption(
    "The chart below shows recent actual daily totals before the target date, "
    "plus the predicted total for the target day."
)
st.plotly_chart(build_daily_total_chart(daily_totals), use_container_width=True)

info_cols = st.columns(3)
info_cols[0].markdown(
    f"**Target**\n\n`{forecast['target_date']}`"
)
info_cols[1].markdown(
    f"**Cutoff (latest allowed actual date)**\n\n`{forecast['cutoff_date']}`"
)
info_cols[2].markdown(
    f"**Training window**\n\n`{forecast['train_window_start']}` to `{forecast['train_window_end']}`"
)

daily_info_cols = st.columns(2)
daily_info_cols[0].markdown(
    f"**Daily total history window**\n\n"
    f"`{daily_totals['history_start_date']}` to `{daily_totals['history_end_date']}`"
)
daily_info_cols[1].markdown(
    f"**Daily total history days shown**\n\n`{daily_totals['lookback_days']}`"
)

download_buffer = io.StringIO()
frame.to_csv(download_buffer, index=False)
st.download_button(
    "Download forecast csv",
    data=download_buffer.getvalue(),
    file_name=f"forecast_{forecast['target_date']}.csv",
    mime="text/csv",
)

st.dataframe(frame, use_container_width=True, hide_index=True)

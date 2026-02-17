from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def regime_timeline(meta: pd.DataFrame, labels: pd.Series, title: str):
    plot_df = meta.copy()
    plot_df["regime"] = labels.astype(str).values
    fig = px.timeline(
        plot_df,
        x_start="start_time",
        x_end="end_time",
        y="station",
        color="regime",
        title=title,
        hover_data=["start_idx", "end_idx"],
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def sensor_series_with_segments(df: pd.DataFrame, timestamp_col: str, value_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[timestamp_col],
            y=df[value_col],
            mode="lines",
            name=value_col,
        )
    )
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title=value_col)
    return fig


def regime_distribution(window_df: pd.DataFrame, title: str):
    counts = window_df["regime_name"].value_counts().rename_axis("regime_name").reset_index(name="n_windows")
    return px.bar(counts, x="regime_name", y="n_windows", color="regime_name", title=title)


def window_sensor_regime_chart(window_df: pd.DataFrame, sensor_mean_col: str, title: str):
    if sensor_mean_col not in window_df.columns:
        return go.Figure()

    fig = px.scatter(
        window_df.sort_values("start_time"),
        x="start_time",
        y=sensor_mean_col,
        color="regime_name",
        title=title,
        hover_data=["end_time", "station", "regime_id"],
    )
    fig.update_traces(mode="markers+lines")
    fig.update_layout(xaxis_title="Time", yaxis_title=sensor_mean_col)
    return fig

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def regime_timeline(meta: pd.DataFrame, labels: pd.Series, title: str):
    plot_df = meta.copy()
    plot_df["regime"] = labels.astype(str).values
    fig = px.scatter(
        plot_df,
        x="start_time",
        y="station",
        color="regime",
        title=title,
        hover_data=["end_time", "start_idx", "end_idx"],
    )
    fig.update_traces(marker_size=8)
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

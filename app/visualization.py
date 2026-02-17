from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def find_wave_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if c.startswith("WAVE_HGT") and c.endswith("_mean")]
    return sorted(candidates)[0] if candidates else None


def timeline_scatter(df: pd.DataFrame, time_col: str, value_col: str, regime_col: str):
    fig = px.scatter(
        df,
        x=time_col,
        y=value_col,
        color=df[regime_col].astype(str),
        title=f"{value_col} by {regime_col}",
        hover_data=["micro_state", "macro_state"],
    )
    fig.update_traces(mode="markers+lines")
    return fig


def regime_distribution(df: pd.DataFrame, regime_col: str):
    counts = df[regime_col].value_counts().sort_index().rename_axis(regime_col).reset_index(name="count")
    return px.bar(counts, x=regime_col, y="count", color=regime_col, title=f"{regime_col} Distribution")


def transition_heatmap(labels: np.ndarray, title: str):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return go.Figure()
    n = int(max(labels) + 1)
    mat = np.zeros((n, n), dtype=float)
    for i in range(1, len(labels)):
        a, b = labels[i - 1], labels[i]
        if 0 <= a < n and 0 <= b < n:
            mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    fig = px.imshow(mat, text_auto=".2f", aspect="auto", title=title)
    fig.update_layout(xaxis_title="To", yaxis_title="From")
    return fig

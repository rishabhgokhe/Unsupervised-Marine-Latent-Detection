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


def run_length_histogram(labels: np.ndarray, title: str):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return go.Figure()

    runs = []
    current = labels[0]
    run_len = 1
    for i in range(1, len(labels)):
        if labels[i] == current:
            run_len += 1
        else:
            runs.append({"state": int(current), "run_length": int(run_len)})
            current = labels[i]
            run_len = 1
    runs.append({"state": int(current), "run_length": int(run_len)})
    run_df = pd.DataFrame(runs)
    return px.histogram(run_df, x="run_length", color="state", nbins=30, title=title)


def micro_macro_heatmap(micro: np.ndarray, macro: np.ndarray, title: str):
    if len(micro) == 0 or len(macro) == 0:
        return go.Figure()
    ct = pd.crosstab(pd.Series(micro, name="micro"), pd.Series(macro, name="macro"))
    fig = px.imshow(ct.values, text_auto=True, aspect="auto", title=title)
    fig.update_layout(xaxis_title="Macro", yaxis_title="Micro")
    return fig


def feature_profile_heatmap(summary_df: pd.DataFrame, title: str):
    if summary_df.empty:
        return go.Figure()
    z = summary_df.values
    fig = px.imshow(z, aspect="auto", title=title, color_continuous_scale="Blues")
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(len(summary_df.columns))), ticktext=list(summary_df.columns)),
        yaxis=dict(tickmode="array", tickvals=list(range(len(summary_df.index))), ticktext=[str(i) for i in summary_df.index]),
    )
    fig.update_yaxes(title="Regime")
    return fig

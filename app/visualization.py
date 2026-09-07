from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_sensor_column(df: pd.DataFrame, prefix: str) -> Optional[str]:
    matches = [c for c in df.columns if c.startswith(f"{prefix}_") and c.endswith("_mean")]
    return sorted(matches)[0] if matches else None


def find_wave_column(df: pd.DataFrame) -> Optional[str]:
    return find_sensor_column(df, "WAVE_HGT")


# Color palette tailored for oceanographic regimes
REGIME_COLORS = {
    0: "#10b981",  # Calm: Emerald
    1: "#0ea5e9",  # Moderate: Ocean Sky Blue
    2: "#ef4444",  # Storm: Tempest Crimson
    3: "#f59e0b",  # Variance: Frontal Amber
    4: "#8b5cf6",  # Deep Swell: Violet
    5: "#ec4899",  # Secondary: Pink
}


def timeline_scatter(df: pd.DataFrame, time_col: str, value_col: str, regime_col: str):
    fig = px.scatter(
        df,
        x=time_col,
        y=value_col,
        color=df[regime_col].astype(str),
        title=f"{value_col} Evolution by {regime_col}",
        hover_data=["micro_state", "macro_state", "station"] if "station" in df.columns else ["micro_state", "macro_state"],
    )
    fig.update_traces(mode="markers+lines", marker=dict(size=5, opacity=0.85))
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title=value_col,
        template="plotly_dark",
        legend_title="Regime",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def synchronized_multisensor_timeline(df: pd.DataFrame, time_col: str = "end_time", regime_col: str = "macro_state_name"):
    """
    4-Tier Synchronized Multi-Sensor Timeline (Wave Height, Wind Speed, Pressure, Sea Temp)
    Allows evaluators to see the simultaneous atmospheric and oceanic transitions.
    """
    wave_col = find_sensor_column(df, "WAVE_HGT")
    wind_col = find_sensor_column(df, "WIND_SPEED")
    pres_col = find_sensor_column(df, "SEA_LVL_PRES")
    temp_col = find_sensor_column(df, "AIR_TEMP") or find_sensor_column(df, "SEA_SURF_TEMP")

    sensors = [
        ("Wave Height (m)", wave_col, "#00f0ff"),
        ("Wind Speed (kts)", wind_col, "#38bdf8"),
        ("Pressure (hPa)", pres_col, "#f59e0b"),
        ("Temperature (°C)", temp_col, "#10b981"),
    ]
    active_sensors = [s for s in sensors if s[1] is not None and s[1] in df.columns]

    if not active_sensors:
        return go.Figure()

    n_rows = len(active_sensors)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[s[0] for s in active_sensors],
    )

    times = pd.to_datetime(df[time_col], errors="coerce")

    for idx, (label, col, color) in enumerate(active_sensors, start=1):
        vals = df[col].astype(float).copy()
        # Scale NOAA raw integer tenths if applicable for realistic meteorological display
        if "Pressure" in label and vals.median() > 2000:
            vals = vals / 10.0
        elif "Temperature" in label and vals.median() > 50:
            vals = vals / 10.0
        elif "Wind" in label and vals.median() > 30:
            vals = vals / 10.0
        elif "Wave" in label and vals.median() > 10:
            vals = vals / 10.0

        fig.add_trace(
            go.Scatter(
                x=times,
                y=vals,
                mode="lines",
                name=label,
                line=dict(color=color, width=1.8),
                hovertemplate=f"<b>{label}</b>: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>",
            ),
            row=idx,
            col=1,
        )
        fig.update_yaxes(title_text=label.split()[0], row=idx, col=1)

    fig.update_layout(
        height=160 * n_rows + 80,
        template="plotly_dark",
        title_text="Synchronized Multi-Sensor Telemetry & Atmospheric Signature",
        showlegend=False,
        margin=dict(l=40, r=30, t=60, b=30),
    )
    return fig


def latent_space_scatter(
    latent_2d: np.ndarray,
    df: pd.DataFrame,
    regime_col: str = "macro_state_name",
    title: str = "Learned Latent Space Manifold (2D PCA Projection)",
):
    """
    Visualizes the 32-dimensional Autoencoder embeddings compressed into 2D.
    Provides direct visual proof of representation clustering and state separation.
    """
    if isinstance(latent_2d, pd.DataFrame) and not isinstance(df, pd.DataFrame):
        latent_2d, df = df, latent_2d

    latent_arr = np.asarray(latent_2d)
    plot_df = pd.DataFrame(
        {
            "PC1": latent_arr[:, 0],
            "PC2": latent_arr[:, 1],
            "Regime": df[regime_col].astype(str).values,
            "Station": df["station"].values if "station" in df.columns else "N/A",
            "Micro": df["micro_state"].values if "micro_state" in df.columns else 0,
        }
    )
    wave_col = find_sensor_column(df, "WAVE_HGT")
    if wave_col:
        plot_df["Wave Hgt"] = df[wave_col].round(2).values

    wind_col = find_sensor_column(df, "WIND_SPEED")
    if wind_col:
        plot_df["Wind Spd"] = df[wind_col].round(2).values

    hover_cols = ["Station", "Micro"]
    if wave_col:
        hover_cols.append("Wave Hgt")
    if wind_col:
        hover_cols.append("Wind Spd")

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Regime",
        title=title,
        hover_data=hover_cols,
        template="plotly_dark",
        opacity=0.82,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="rgba(255, 255, 255, 0.2)")))
    fig.update_layout(
        xaxis_title="Latent Dimension 1 (Principal Axis)",
        yaxis_title="Latent Dimension 2 (Orthogonal Axis)",
        legend_title="Macro Regime",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def regime_radar_chart(df: pd.DataFrame, regime_col: str = "macro_state_name"):
    """
    Polar / Spider Radar chart showing the 6-parameter normalized feature fingerprint
    (Wave Hgt, Wind Speed, Pressure, Wave Period, Swell Hgt, Duration) for each regime.
    """
    wave_col = find_sensor_column(df, "WAVE_HGT")
    wind_col = find_sensor_column(df, "WIND_SPEED")
    pres_col = find_sensor_column(df, "SEA_LVL_PRES")
    period_col = find_sensor_column(df, "WAVE_PERIOD")
    swell_col = find_sensor_column(df, "SWELL_HGT")

    metrics_map = [
        ("Wave Height", wave_col),
        ("Wind Velocity", wind_col),
        ("Inverse Pressure (Depression)", pres_col),
        ("Wave Period", period_col),
        ("Swell Height", swell_col),
        ("Window Duration", "duration_hours" if "duration_hours" in df.columns else None),
    ]
    active_metrics = [(name, col) for name, col in metrics_map if col and col in df.columns]

    if len(active_metrics) < 3:
        return go.Figure()

    categories = [m[0] for m in active_metrics]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    grp = df.groupby(regime_col)
    regimes = sorted(df[regime_col].dropna().unique().tolist())

    # Normalize each column across regimes to 0.0 - 1.0 for equitable radar axes
    means_dict = {}
    for name, col in active_metrics:
        if name.startswith("Inverse Pressure"):
            # Lower pressure indicates stormier regime -> invert so higher = stormier
            raw = grp[col].mean()
            val = raw.max() - raw
        else:
            val = grp[col].mean()
        lo, hi = val.min(), val.max()
        normed = (val - lo) / (hi - lo) if (hi - lo) > 1e-6 else val * 0 + 0.5
        means_dict[name] = normed

    norm_df = pd.DataFrame(means_dict)

    for r_name in regimes:
        if r_name not in norm_df.index:
            continue
        vals = [float(norm_df.loc[r_name, cat]) for cat in categories]
        vals_closed = vals + [vals[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=categories_closed,
                fill="toself",
                name=str(r_name),
                opacity=0.65,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
        ),
        showlegend=True,
        template="plotly_dark",
        title="Regime Fingerprint Radar (Normalized Physical Dimensions)",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def reconstruction_error_timeline(df: pd.DataFrame, error_col: str = "reconstruction_error", time_col: str = "end_time"):
    """
    Plots the Autoencoder reconstruction error over time with a 3-sigma statistical threshold line.
    """
    if error_col not in df.columns:
        return go.Figure()

    errors = df[error_col].to_numpy(dtype=float)
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    threshold = mean_err + 3.0 * std_err

    times = pd.to_datetime(df[time_col], errors="coerce")
    is_anomaly = errors > threshold

    fig = go.Figure()

    # Base line
    fig.add_trace(
        go.Scatter(
            x=times,
            y=errors,
            mode="lines",
            name="Reconstruction MSE",
            line=dict(color="#00f0ff", width=1.5),
            hovertemplate="Time: %{x}<br>MSE: %{y:.4f}<extra></extra>",
        )
    )

    # Threshold line
    fig.add_trace(
        go.Scatter(
            x=[times.min(), times.max()],
            y=[threshold, threshold],
            mode="lines",
            name=f"Threshold (μ + 3σ = {threshold:.3f})",
            line=dict(color="#ef4444", width=2, dash="dash"),
        )
    )

    # Anomaly marker points
    if np.any(is_anomaly):
        fig.add_trace(
            go.Scatter(
                x=times[is_anomaly],
                y=errors[is_anomaly],
                mode="markers",
                name=f"Anomalous Windows ({int(np.sum(is_anomaly))})",
                marker=dict(color="#f43f5e", size=8, symbol="diamond"),
                hovertemplate="<b>ANOMALY</b><br>Time: %{x}<br>MSE: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="Autoencoder Reconstruction Error Timeline & Anomaly Threshold",
        xaxis_title="Time",
        yaxis_title="Reconstruction Error (MSE)",
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def fleet_risk_gauge(risk_score: float, fleet_level: str = "Low"):
    """
    Radial Gauge chart showing Fleet-Wide Risk Score.
    """
    pct = int(round(risk_score * 100))
    bar_color = "#10b981" if pct < 33 else ("#f59e0b" if pct < 66 else "#ef4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": f"Fleet Early Warning Risk: <b>{fleet_level.upper()}</b>", "font": {"size": 18, "color": "#f8fafc"}},
            number={"suffix": "%", "font": {"size": 36, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#030a16",
                "borderwidth": 1,
                "bordercolor": "#1e293b",
                "steps": [
                    {"range": [0, 33], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [33, 66], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [66, 100], "color": "rgba(239, 68, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.75,
                    "value": 66,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        template="plotly_dark",
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def seasonal_stacked_chart(share_pivot: pd.DataFrame):
    """
    100% Stacked Bar chart showing monthly regime evolution.
    """
    if share_pivot.empty:
        return go.Figure()

    fig = go.Figure()
    for col in share_pivot.columns:
        fig.add_trace(
            go.Bar(
                name=str(col),
                x=share_pivot.index,
                y=share_pivot[col],
                hovertemplate=f"Month: %{{x}}<br><b>{col}</b>: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        title="Seasonal Regime Distribution by Month (%)",
        xaxis_title="Month",
        yaxis_title="Regime Share (%)",
        legend_title="Regime",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def sensor_health_scatter(health_df: pd.DataFrame, station_col: str):
    """
    Interactive 2D scatter of buoy sensor fleet health: Missing Rate vs Overall Health Score.
    """
    if health_df.empty or "health_score" not in health_df.columns:
        return go.Figure()

    fig = px.scatter(
        health_df,
        x="missing_rate",
        y="health_score",
        color="status",
        color_discrete_map={"Good": "#10b981", "Warning": "#f59e0b", "Critical": "#ef4444"},
        hover_data=[station_col, "flatline_rate", "spike_rate", "gap_rate"],
        title="Buoy Sensor Health Fleet Distribution",
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="rgba(255, 255, 255, 0.25)")))
    fig.update_layout(
        xaxis_title="Missing Telemetry Rate",
        yaxis_title="Computed Health Score (0.0 - 1.0)",
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def research_benchmark_bars(model_metrics: Dict[str, Dict[str, float]]):
    """
    Bar chart comparing Silhouette Score and Davies-Bouldin index across models.
    """
    if not model_metrics:
        return go.Figure()

    records = []
    for model_name, metrics in model_metrics.items():
        sil = metrics.get("silhouette_post", metrics.get("silhouette", metrics.get("silhouette_embed")))
        db = metrics.get("davies_bouldin")
        dur = metrics.get("mean_regime_duration")
        if sil is not None:
            records.append(
                {
                    "Model": model_name,
                    "Silhouette Score (Higher is Better)": float(sil),
                    "Davies-Bouldin (Lower is Better)": float(db) if db is not None else np.nan,
                    "Mean Duration (hr)": float(dur) if dur is not None else np.nan,
                }
            )

    if not records:
        return go.Figure()

    df_b = pd.DataFrame(records).sort_values("Silhouette Score (Higher is Better)", ascending=False)
    fig = px.bar(
        df_b,
        x="Model",
        y="Silhouette Score (Higher is Better)",
        color="Model",
        title="Model Benchmark Comparison: Cluster Cohesion (Silhouette Score)",
        text_auto=".3f",
        template="plotly_dark",
    )
    fig.update_layout(
        xaxis_title="Evaluated Architecture",
        yaxis_title="Silhouette Score",
        showlegend=False,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def bic_selection_curve(diag_dict: Dict):
    """
    Plots the Bayesian Information Criterion (BIC) vs candidate states (K) for GMM and HMM.
    Illustrates the optimal model selection elbow.
    """
    gmm_bic = diag_dict.get("model_selection", {}).get("gmm_bic_by_k", {})
    hmm_bic = diag_dict.get("hmm_bic_by_states", {})

    if not gmm_bic and not hmm_bic:
        return go.Figure()

    fig = go.Figure()
    if gmm_bic and isinstance(gmm_bic, dict):
        clean_gmm = {int(k): float(v) for k, v in gmm_bic.items() if str(k).isdigit()}
        if clean_gmm:
            ks = sorted(clean_gmm.keys())
            vals = [clean_gmm[k] for k in ks]
            fig.add_trace(
                go.Scatter(
                    x=ks,
                    y=vals,
                    mode="lines+markers",
                    name="GMM BIC",
                    line=dict(color="#0ea5e9", width=2),
                    marker=dict(size=7),
                )
            )

    if hmm_bic and isinstance(hmm_bic, dict):
        flat_hmm = {int(k): float(v) for k, v in hmm_bic.items() if np.isscalar(v) and str(k).isdigit()}
        if flat_hmm:
            ks = sorted(flat_hmm.keys())
            vals = [flat_hmm[k] for k in ks]
            fig.add_trace(
                go.Scatter(
                    x=ks,
                    y=vals,
                    mode="lines+markers",
                    name="Raw Feature HMM BIC",
                    line=dict(color="#10b981", width=2.5),
                    marker=dict(size=8, symbol="diamond"),
                )
            )

        dense_sub = hmm_bic.get("dense_ae")
        if isinstance(dense_sub, dict):
            clean_dense = {int(k): float(v) for k, v in dense_sub.items() if str(k).isdigit()}
            if clean_dense:
                d_ks = sorted(clean_dense.keys())
                d_vals = [clean_dense[k] for k in d_ks]
                fig.add_trace(
                    go.Scatter(
                        x=d_ks,
                        y=d_vals,
                        mode="lines+markers",
                        name="Dense AE Latent HMM BIC",
                        line=dict(color="#f59e0b", width=2.5, dash="dash"),
                        marker=dict(size=8, symbol="square"),
                    )
                )

    fig.update_layout(
        template="plotly_dark",
        title="State Selection via Bayesian Information Criterion (BIC Minimization)",
        xaxis_title="Candidate States (K)",
        yaxis_title="BIC Score (Lower = Better Fit)",
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def regime_distribution(df: pd.DataFrame, regime_col: str):
    counts = df[regime_col].value_counts().sort_index().rename_axis(regime_col).reset_index(name="count")
    return px.bar(
        counts,
        x=regime_col,
        y="count",
        color=regime_col,
        title=f"{regime_col} Distribution",
        template="plotly_dark",
    )


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

    fig = px.imshow(mat, text_auto=".2f", aspect="auto", title=title, template="plotly_dark")
    fig.update_layout(xaxis_title="To Regime", yaxis_title="From Regime")
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
    fig = px.histogram(run_df, x="run_length", color="state", nbins=30, title=title, template="plotly_dark")
    fig.update_layout(xaxis_title="Run Duration (Steps)", yaxis_title="Frequency")
    return fig


def micro_macro_heatmap(micro: np.ndarray, macro: np.ndarray, title: str):
    if len(micro) == 0 or len(macro) == 0:
        return go.Figure()
    ct = pd.crosstab(pd.Series(micro, name="micro"), pd.Series(macro, name="macro"))
    fig = px.imshow(ct, text_auto=True, aspect="auto", title=title, template="plotly_dark")
    fig.update_layout(xaxis_title="Macro State", yaxis_title="Micro State")
    return fig


def feature_profile_heatmap(summary_df: pd.DataFrame, title: str):
    if summary_df.empty:
        return go.Figure()
    z = summary_df.values
    fig = px.imshow(z, aspect="auto", title=title, color_continuous_scale="Blues", template="plotly_dark")
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(len(summary_df.columns))), ticktext=list(summary_df.columns)),
        yaxis=dict(tickmode="array", tickvals=list(range(len(summary_df.index))), ticktext=[str(i) for i in summary_df.index]),
        yaxis_title="Regime",
    )
    return fig


def geospatial_fleet_map(
    station_map_df: pd.DataFrame,
    color_by: str = "risk_level",
    basemap_mode: str = "🛰️ Satellite (Google Earth)",
    zoom: float = 1.3,
) -> go.Figure:
    """
    Renders an interactive real-world slippy map of marine buoy centers,
    displaying real-time weather conditions, regime classification, and operational risk.
    """
    if (
        station_map_df.empty
        or "latitude" not in station_map_df.columns
        or "longitude" not in station_map_df.columns
    ):
        return go.Figure()

    plot_df = station_map_df.copy()
    plot_df = plot_df.dropna(subset=["latitude", "longitude"])
    if plot_df.empty:
        return go.Figure()

    # Ensure formatted hover fields
    if "risk_score" in plot_df.columns:
        plot_df["risk_pct"] = (plot_df["risk_score"] * 100.0).round(1).astype(str) + "%"
        plot_df["marker_size"] = plot_df["risk_score"].apply(lambda s: 12 + float(s) * 18)
    else:
        plot_df["risk_pct"] = "N/A"
        plot_df["marker_size"] = 14

    color_col = color_by if color_by in plot_df.columns else "risk_level"
    color_discrete_map = None
    if color_col == "risk_level":
        color_discrete_map = {
            "Low": "#10b981",
            "Medium": "#f59e0b",
            "High": "#ef4444",
            "Unknown": "#94a3b8",
        }

    hover_cols = {
        "risk_level": True,
        "risk_pct": True,
        "latitude": ":.2f",
        "longitude": ":.2f",
        "marker_size": False,
    }
    if "macro_state_name" in plot_df.columns:
        hover_cols["macro_state_name"] = True
    if "wind_mean" in plot_df.columns:
        hover_cols["wind_mean"] = ":.1f"
    if "wave_mean" in plot_df.columns:
        hover_cols["wave_mean"] = ":.2f"
    if "pres_mean" in plot_df.columns:
        hover_cols["pres_mean"] = ":.1f"

    center_lat = float(plot_df["latitude"].mean())
    center_lon = float(plot_df["longitude"].mean())

    fig = px.scatter_map(
        plot_df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        color_discrete_map=color_discrete_map,
        hover_name="station" if "station" in plot_df.columns else None,
        hover_data=hover_cols,
        size="marker_size",
        size_max=24,
        zoom=zoom,
        center=dict(lat=center_lat, lon=center_lon),
        labels={
            "risk_level": "Risk Level",
            "risk_pct": "Risk Score",
            "macro_state_name": "Marine Regime",
            "wind_mean": "Wind (m/s)",
            "wave_mean": "Wave (m)",
            "pres_mean": "Pres (hPa)",
            "latitude": "Lat",
            "longitude": "Lon",
        },
    )

    # Apply selected real-world basemap tile layer
    if "Satellite" in basemap_mode:
        fig.update_layout(
            map=dict(
                style="white-bg",
                layers=[
                    {
                        "below": "traces",
                        "sourcetype": "raster",
                        "source": [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ],
                    }
                ],
            )
        )
    elif "OpenStreetMap" in basemap_mode or "Street" in basemap_mode:
        fig.update_layout(map=dict(style="open-street-map"))
    else:
        # Dark Naval Ops default
        fig.update_layout(map=dict(style="carto-darkmatter"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#030a16",
        margin=dict(l=0, r=0, t=10, b=0),
        height=620,
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            font=dict(size=12, color="#f8fafc"),
        ),
    )
    return fig

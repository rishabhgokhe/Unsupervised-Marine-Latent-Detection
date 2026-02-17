from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.config import ProjectConfig, load_config
from src.evaluation.regime_summary import (
    build_window_regime_frame,
    infer_semantic_regime_names,
    regime_summary_table,
)
from src.pipeline.regime_pipeline import run_pipeline, save_artifacts
from src.visualization.plots import (
    regime_distribution,
    regime_timeline,
    sensor_series_with_segments,
    window_sensor_regime_chart,
)


@st.cache_data(show_spinner=False)
def _read_preview(path: str, nrows: int = 10) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def _with_uploaded_data(cfg: ProjectConfig, uploaded_file) -> ProjectConfig:
    tmp_dir = Path(tempfile.gettempdir()) / "marine_regime_streamlit"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    data_path = tmp_dir / uploaded_file.name
    data_path.write_bytes(uploaded_file.getbuffer())
    cfg.data.input_path = str(data_path)
    return cfg


def _get_window_regime_df(result, selected_model: str) -> pd.DataFrame:
    labels = result.model_labels[selected_model]
    window_df = build_window_regime_frame(result.windowed.meta, result.windowed.X, labels)
    name_map = infer_semantic_regime_names(window_df)
    window_df["regime_name"] = window_df["regime_id"].map(name_map).fillna(window_df["regime_id"].astype(str))
    return window_df


def _pick_multiscale_mean_cols(df: pd.DataFrame) -> list[str]:
    preferred_prefix = ["WIND_SPEED", "WAVE_HGT", "SEA_LVL_PRES", "AIR_TEMP"]
    cols = []
    for p in preferred_prefix:
        matches = [c for c in df.columns if c.startswith(f"{p}_") and c.endswith("_mean")]
        cols.extend(sorted(matches)[:2])
    return cols


def main() -> None:
    st.set_page_config(page_title="Marine Regime Discovery", layout="wide")

    default_cfg_path = "configs/config.yml"
    cfg_path = st.sidebar.text_input("Config Path", value=default_cfg_path)
    cfg = load_config(cfg_path)

    st.sidebar.subheader("Model Controls")
    cfg.deep.enabled = st.sidebar.checkbox("Enable Deep LSTM Autoencoder", value=cfg.deep.enabled)
    cfg.deep.enable_vae = st.sidebar.checkbox("Enable VAE Ablation", value=cfg.deep.enable_vae)
    if cfg.deep.enabled:
        cfg.deep.epochs = st.sidebar.slider("Deep Epochs", min_value=5, max_value=100, value=cfg.deep.epochs, step=5)
        cfg.deep.seq_len = st.sidebar.slider("Deep Sequence Length", min_value=2, max_value=24, value=cfg.deep.seq_len, step=1)
    if cfg.deep.enable_vae:
        cfg.deep.vae_latent_dim = st.sidebar.slider("VAE Latent Dim", min_value=2, max_value=64, value=cfg.deep.vae_latent_dim, step=2)
        cfg.deep.vae_beta = st.sidebar.slider("VAE Beta", min_value=0.1, max_value=5.0, value=float(cfg.deep.vae_beta), step=0.1)

    st.title(cfg.app.app_title)
    st.caption("Unsupervised hidden-regime discovery for multivariate marine time-series")
    st.caption(f"Author: {cfg.app.author}")

    uploaded_file = st.file_uploader("Upload CSV (optional, overrides config path)", type=["csv"])
    if uploaded_file is not None:
        cfg = _with_uploaded_data(cfg, uploaded_file)

    st.subheader("Dataset Preview")
    preview_df = _read_preview(cfg.data.input_path)
    st.dataframe(preview_df, use_container_width=True)

    run_button = st.button("Run Full Pipeline", type="primary")
    if run_button:
        with st.spinner("Running ingestion -> preprocessing -> features -> models -> evaluation"):
            result = run_pipeline(cfg)
        st.session_state["pipeline_result"] = result
        st.success("Pipeline complete")

    if "pipeline_result" not in st.session_state:
        st.info("Run the pipeline to view regimes, charts, and statistics.")
        return

    result = st.session_state["pipeline_result"]
    if not result.model_labels:
        st.error("No regime labels produced. Check data and model settings.")
        return

    st.subheader("Model Metrics")
    metrics_df = pd.DataFrame(result.model_metrics).T
    st.dataframe(metrics_df, use_container_width=True)

    available_models = sorted(result.model_labels.keys())
    selected_model = st.selectbox("Choose model", options=available_models)
    window_df = _get_window_regime_df(result, selected_model)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Windows", f"{len(window_df):,}")
    c2.metric("Detected Regimes", int(window_df["regime_id"].nunique()))
    c3.metric("Model", selected_model)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Regime Timeline", "Regime Stats", "Sensor Charts", "Temporal Diagnostics", "Quality", "Downloads"]
    )

    with tab1:
        st.plotly_chart(
            regime_timeline(
                window_df[["station", "start_time", "end_time", "start_idx", "end_idx"]],
                window_df["regime_name"],
                title=f"Regime Timeline ({selected_model})",
            ),
            use_container_width=True,
        )

    with tab2:
        key_features = _pick_multiscale_mean_cols(window_df)
        summary_df = regime_summary_table(window_df, key_features)
        st.dataframe(summary_df, use_container_width=True)
        st.plotly_chart(regime_distribution(window_df, "Regime Distribution by Window Count"), use_container_width=True)
        if result.pca_projection is not None:
            st.subheader("PCA Sanity Projection (Window Feature Space)")
            pca_view = result.pca_projection.copy()
            pca_view["regime_name"] = window_df["regime_name"].values
            st.scatter_chart(data=pca_view, x="pc1", y="pc2", color="regime_name")

    with tab3:
        sensor_cols = [c for c in window_df.columns if c.endswith("_mean")]
        if sensor_cols:
            selected_sensor = st.selectbox("Window Sensor for Regime View", sensor_cols)
            sensor_fig = window_sensor_regime_chart(
                window_df,
                sensor_mean_col=selected_sensor,
                title=f"{selected_sensor} vs Time by Regime",
            )
            if result.changepoints is not None:
                sorted_windows = window_df.sort_values("start_time").reset_index(drop=True)
                cp_indices = [i for i in result.changepoints.break_indices if i < len(sorted_windows)]
                for i in cp_indices:
                    sensor_fig.add_vline(x=sorted_windows.loc[i, "start_time"], line_dash="dash", line_color="red")
            st.plotly_chart(sensor_fig, use_container_width=True)

        raw_sensor_options = [c for c in cfg.data.numeric_columns if c in result.processed_data.columns]
        if raw_sensor_options:
            raw_sensor = st.selectbox("Raw Sensor Series", raw_sensor_options)
            st.plotly_chart(
                sensor_series_with_segments(
                    result.processed_data.sort_values(cfg.data.timestamp_col),
                    timestamp_col=cfg.data.timestamp_col,
                    value_col=raw_sensor,
                    title=f"Raw Time Series: {raw_sensor}",
                ),
                use_container_width=True,
            )

    with tab4:
        st.subheader("Model Selection Curves")
        diags = result.diagnostics or {}
        selection = diags.get("model_selection", {})
        if selection:
            for k, v in selection.items():
                if v:
                    df = pd.DataFrame({"k": list(v.keys()), "score": list(v.values())}).sort_values("k")
                    st.line_chart(df.set_index("k"))

        st.subheader("Duration Statistics")
        dur = (diags.get("duration_stats", {}) if diags else {})
        if dur:
            st.dataframe(pd.DataFrame(dur).T, use_container_width=True)

        st.subheader("Transition Matrix")
        trans = (diags.get("transition_matrices", {}) if diags else {})
        if trans and selected_model in trans:
            st.dataframe(pd.DataFrame(trans[selected_model]), use_container_width=True)

    with tab5:
        st.subheader("Quality Report")
        st.json(result.quality_report)
        if result.changepoints is not None:
            st.subheader("Detected Change Points")
            st.write(result.changepoints.break_indices)

    with tab6:
        out_dir = Path("outputs") / "streamlit_latest"
        save_artifacts(result, out_dir)
        st.caption(f"Artifacts saved at `{out_dir}`")

        metrics_json = json.dumps(result.model_metrics, indent=2)
        st.download_button(
            label="Download model_metrics.json",
            data=metrics_json,
            file_name="model_metrics.json",
            mime="application/json",
        )

        regimes_csv = window_df.copy()
        st.download_button(
            label="Download window_regimes_with_stats.csv",
            data=regimes_csv.to_csv(index=False).encode("utf-8"),
            file_name="window_regimes_with_stats.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

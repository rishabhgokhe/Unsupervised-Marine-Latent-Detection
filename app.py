from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.config import ProjectConfig, load_config
from src.pipeline.regime_pipeline import run_pipeline, save_artifacts
from src.visualization.plots import regime_timeline


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


def main() -> None:
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
    if not run_button:
        return

    with st.spinner("Running ingestion -> preprocessing -> features -> models -> evaluation"):
        result = run_pipeline(cfg)

    st.success("Pipeline complete")

    st.subheader("Model Metrics")
    metrics_df = pd.DataFrame(result.model_metrics).T
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Regime Timelines")
    available_models = sorted(result.model_labels.keys())
    selected_model = st.selectbox("Choose model", options=available_models)
    fig = regime_timeline(
        result.windowed.meta,
        pd.Series(result.model_labels[selected_model]),
        title=f"Window regimes ({selected_model})",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quality Report")
    st.json(result.quality_report)

    if result.changepoints is not None:
        st.subheader("Detected Change Points")
        st.write(result.changepoints.break_indices)

    out_dir = Path("outputs") / "streamlit_latest"
    save_artifacts(result, out_dir)

    st.subheader("Download Artifacts")
    st.caption(f"Artifacts saved at `{out_dir}`")

    metrics_json = json.dumps(result.model_metrics, indent=2)
    st.download_button(
        label="Download model_metrics.json",
        data=metrics_json,
        file_name="model_metrics.json",
        mime="application/json",
    )

    regimes_csv = result.windowed.meta.copy()
    for name, labels in result.model_labels.items():
        regimes_csv[f"regime_{name}"] = labels
    st.download_button(
        label="Download window_regimes.csv",
        data=regimes_csv.to_csv(index=False).encode("utf-8"),
        file_name="window_regimes.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

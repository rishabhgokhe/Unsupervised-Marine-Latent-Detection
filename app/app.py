from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from app.inference import run_inference
from app.model_loader import InferenceModels, load_models
from app.preprocessing import preprocess_input
from app.visualization import find_wave_column, regime_distribution, timeline_scatter, transition_heatmap
from src.core.config import load_config


@st.cache_resource(show_spinner=False)
def load_all_models(artifacts_dir: str) -> InferenceModels:
    return load_models(artifacts_dir)


def _window_output_frame(meta: pd.DataFrame, window_features: pd.DataFrame, micro: pd.Series, macro: pd.Series) -> pd.DataFrame:
    out = meta.copy().reset_index(drop=True)
    out["micro_state"] = micro.values
    out["macro_state"] = macro.values

    mean_cols = [c for c in window_features.columns if c.endswith("_mean")]
    for col in mean_cols:
        out[col] = window_features[col].values
    return out


def main() -> None:
    st.set_page_config(layout="wide", page_title="Marine Regime Segmentation System")
    st.title("Marine Regime Segmentation System")
    st.caption("Inference-only deployment: loads trained artifacts and predicts macro/micro regimes")

    st.sidebar.header("Deployment")
    artifacts_dir = st.sidebar.text_input("Artifacts directory", value="outputs/latest")
    cfg_path = st.sidebar.text_input("Config path", value="configs/config.yml")

    try:
        models = load_all_models(artifacts_dir)
        cfg = load_config(cfg_path)
    except Exception as exc:
        st.error(f"Failed to load models/config: {exc}")
        st.info("Ensure artifacts exist: feature_scaler.pkl, autoencoder_dense.pt, hmm.pkl, macro_mapping.pkl")
        return

    st.sidebar.subheader("Model Info")
    dense_cfg = models.inference_config.get("dense_autoencoder", {})
    st.sidebar.write(f"Dense AE latent dim: {dense_cfg.get('latent_dim', 'n/a')}")
    st.sidebar.write(f"HMM states: {getattr(models.hmm_model, 'n_components', 'n/a')}")
    st.sidebar.write(f"Macro regimes: {len(set(models.macro_mapping.values()))}")

    mode = st.radio("Mode", ["Offline", "Simulated Streaming"], horizontal=True)

    uploaded = st.file_uploader("Upload Marine Dataset CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to run inference.")
        return

    df = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))
    st.success("Dataset loaded successfully")
    st.dataframe(df.head(10), use_container_width=True)

    try:
        prep = preprocess_input(df, cfg=cfg, scaler=models.scaler, inference_config=models.inference_config)
        latent, micro_states, macro_states = run_inference(
            prep.x_scaled,
            ae_model=models.ae_model,
            hmm_model=models.hmm_model,
            macro_mapping=models.macro_mapping,
        )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    out_df = _window_output_frame(
        prep.windowed.meta,
        prep.windowed.X,
        micro=pd.Series(micro_states),
        macro=pd.Series(macro_states),
    )

    if mode == "Simulated Streaming":
        last_n = st.slider("Show last N windows", min_value=20, max_value=max(20, len(out_df)), value=min(200, len(out_df)), step=10)
        out_view = out_df.tail(last_n).reset_index(drop=True)
    else:
        out_view = out_df

    c1, c2, c3 = st.columns(3)
    c1.metric("Windows", f"{len(out_view):,}")
    c2.metric("Micro States", int(out_view["micro_state"].nunique()))
    c3.metric("Macro Regimes", int(out_view["macro_state"].nunique()))

    st.subheader("Regime Timeline")
    wave_col = find_wave_column(out_view)
    if wave_col is not None:
        fig_timeline = timeline_scatter(out_view, time_col="end_time", value_col=wave_col, regime_col="macro_state")
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.warning("No WAVE_HGT mean feature found for timeline chart.")

    st.subheader("Regime Distribution")
    st.plotly_chart(regime_distribution(out_view, "macro_state"), use_container_width=True)

    st.subheader("Regime Statistics")
    feature_cols = [c for c in out_view.columns if c.endswith("_mean")]
    if feature_cols:
        summary = out_view.groupby("macro_state")[feature_cols].mean().sort_index()
        st.dataframe(summary, use_container_width=True)

    st.subheader("Transition Matrix")
    st.plotly_chart(transition_heatmap(out_view["macro_state"].values, "Macro Transition Matrix"), use_container_width=True)

    st.subheader("Download Labels")
    st.download_button(
        label="Download regime_labels.csv",
        data=out_view.to_csv(index=False).encode("utf-8"),
        file_name="regime_labels.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

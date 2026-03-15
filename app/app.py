from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.inference import compute_reconstruction_errors, run_inference
from app.model_loader import InferenceModels, load_models
from app.preprocessing import preprocess_input
from app.visualization import (
    feature_profile_heatmap,
    find_wave_column,
    micro_macro_heatmap,
    regime_distribution,
    run_length_histogram,
    timeline_scatter,
    transition_heatmap,
)
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

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    out["duration_hours"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 3600.0
    return out


def _load_input_df(uploaded_file, sample_path: Path) -> pd.DataFrame | None:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(uploaded_file)
        return pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))
    if sample_path.exists():
        if sample_path.suffix.lower() == ".parquet":
            return pd.read_parquet(sample_path)
        return pd.read_csv(sample_path)
    return None


def _top_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["WIND_SPEED", "WAVE_HGT", "SEA_LVL_PRES", "AIR_TEMP", "SWELL_HGT", "WAVE_PERIOD"]
    selected = []
    for p in preferred:
        cols = [c for c in df.columns if c.startswith(f"{p}_") and c.endswith("_mean")]
        if cols:
            selected.append(sorted(cols)[0])
    return selected[:8]


def _build_regime_notes(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = _top_feature_columns(df)
    if not feature_cols:
        return pd.DataFrame()

    grp = df.groupby("macro_state")
    stats = grp[feature_cols].mean()
    stats["avg_duration_hr"] = grp["duration_hours"].mean()

    note_rows = []
    for rid, row in stats.iterrows():
        hints = []
        for col in feature_cols[:3]:
            hints.append(f"{col}: {row[col]:.2f}")
        note_rows.append(
            {
                "macro_state": int(rid),
                "avg_duration_hr": float(row["avg_duration_hr"]),
                "profile_hint": " | ".join(hints),
                "interpretation": f"Marine regime {int(rid)}",
            }
        )
    return pd.DataFrame(note_rows).sort_values("macro_state")


def _first_mean_col(df: pd.DataFrame, prefix: str) -> str | None:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_") and c.endswith("_mean")]
    return sorted(cols)[0] if cols else None


def _risk_snapshot(df: pd.DataFrame) -> tuple[str, str]:
    wave_col = _first_mean_col(df, "WAVE_HGT")
    wind_col = _first_mean_col(df, "WIND_SPEED")
    if wave_col is None and wind_col is None:
        return "Unknown", "No WAVE_HGT/WIND_SPEED mean features available."

    recent = df.tail(max(20, min(len(df), 120)))
    score = 0.0
    details = []
    if wave_col is not None:
        wave_val = float(recent[wave_col].mean())
        score += wave_val
        details.append(f"wave={wave_val:.2f}")
    if wind_col is not None:
        wind_val = float(recent[wind_col].mean())
        score += wind_val / 15.0
        details.append(f"wind={wind_val:.2f}")

    if score >= 4.5:
        return "High", "Rough/storm tendency in recent windows (" + ", ".join(details) + ")."
    if score >= 2.5:
        return "Medium", "Moderate marine variability (" + ", ".join(details) + ")."
    return "Low", "Relatively calm marine behavior (" + ", ".join(details) + ")."


def _next_macro_probabilities(
    hmm_model: object,
    current_micro: int,
    macro_mapping: dict[int, int] | None,
) -> pd.DataFrame:
    trans = np.asarray(getattr(hmm_model, "transmat_", np.array([])), dtype=float)
    if trans.ndim != 2 or trans.shape[0] == 0:
        return pd.DataFrame()
    if current_micro < 0 or current_micro >= trans.shape[0]:
        return pd.DataFrame()

    p_next_micro = trans[current_micro]
    if macro_mapping is None:
        probs = {int(i): float(p_next_micro[i]) for i in range(len(p_next_micro))}
    else:
        probs: dict[int, float] = {}
        for micro_id, prob in enumerate(p_next_micro):
            macro_id = int(macro_mapping.get(int(micro_id), int(micro_id)))
            probs[macro_id] = probs.get(macro_id, 0.0) + float(prob)

    out = pd.DataFrame(
        [{"macro_state": int(k), "probability": float(v)} for k, v in probs.items()]
    ).sort_values("probability", ascending=False)
    return out.reset_index(drop=True)


def _infer_macro_names(df: pd.DataFrame) -> dict[int, str]:
    if "macro_state" not in df.columns or df.empty:
        return {}
    macro_ids = sorted(int(v) for v in df["macro_state"].dropna().unique())
    if not macro_ids:
        return {}

    wave_col = _first_mean_col(df, "WAVE_HGT")
    wind_col = _first_mean_col(df, "WIND_SPEED")
    pres_col = _first_mean_col(df, "SEA_LVL_PRES")

    grp = df.groupby("macro_state")
    summary = pd.DataFrame(index=macro_ids)
    if wave_col is not None:
        summary["wave"] = grp[wave_col].mean()
    if wind_col is not None:
        summary["wind"] = grp[wind_col].mean()
    if pres_col is not None:
        summary["pressure"] = grp[pres_col].mean()

    if summary.empty:
        return {rid: f"Regime {rid}" for rid in macro_ids}

    score = pd.Series(0.0, index=summary.index, dtype=float)
    if "wave" in summary:
        score += summary["wave"].rank(pct=True)
    if "wind" in summary:
        score += summary["wind"].rank(pct=True)
    if "pressure" in summary:
        # Lower pressure usually indicates harsher weather.
        score += (1.0 - summary["pressure"].rank(pct=True))

    ordered = [int(i) for i in score.sort_values().index.tolist()]

    names: dict[int, str] = {}
    if len(ordered) == 1:
        names[ordered[0]] = "Stable Marine"
        return names
    if len(ordered) == 2:
        labels = ["Calm", "Rough"]
    elif len(ordered) == 3:
        labels = ["Calm", "Rough", "Storm-like"]
    else:
        labels = ["Calm", "Moderate", "Rough", "Storm-like"]

    for i, rid in enumerate(ordered):
        if i < len(labels):
            names[rid] = labels[i]
        else:
            names[rid] = f"Severe-{i - len(labels) + 1}"
    return names


def main() -> None:
    st.set_page_config(layout="wide", page_title="Unsupervised Marine Hidden Regime Discovery")

    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(120deg, #032b44, #0f4c5c 55%, #1d7874);
            border-radius: 14px;
            padding: 18px 22px;
            color: #f4fbff;
            margin-bottom: 10px;
        }
        .hero h1 { margin: 0 0 4px 0; font-size: 1.8rem; }
        .hero p { margin: 0; opacity: 0.92; }
        </style>
        <div class="hero">
            <h1>Hierarchical Latent State Segmentation of Marine Time-Series Data Using HMM</h1>
            <p>NOAA-style marine telemetry -> latent states -> temporal regimes -> operational regime intelligence dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Deployment")
    artifacts_dir = st.sidebar.text_input("Artifacts directory", value="outputs/latest")
    cfg_path = st.sidebar.text_input("Config path", value="configs/config.yml")

    try:
        models = load_all_models(artifacts_dir)
        cfg = load_config(cfg_path)
    except Exception as exc:
        st.error(f"Failed to load models/config: {exc}")
        st.info("Required artifacts: feature_scaler.pkl and hmm.pkl. Optional for hierarchical mode: autoencoder_dense.pt, macro_mapping.pkl, dense_autoencoder_config.json.")
        return

    st.sidebar.subheader("Model Info")
    inf_cfg = models.inference_config or {}
    dense_cfg = inf_cfg.get("dense_autoencoder") or {}
    st.sidebar.write(f"Inference mode: {models.mode}")
    st.sidebar.write(f"Dense AE latent dim: {dense_cfg.get('latent_dim', 'n/a')}")
    st.sidebar.write(f"HMM states: {getattr(models.hmm_model, 'n_components', 'n/a')}")
    macro_count = len(set(models.macro_mapping.values())) if models.macro_mapping else int(getattr(models.hmm_model, "n_components", 0))
    st.sidebar.write(f"Macro regimes: {macro_count}")
    if models.mode == "hmm_only":
        st.sidebar.warning("Fallback mode active: macro states mirror micro states.")

    mode = st.radio("Processing mode", ["Offline", "Simulated Streaming"], horizontal=True)

    sample_path = Path("data/raw/merged final.parquet")
    uploaded = st.file_uploader("Upload Marine Dataset", type=["csv", "parquet"])
    use_sample = st.toggle("Use default NOAA sample dataset (data/raw/merged final.parquet)", value=(uploaded is None))

    df = _load_input_df(uploaded, sample_path if use_sample else Path("__missing__"))
    if df is None:
        st.info("Upload a CSV or enable the default NOAA sample toggle.")
        return

    st.success("Dataset loaded successfully")
    st.dataframe(df.head(10), width="stretch")

    try:
        prep = preprocess_input(df, cfg=cfg, scaler=models.scaler, inference_config=models.inference_config)
        latent, micro_states, macro_states = run_inference(
            prep.x_scaled,
            ae_model=models.ae_model,
            hmm_model=models.hmm_model,
            macro_mapping=models.macro_mapping,
        )
        reconstruction_errors = compute_reconstruction_errors(models.ae_model, prep.x_scaled)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    out_df = _window_output_frame(prep.windowed.meta, prep.windowed.X, micro=pd.Series(micro_states), macro=pd.Series(macro_states))
    macro_name_map = _infer_macro_names(out_df)
    out_df["macro_state_name"] = out_df["macro_state"].map(macro_name_map).fillna(out_df["macro_state"].map(lambda x: f"Regime {int(x)}"))
    if reconstruction_errors is not None and len(reconstruction_errors) == len(out_df):
        out_df["reconstruction_error"] = reconstruction_errors

    if mode == "Simulated Streaming":
        max_len = max(50, len(out_df))
        last_n = st.slider("Live window scope (last N windows)", min_value=50, max_value=max_len, value=min(300, max_len), step=10)
        out_view = out_df.tail(last_n).reset_index(drop=True)
    else:
        out_view = out_df

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Windows", f"{len(out_view):,}")
    c3.metric("Micro States", int(out_view["micro_state"].nunique()))
    c4.metric("Macro Regimes", int(out_view["macro_state_name"].nunique()))
    c5.metric("Avg Duration (hr)", f"{out_view['duration_hours'].mean():.2f}")
    risk_level, risk_note = _risk_snapshot(out_view)
    st.info(f"Operational Risk Meter: **{risk_level}** | {risk_note}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Regime Timeline", "Regime Intelligence", "Operational Insights", "Transitions & Stability", "Feature Profiles", "Export"]
    )

    with tab1:
        st.subheader("Macro Regime Timeline")
        wave_col = find_wave_column(out_view)
        if wave_col is not None:
            st.plotly_chart(timeline_scatter(out_view, time_col="end_time", value_col=wave_col, regime_col="macro_state_name"), width="stretch")
        else:
            st.warning("No WAVE_HGT mean feature found for timeline chart.")

        st.subheader("Regime Distribution")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(regime_distribution(out_view, "macro_state_name"), width="stretch")
        with right:
            st.plotly_chart(regime_distribution(out_view, "micro_state"), width="stretch")

    with tab2:
        st.subheader("Regime Cards")
        regime_counts = out_view["macro_state_name"].value_counts().sort_index()
        cols = st.columns(min(6, len(regime_counts))) if len(regime_counts) > 0 else []
        for i, (rid, cnt) in enumerate(regime_counts.items()):
            cols[i % len(cols)].metric(str(rid), f"{int(cnt)} windows")

        st.subheader("Interpretation Summary")
        notes_df = _build_regime_notes(out_view)
        if not notes_df.empty:
            notes_df["macro_state_name"] = notes_df["macro_state"].map(macro_name_map).fillna(
                notes_df["macro_state"].map(lambda x: f"Regime {int(x)}")
            )
            notes_df["interpretation"] = notes_df["macro_state_name"]
            st.dataframe(notes_df, width="stretch")
        else:
            st.info("No summary features available yet.")

    with tab3:
        st.subheader("Next-Regime Probability")
        current_micro = int(out_view["micro_state"].iloc[-1]) if len(out_view) else -1
        next_probs = _next_macro_probabilities(models.hmm_model, current_micro, models.macro_mapping)
        if not next_probs.empty:
            next_probs["macro_state_name"] = next_probs["macro_state"].map(macro_name_map).fillna(
                next_probs["macro_state"].map(lambda x: f"Regime {int(x)}")
            )
            st.dataframe(next_probs, width="stretch")
            top = next_probs.iloc[0]
            st.success(
                f"Most likely next macro regime: {str(top['macro_state_name'])} "
                f"(p={float(top['probability']):.2f})"
            )
        else:
            st.info("Transition matrix unavailable for probability preview.")

        st.subheader("Top Anomaly Windows (AE Reconstruction Error)")
        if "reconstruction_error" in out_view.columns:
            cols = ["start_time", "end_time", "macro_state_name", "micro_state", "reconstruction_error"]
            wave_col = _first_mean_col(out_view, "WAVE_HGT")
            wind_col = _first_mean_col(out_view, "WIND_SPEED")
            if wave_col is not None:
                cols.append(wave_col)
            if wind_col is not None:
                cols.append(wind_col)
            anom = out_view.nlargest(10, "reconstruction_error")[cols].reset_index(drop=True)
            st.dataframe(anom, width="stretch")
        else:
            st.info("Anomaly panel requires dense autoencoder artifacts.")

    with tab4:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(transition_heatmap(out_view["macro_state"].values, "Macro Transition Matrix"), width="stretch")
            st.plotly_chart(run_length_histogram(out_view["macro_state"].values, "Macro Regime Run-Length Histogram"), width="stretch")
        with right:
            st.plotly_chart(transition_heatmap(out_view["micro_state"].values, "Micro-State Transition Matrix"), width="stretch")
            st.plotly_chart(micro_macro_heatmap(out_view["micro_state"].values, out_view["macro_state"].values, "Micro -> Macro Occupancy"), width="stretch")

    with tab5:
        st.subheader("Regime Feature Profiles")
        selected_features = _top_feature_columns(out_view)
        if selected_features:
            summary = out_view.groupby("macro_state_name")[selected_features].mean().sort_index()
            st.dataframe(summary, width="stretch")
            st.plotly_chart(feature_profile_heatmap(summary, "Macro Regime Feature Heatmap"), width="stretch")
        else:
            st.info("No *_mean feature columns found for profile analysis.")

        st.subheader("Latent Space Snapshot")
        lat_df = pd.DataFrame(latent)
        if lat_df.shape[1] >= 2:
            lat_plot = pd.DataFrame({"latent_1": lat_df.iloc[:, 0], "latent_2": lat_df.iloc[:, 1], "macro": out_view["macro_state_name"].values})
            st.scatter_chart(lat_plot, x="latent_1", y="latent_2", color="macro")

    with tab6:
        st.subheader("Download Regime-Labeled Window Dataset")
        st.download_button(
            label="Download regime_labels.csv",
            data=out_view.to_csv(index=False).encode("utf-8"),
            file_name="regime_labels.csv",
            mime="text/csv",
        )

        st.subheader("Project Reminder")
        st.markdown(
            "This app demonstrates your project goal: **Unsupervised Discovery of Hidden Regimes in Multivariate Time-Series on Marine Data**."
        )


if __name__ == "__main__":
    main()

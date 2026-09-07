from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.inference import compute_reconstruction_errors, run_inference
from app.model_loader import InferenceModels, load_models
from app.preprocessing import preprocess_input
from app.ui_helpers import (
    build_regime_notes,
    compute_latent_pca,
    extract_station_coordinates,
    first_mean_col,
    infer_macro_names,
    macro_severity_map,
    monthly_regime_shares,
    next_macro_probabilities,
    normalize_input_columns,
    operational_planning_summary,
    risk_snapshot,
    sensor_health_report,
    station_early_warning,
    top_feature_columns,
    window_output_frame,
)
import json
from app.visualization import (
    bic_selection_curve,
    feature_profile_heatmap,
    find_wave_column,
    fleet_risk_gauge,
    geospatial_fleet_map,
    latent_space_scatter,
    micro_macro_heatmap,
    reconstruction_error_timeline,
    regime_distribution,
    regime_radar_chart,
    research_benchmark_bars,
    run_length_histogram,
    seasonal_stacked_chart,
    sensor_health_scatter,
    synchronized_multisensor_timeline,
    timeline_scatter,
    transition_heatmap,
)
from src.core.config import load_config


@st.cache_resource(show_spinner=False)
def load_all_models(artifacts_dir: str) -> InferenceModels:
    return load_models(artifacts_dir)


def _safe_parquet_columns(source, requested_cols: list[str] | None) -> list[str] | None:
    if requested_cols is None:
        return None
    geo_candidates = ["LATITUDE", "LONGITUDE", "latitude", "longitude", "LAT", "LON"]
    try:
        import pyarrow.parquet as pq
        available = set(pq.read_schema(source).names)
        actual = [c for c in requested_cols if c in available]
        for g in geo_candidates:
            if g in available and g not in actual:
                actual.append(g)
        return actual if actual else None
    except Exception:
        return requested_cols


def _load_input_df(
    uploaded_file,
    sample_path: Path,
    columns: list[str] | None = None,
    row_cap: int | None = None,
    use_last_rows: bool = True,
) -> pd.DataFrame | None:
    def _apply_row_cap(frame: pd.DataFrame) -> pd.DataFrame:
        if row_cap is None:
            return frame
        return frame.tail(row_cap) if use_last_rows else frame.head(row_cap)

    geo_candidates = {"LATITUDE", "LONGITUDE", "latitude", "longitude", "LAT", "LON"}

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".parquet":
            p_cols = _safe_parquet_columns(uploaded_file, columns)
            df = pd.read_parquet(uploaded_file, columns=p_cols)
            return _apply_row_cap(df)
        read_kwargs = {}
        if columns is not None:
            target_cols = set(columns).union(geo_candidates)
            read_kwargs["usecols"] = lambda c: c in target_cols
        if row_cap is not None and not use_last_rows:
            read_kwargs["nrows"] = row_cap
        df = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")), **read_kwargs)
        return _apply_row_cap(df)
    if sample_path.exists():
        if sample_path.suffix.lower() == ".parquet":
            p_cols = _safe_parquet_columns(sample_path, columns)
            df = pd.read_parquet(sample_path, columns=p_cols)
            return _apply_row_cap(df)
        read_kwargs = {}
        if columns is not None:
            target_cols = set(columns).union(geo_candidates)
            read_kwargs["usecols"] = lambda c: c in target_cols
        if row_cap is not None and not use_last_rows:
            read_kwargs["nrows"] = row_cap
        df = pd.read_csv(sample_path, **read_kwargs)
        return _apply_row_cap(df)
    return None


def _render_docs_page() -> None:
    st.markdown("## Project Docs")
    if st.button("Back to Project Overview", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()
    readme_path = Path("README.md")
    if readme_path.exists():
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.info("README.md not found in project root.")


def main() -> None:
    st.set_page_config(layout="wide", page_title="Marine Time-Series Segmentation Using HMM", page_icon="🌊")

    if st.session_state.get("page") == "docs":
        _render_docs_page()
        return

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .block-container {
            padding: 0.8rem 1rem 1rem 1rem;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.6rem 0 0.8rem 0;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            margin-bottom: 0.35rem;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            margin-bottom: 0.4rem;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border-radius: 10px;
        }
        section[data-testid="stSidebar"] label {
            font-weight: 600;
        }
        .hero {
            background: linear-gradient(135deg, #021B2B, #0B4F6C 55%, #1D7874);
            border-radius: 14px;
            padding: 20px 24px;
            color: #f4fbff;
            margin-bottom: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.18);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .hero h1 {
            margin: 0 0 6px 0;
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff;
        }
        .hero p {
            margin: 0;
            font-size: 0.95rem;
            opacity: 0.92;
            max-width: 840px;
            line-height: 1.5;
            color: #e2f1f8;
        }
        .status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 6px 0 12px 0;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .pill-primary {
            background: rgba(11, 79, 108, 0.35);
            color: #a9d6e5;
            border-color: rgba(11, 79, 108, 0.6);
        }
        .pill-muted {
            background: rgba(255, 255, 255, 0.05);
            color: #cbd5e1;
        }
        .pill-ok {
            background: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
            border-color: rgba(46, 204, 113, 0.3);
        }
        </style>
        <div class="hero">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div style="flex: 1; min-width: 320px;">
                    <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.12); padding: 3px 12px; border-radius: 20px; font-size: 0.76rem; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
                        <span>🌊 Research Thesis</span>
                        <span>•</span>
                        <span>Final Year Major Project</span>
                    </div>
                    <h1>Marine Time-Series Segmentation Using HMM</h1>
                    <p>
                        Unsupervised temporal regime discovery & latent state clustering from high-frequency NOAA ocean buoy telemetry using PyTorch Deep Autoencoders and Gaussian Hidden Markov Models.
                    </p>
                </div>
                <div>
                    <div style="background: rgba(0,0,0,0.3); padding: 10px 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); backdrop-filter: blur(8px);">
                        <div style="font-size: 0.72rem; text-transform: uppercase; color: #94d2bd; font-weight: 700; letter-spacing: 0.5px;">Active Dataset</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin: 2px 0;">currentdataset.parquet</div>
                        <div style="font-size: 0.75rem; color: #a9d6e5;">3.71M Records • 107 Buoys</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    artifacts_dir = "artifacts/latest"
    cfg_path = "configs/config_major.yml" if Path("configs/config_major.yml").exists() else "configs/config.yml"

    with st.sidebar:
        st.markdown("### 🌊 Marine HMM Engine")
        st.caption("Temporal Marine Time-Series Segmentation")
        st.divider()

        mode = st.radio(
            "Processing Workflow",
            ["Offline Batch", "Simulated Streaming"],
            index=0,
            help="Offline processes historical windows; Streaming simulates rolling live telemetry.",
        )

        st.subheader("Data Scope")
        max_rows = st.slider(
            "Telemetry Row Cap",
            min_value=50_000,
            max_value=1_000_000,
            value=300_000,
            step=50_000,
            help="Row limit to ingest from currentdataset.parquet. 300,000 provides deep coverage with sub-second inference.",
        )
        use_last_rows = st.toggle("Focus on Recent Windows", value=True)

        with st.expander("⚙️ Advanced Settings", expanded=False):
            artifacts_dir = st.text_input("Artifacts directory", value=artifacts_dir)
            cfg_path = st.text_input("Config path", value=cfg_path)

    try:
        models = load_all_models(artifacts_dir)
        cfg = load_config(cfg_path)
    except Exception as exc:
        st.error(f"Failed to load models/config: {exc}")
        st.info("Required artifacts: feature_scaler.pkl and hmm.pkl.")
        return

    device_label = "GPU (cuda)" if str(models.device).lower().startswith("cuda") else f"CPU ({models.device})"

    with st.sidebar:
        st.divider()
        st.subheader("Model Engine Spec")
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.04); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.82rem; line-height: 1.6;">
                <div>⚡ <b>Pipeline:</b> Dense AE (288D → 32D)</div>
                <div>🔄 <b>Sequence Model:</b> Gaussian HMM</div>
                <div>🏷️ <b>States:</b> 6 Micro → 4 Macro Regimes</div>
                <div>💻 <b>Device:</b> {device_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sample_candidates = [
        Path("data/data/currentdataset.parquet"),
        Path("data/currentdataset.parquet"),
        Path("data/raw/merged final.parquet"),
        Path("data/raw/real data.csv"),
    ]
    sample_path = next((p for p in sample_candidates if p.exists()), Path("data/data/currentdataset.parquet"))

    st.markdown(
        f"""
        <div class="status-row">
            <span class="status-pill pill-primary">⚡ Compute: {device_label}</span>
            <span class="status-pill pill-primary">🧠 Pipeline: {models.mode}</span>
            <span class="status-pill pill-ok">🟢 Status: Inference Active</span>
            <span class="status-pill pill-muted">📂 Source: {sample_path.name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_src1, col_src2 = st.columns([3, 1])
    with col_src1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; background: rgba(15, 76, 92, 0.25); border: 1px solid rgba(29, 120, 116, 0.4); padding: 10px 16px; border-radius: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">🌊</span>
                <div>
                    <span style="color: #e2f1f8; font-weight: 600;">Primary Telemetry Source:</span> 
                    <code style="color: #38ef7d; background: rgba(0,0,0,0.3); padding: 2px 8px; border-radius: 4px; font-weight: 700;">{sample_path.name}</code>
                    <span style="color: #94d2bd; font-size: 0.85rem; margin-left: 8px;">(High-Frequency NOAA Buoy Archive • 3.71M Records)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_src2:
        show_uploader = st.checkbox("Upload Custom Data", value=False, help="Upload an external CSV or Parquet buoy dataset")

    uploaded = None
    if show_uploader:
        uploaded = st.file_uploader("Upload buoy dataset (CSV or Parquet)", type=["csv", "parquet"])
    use_sample = (uploaded is None)

    inf_cfg = models.inference_config or {}
    inferred_numeric = list(inf_cfg.get("numeric_columns", cfg.data.numeric_columns))
    inferred_directional = list(inf_cfg.get("directional_columns", cfg.data.directional_columns))
    needed_columns = [cfg.data.station_col, cfg.data.timestamp_col, *inferred_numeric, *inferred_directional]

    spinner_text = f"Loading + processing dataset on {device_label}..."
    with st.spinner(spinner_text):
        df = _load_input_df(
            uploaded,
            sample_path if use_sample else Path("__missing__"),
            columns=needed_columns,
            row_cap=max_rows,
            use_last_rows=use_last_rows,
        )
    if df is None:
        st.info("Upload a CSV or enable the default NOAA sample toggle.")
        return

    df = normalize_input_columns(df, cfg.data.station_col, cfg.data.timestamp_col)

    with st.expander("🔍 Inspect Ingested Buoy Telemetry (Sample 10 Rows)", expanded=False):
        st.caption(f"Displaying sample records across {len(df.columns)} telemetry variables for {df[cfg.data.station_col].nunique()} buoy stations.")
        st.dataframe(df.head(10), width="stretch")

    try:
        with st.spinner(f"Preprocessing features on {device_label}..."):
            prep = preprocess_input(df, cfg=cfg, scaler=models.scaler, inference_config=models.inference_config)
        with st.spinner(f"Running inference on {device_label}..."):
            latent, micro_states, macro_states = run_inference(
                prep.x_scaled,
                ae_model=models.ae_model,
                hmm_model=models.hmm_model,
                macro_mapping=models.macro_mapping,
                device=models.device,
            )
            reconstruction_errors = compute_reconstruction_errors(models.ae_model, prep.x_scaled, device=models.device)
            latent_2d = compute_latent_pca(latent)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.exception(exc)
        return

    out_df = window_output_frame(prep.windowed.meta, prep.windowed.X, micro=pd.Series(micro_states), macro=pd.Series(macro_states))
    macro_name_map = infer_macro_names(out_df)
    out_df["macro_state_name"] = out_df["macro_state"].map(macro_name_map).fillna(out_df["macro_state"].map(lambda x: f"Regime {int(x)}"))
    if reconstruction_errors is not None and len(reconstruction_errors) == len(out_df):
        out_df["reconstruction_error"] = reconstruction_errors

    if mode == "Simulated Streaming":
        max_len = max(50, len(out_df))
        last_n = st.slider("Live window scope (last N windows)", min_value=50, max_value=max_len, value=min(300, max_len), step=10)
        out_view = out_df.tail(last_n).reset_index(drop=True)
        latent_view_2d = latent_2d[-last_n:] if latent_2d is not None and len(latent_2d) >= last_n else latent_2d
    else:
        out_view = out_df
        latent_view_2d = latent_2d

    station_col = "station" if "station" in out_view.columns else cfg.data.station_col
    early_warning_df = station_early_warning(out_view, station_col, models.hmm_model, models.macro_mapping)
    geo_df = extract_station_coordinates(df, station_col)
    share_pivot, dominant_months, station_dominant = monthly_regime_shares(out_view, station_col)
    health_df = sensor_health_report(
        prep.processed,
        station_col=cfg.data.station_col,
        timestamp_col=cfg.data.timestamp_col,
        numeric_columns=cfg.data.numeric_columns,
    )
    overall_plan, station_plan = operational_planning_summary(out_view, station_col)

    # Clean Sidebar Exports
    with st.sidebar:
        st.divider()
        with st.expander("📥 Export Inference Data", expanded=False):
            st.download_button(
                label="📥 Regime Labels (CSV)",
                data=out_view.to_csv(index=False).encode("utf-8"),
                file_name="regime_labels.csv",
                mime="text/csv",
                key="dl_regime_labels",
                use_container_width=True,
            )
            if not early_warning_df.empty:
                st.download_button(
                    label="⚠️ Early Warning Signals (CSV)",
                    data=early_warning_df.to_csv(index=False).encode("utf-8"),
                    file_name="early_warning.csv",
                    mime="text/csv",
                    key="dl_early_warning",
                    use_container_width=True,
                )
            if not dominant_months.empty:
                st.download_button(
                    label="📊 Seasonal Summary (CSV)",
                    data=dominant_months.to_csv(index=False).encode("utf-8"),
                    file_name="seasonal_summary.csv",
                    mime="text/csv",
                    key="dl_seasonal_summary",
                    use_container_width=True,
                )
            if not overall_plan.empty:
                st.download_button(
                    label="🗓️ Operational Planning (CSV)",
                    data=overall_plan.to_csv(index=False).encode("utf-8"),
                    file_name="operational_planning.csv",
                    mime="text/csv",
                    key="dl_operational_planning",
                    use_container_width=True,
                )
            if not health_df.empty:
                st.download_button(
                    label="🩺 Sensor Health Report (CSV)",
                    data=health_df.to_csv(index=False).encode("utf-8"),
                    file_name="sensor_health.csv",
                    mime="text/csv",
                    key="dl_sensor_health",
                    use_container_width=True,
                )

    # Executive KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Telemetry Records", f"{len(df):,}", help="Total NOAA buoy observation rows processed")
    c2.metric("Multiscale Windows", f"{len(out_view):,}", help="6h, 24h, 72h sliding windows evaluated")
    c3.metric("Micro Latent States", f"{int(out_view['micro_state'].nunique())}", help="Fine-grained HMM Gaussian mixture states")
    macro_seq = out_view["macro_state"].dropna().values
    if len(macro_seq) > 1:
        diffs = np.diff(macro_seq) != 0
        run_lens = np.diff(np.concatenate([[-1], np.where(diffs)[0], [len(macro_seq) - 1]]))
        step_hr = float(getattr(getattr(cfg, "features", None), "step_size", 6.0))
        persistence_val = float(np.mean(run_lens) * step_hr)
    else:
        persistence_val = float(out_view["duration_hours"].mean()) if "duration_hours" in out_view.columns else 0.0
    c5.metric("Regime Persistence", f"{persistence_val:.1f} hrs", help="Mean continuous duration in a single sea regime")

    risk_level, risk_note = risk_snapshot(out_view)
    risk_color = "#2ecc71" if risk_level == "Low" else "#f39c12" if risk_level == "Moderate" else "#e74c3c"
    dominant_name = out_view["macro_state_name"].mode().iloc[0] if not out_view.empty else "N/A"

    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, rgba(3, 43, 68, 0.45), rgba(15, 76, 92, 0.35)); border-left: 5px solid {risk_color}; border-radius: 10px; padding: 12px 20px; margin: 10px 0 16px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; border: 1px solid rgba(255,255,255,0.08);">
            <div>
                <span style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px; color: #94d2bd; font-weight: 700;">Fleet Operational Advisory</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-top: 2px;">
                    Operational Risk Meter: <span style="color: {risk_color};">{risk_level.upper()}</span> • {risk_note}
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #a9d6e5; background: rgba(0,0,0,0.25); padding: 6px 14px; border-radius: 8px;">
                Dominant Fleet State: <b style="color: #ffffff;">{dominant_name}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab_map, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Overview",
            "Regime Intelligence",
            "Early Warning",
            "🗺️ Fleet Geo Map",
            "Seasonal Insights",
            "Sensor Health",
            "Operational Planning",
            "Research Results",
        ]
    )

    with tab1:
        st.subheader("Overview")
        st.markdown(
            "This tab gives the big picture: how regimes evolve over time, how sensors co-vary, and how frequently each regime appears."
        )

        view_mode = st.radio(
            "Telemetry View",
            ["Synchronized Multi-Sensor Timeline", "Single Metric Scatter"],
            horizontal=True,
            key="overview_view_mode",
        )
        if view_mode == "Synchronized Multi-Sensor Timeline":
            st.plotly_chart(synchronized_multisensor_timeline(out_view), width="stretch")
        else:
            metric_candidates = [c for c in out_view.columns if c.endswith("_mean")]
            default_ix = 0
            for i, c in enumerate(metric_candidates):
                if "WIND_SPEED" in c:
                    default_ix = i
                    break
            selected_metric = st.selectbox("Select Telemetry Variable to Plot", options=metric_candidates, index=default_ix if metric_candidates else 0)
            if selected_metric is not None:
                st.plotly_chart(
                    timeline_scatter(out_view, time_col="end_time", value_col=selected_metric, regime_col="macro_state_name"),
                    width="stretch",
                )

        st.subheader("Regime Physical Fingerprint (Radar Profile)")
        st.markdown(
            "Normalized physical characteristics across 6 key atmospheric and oceanic dimensions. "
            "Shows why the unsupervised model separated these distinct regimes."
        )
        st.plotly_chart(regime_radar_chart(out_view), width="stretch")

        st.subheader("Regime Distribution")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(regime_distribution(out_view, "macro_state_name"), width="stretch")
        with right:
            st.plotly_chart(regime_distribution(out_view, "micro_state"), width="stretch")

        st.subheader("Interpretation Summary")
        st.markdown(
            "We summarize each macro regime using average feature hints and average duration. "
            "This makes the unsupervised clusters explainable."
        )
        notes_df = build_regime_notes(out_view)
        if not notes_df.empty:
            notes_df["macro_state_name"] = notes_df["macro_state"].map(macro_name_map).fillna(
                notes_df["macro_state"].map(lambda x: f"Regime {int(x)}")
            )
            notes_df["interpretation"] = notes_df["macro_state_name"]
            st.dataframe(notes_df, width="stretch")
        else:
            st.info("No summary features available yet.")

    with tab2:
        st.subheader("Regime Intelligence & Latent Dynamics")
        st.markdown(
            "Explore the internal representations learned by the deep autoencoder and the temporal transition structure."
        )

        st.subheader("Learned Latent Space Manifold (2D PCA Projection)")
        st.markdown(
            "Direct visual proof of representation learning: 32-dimensional Autoencoder embeddings "
            "projected onto principal axes and color-coded by discovered macro regimes."
        )
        if latent_view_2d is not None and len(latent_view_2d) == len(out_view):
            st.plotly_chart(latent_space_scatter(out_view, latent_view_2d), width="stretch")
        else:
            st.info("Latent projection requires dense autoencoder embeddings.")

        st.subheader("Autoencoder Reconstruction Anomaly Curve")
        st.markdown(
            "Windows exceeding the statistical threshold (μ + 3σ) indicate unusual sea states or sensor anomalies."
        )
        if "reconstruction_error" in out_view.columns:
            st.plotly_chart(reconstruction_error_timeline(out_view), width="stretch")

        st.subheader("Next-Regime Probability")
        current_micro = int(out_view["micro_state"].iloc[-1]) if len(out_view) else -1
        next_probs = next_macro_probabilities(models.hmm_model, current_micro, models.macro_mapping)
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

        st.subheader("Transitions & Stability")
        st.markdown("These plots show the persistence and cross-occupancy of regimes.")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                transition_heatmap(out_view["macro_state"].values, "Macro Transition Matrix"),
                width="stretch",
            )
            st.plotly_chart(
                run_length_histogram(out_view["macro_state"].values, "Macro Regime Run-Length Histogram"),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                transition_heatmap(out_view["micro_state"].values, "Micro-State Transition Matrix"),
                width="stretch",
            )
            st.plotly_chart(
                micro_macro_heatmap(out_view["micro_state"].values, out_view["macro_state"].values, "Micro -> Macro Occupancy"),
                width="stretch",
            )

    with tab3:
        st.subheader("Early Warning & Risk Assessment")
        st.markdown(
            "Composite Risk Score integrates Station Physical Telemetry (Wind / Wave / Pressure deficit), Regime Severity, and HMM Transition Risk into Adverse Regimes."
        )
        if early_warning_df.empty:
            st.info("Not enough wave/wind features to compute early warning signals.")
        else:
            avg_risk = float(early_warning_df["risk_score"].mean()) if "risk_score" in early_warning_df.columns else 0.05
            fleet_level = "High" if avg_risk >= 0.60 else ("Moderate" if avg_risk >= 0.30 else "Low")

            k1, k2, k3, k4 = st.columns(4)
            n_total = len(early_warning_df)
            n_high = int((early_warning_df["risk_level"] == "High").sum())
            n_med = int((early_warning_df["risk_level"] == "Medium").sum())
            n_low = int((early_warning_df["risk_level"] == "Low").sum())

            k1.metric("Fleet Mean Risk", f"{avg_risk * 100:.1f}%", help="Network-wide average operational risk score")
            k2.metric("High-Risk Buoys", f"{n_high}", delta=f"{n_high / n_total * 100:.1f}%" if n_total else "0%", delta_color="inverse" if n_high > 0 else "normal")
            k3.metric("Medium-Risk Buoys", f"{n_med}", delta=f"{n_med / n_total * 100:.1f}%" if n_total else "0%", delta_color="off")
            k4.metric("Nominal / Low Buoys", f"{n_low}", delta=f"{n_low / n_total * 100:.1f}%" if n_total else "0%", delta_color="normal")

            st.plotly_chart(fleet_risk_gauge(avg_risk, fleet_level), width="stretch")

            st.subheader("Station Risk League Table")
            st.dataframe(early_warning_df.head(25), width="stretch")

            st.subheader("Macro Regime Severity Map")
            severity_map = macro_severity_map(out_view)
            st.dataframe(severity_map, width="stretch")

    with tab_map:
        st.subheader("Geospatial Marine Fleet Surveillance")
        st.markdown(
            "Interactive global telemetry map plotting live buoy centers, active meteorological conditions, and operational risk tiers."
        )

        if geo_df.empty or early_warning_df.empty:
            st.info("No geographical coordinates (LATITUDE/LONGITUDE) detected in current dataset.")
        else:
            st_col_geo = station_col if station_col in geo_df.columns else "STATION"
            st_col_ew = station_col if station_col in early_warning_df.columns else "station"
            map_df = early_warning_df.merge(geo_df, left_on=st_col_ew, right_on=st_col_geo, how="inner")

            if "macro_state" in map_df.columns and "macro_name_map" in locals() and macro_name_map:
                map_df["macro_state_name"] = map_df["macro_state"].map(macro_name_map).fillna(map_df["macro_state"].map(lambda x: f"Regime {int(x)}"))

            if map_df.empty:
                st.info("Buoy station identifiers between coordinates and telemetry could not be matched.")
            else:
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                n_plotted = len(map_df)
                n_high_map = int((map_df["risk_level"] == "High").sum())
                n_med_map = int((map_df["risk_level"] == "Medium").sum())
                n_low_map = int((map_df["risk_level"] == "Low").sum())

                m_c1.metric("Buoy Centers Plotted", f"{n_plotted}", help="Total geolocated marine stations")
                m_c2.metric("High-Risk Hotspots", f"{n_high_map}", delta_color="inverse" if n_high_map > 0 else "normal")
                m_c3.metric("Medium-Risk Zones", f"{n_med_map}")
                m_c4.metric("Calm / Low-Risk Buoys", f"{n_low_map}")

                t_col1, t_col2, t_col3 = st.columns(3)
                with t_col1:
                    basemap_style = st.selectbox(
                        "Basemap View",
                        [
                            "🛰️ Satellite (Google Earth)",
                            "🗺️ OpenStreetMap (Google Maps View)",
                            "⚓ Dark Naval Ops",
                        ],
                        index=0,
                        help="Choose between photorealistic satellite imagery or street/terrain vector tiles",
                    )
                with t_col2:
                    color_encoding = st.selectbox(
                        "Color Markers By",
                        ["risk_level", "macro_state_name"],
                        index=0,
                        format_func=lambda x: "Operational Risk Level (Red/Amber/Green)" if x == "risk_level" else "Marine Regime Classification",
                        help="Switch marker colors between risk level and regime types",
                    )
                with t_col3:
                    risk_filter = st.selectbox(
                        "Filter Buoys on Map",
                        ["All Buoys", "Elevated Risk (High & Medium)", "High Risk Only", "Low Risk Only"],
                        index=0,
                    )

                filtered_map_df = map_df.copy()
                if risk_filter == "High Risk Only":
                    filtered_map_df = filtered_map_df[filtered_map_df["risk_level"] == "High"]
                elif risk_filter == "Elevated Risk (High & Medium)":
                    filtered_map_df = filtered_map_df[filtered_map_df["risk_level"].isin(["High", "Medium"])]
                elif risk_filter == "Low Risk Only":
                    filtered_map_df = filtered_map_df[filtered_map_df["risk_level"] == "Low"]

                if filtered_map_df.empty:
                    st.warning("No buoy stations match the selected risk filter.")
                else:
                    map_fig = geospatial_fleet_map(
                        filtered_map_df,
                        color_by=color_encoding,
                        basemap_mode=basemap_style,
                    )
                    st.plotly_chart(map_fig, width="stretch")

                st.subheader("📍 Buoy Station Telemetry Inspector")
                station_options = sorted(map_df[st_col_ew].dropna().unique().tolist())
                selected_buoy = st.selectbox("Select buoy station to inspect", options=station_options, index=0 if station_options else None)
                if selected_buoy is not None:
                    buoy_data = map_df[map_df[st_col_ew] == selected_buoy].iloc[0]
                    b_card1, b_card2, b_card3, b_card4, b_card5 = st.columns(5)
                    b_card1.metric("Coordinates", f"{buoy_data.get('latitude', 0.0):.2f}°, {buoy_data.get('longitude', 0.0):.2f}°")
                    b_card2.metric("Regime", str(buoy_data.get("macro_state_name", f"Regime {buoy_data.get('macro_state')}")))
                    b_card3.metric("Wind Speed", f"{buoy_data.get('wind_mean', 0.0):.1f} m/s")
                    b_card4.metric("Wave Height", f"{buoy_data.get('wave_mean', 0.0):.2f} m")
                    b_risk = float(buoy_data.get("risk_score", 0.0))
                    b_card5.metric("Risk Score", f"{b_risk * 100:.1f}%", buoy_data.get("risk_level", "Low"))

    with tab4:
        st.subheader("Seasonal Regime Insights")
        st.markdown(
            "Monthly regime behavior across buoy stations demonstrating seasonal cyclonic and calm trends."
        )
        if share_pivot.empty:
            st.info("Seasonal summary requires end_time and macro_state_name columns.")
        else:
            st.subheader("Seasonal Regime Evolution (% Share by Month)")
            st.plotly_chart(seasonal_stacked_chart(share_pivot), width="stretch")

            st.subheader("Monthly Regime Share Data (%)")
            st.dataframe(share_pivot, width="stretch")

            st.subheader("Dominant Regime Per Month")
            wave_col = first_mean_col(out_view, "WAVE_HGT")
            wind_col = first_mean_col(out_view, "WIND_SPEED")
            extra = out_view.copy()
            extra["month"] = pd.to_datetime(extra["end_time"], errors="coerce").dt.to_period("M").astype(str)
            monthly_means = extra.groupby("month")
            if wave_col is not None:
                dominant_months["avg_wave"] = dominant_months["month"].map(monthly_means[wave_col].mean())
            if wind_col is not None:
                dominant_months["avg_wind"] = dominant_months["month"].map(monthly_means[wind_col].mean())
            if not dominant_months.empty and dominant_months["dominant_share"].max() <= 1.0:
                dominant_months["dominant_share"] = (dominant_months["dominant_share"] * 100.0).round(2)
            st.dataframe(dominant_months, width="stretch")

            st.subheader("Station-Level Monthly Dominant Regime")
            st_col = station_col if station_col in out_view.columns else ("station" if "station" in out_view.columns else "STATION")
            stations = sorted(out_view[st_col].dropna().unique().tolist()) if st_col in out_view.columns else []
            selected_station = st.selectbox("Select station", options=stations, index=0 if stations else None)
            if selected_station is not None and not station_dominant.empty:
                dom_st_col = station_col if station_col in station_dominant.columns else ("station" if "station" in station_dominant.columns else "STATION")
                view = station_dominant[station_dominant[dom_st_col] == selected_station].copy()
                if not view.empty and view["dominant_share"].max() <= 1.0:
                    view["dominant_share"] = (view["dominant_share"] * 100.0).round(2)
                st.dataframe(view, width="stretch")

    with tab5:
        st.subheader("Sensor Health Monitoring")
        st.markdown(
            "Composite health score evaluating Missing Data (45%), Flatline Behavior (25%), MAD Spikes (20%), and Timestamp Gaps (10%)."
        )
        if health_df.empty:
            st.info("Not enough numeric columns to compute sensor health.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            total_st = len(health_df)
            n_good = int((health_df["status"] == "Good").sum())
            n_warn = int((health_df["status"] == "Warning").sum())
            n_crit = int((health_df["status"] == "Critical").sum())
            h1.metric("Total Buoy Stations", total_st)
            h2.metric("Good Health", f"{n_good} ({n_good / max(total_st, 1) * 100:.1f}%)")
            h3.metric("Warning Status", f"{n_warn} ({n_warn / max(total_st, 1) * 100:.1f}%)")
            h4.metric("Critical Sensors", f"{n_crit} ({n_crit / max(total_st, 1) * 100:.1f}%)")

            st.plotly_chart(sensor_health_scatter(health_df, station_col=cfg.data.station_col), width="stretch")
            st.dataframe(health_df.head(50), width="stretch")

    with tab6:
        st.subheader("Operational Planning")
        st.markdown(
            "Safe operational windows for maritime transit, fishing, and maintenance based on low-severity regime dominance."
        )
        if overall_plan.empty:
            st.info("Operational planning requires end_time and macro_state columns.")
        else:
            st.subheader("Overall Recommended Months")
            st.dataframe(overall_plan, width="stretch")

            st.subheader("Station-Level Recommended Months")
            stations = sorted(out_view[station_col].dropna().unique().tolist())
            selected_station = st.selectbox("Select station for planning", options=stations, index=0 if stations else None)
            if selected_station is not None:
                view = station_plan[station_plan[station_col] == selected_station].copy()
                st.dataframe(view, width="stretch")

            st.caption("Recommended criteria: low regime share >= 60% and high regime share <= 15%.")

    with tab7:
        st.subheader("Research Results & Empirical Benchmarks")
        st.markdown(
            "Research experiment benchmarks from 107 buoy stations, evaluated for the IEEE conference publication."
        )

        exp_dir = Path("experiments/exp_major_final")
        exp_json = exp_dir / "experiment.json"
        comp_csv = Path("experiments/comparative_results.csv") if Path("experiments/comparative_results.csv").exists() else exp_dir / "comparative_rows.csv"

        metrics_dict = {}
        diag_dict = {}

        # Prioritize rich major experiment metrics (8 models including Dense AE & Macro HMM)
        major_metrics_path = Path(artifacts_dir) / "model_metrics.json"
        if not major_metrics_path.exists():
            major_metrics_path = Path("outputs/major_latest/model_metrics.json")
        if major_metrics_path.exists():
            try:
                metrics_dict = json.loads(major_metrics_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        major_diag_path = Path(artifacts_dir) / "model_diagnostics.json"
        if not major_diag_path.exists():
            major_diag_path = Path("outputs/major_latest/model_diagnostics.json")
        if major_diag_path.exists():
            try:
                diag_dict = json.loads(major_diag_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if exp_json.exists():
            try:
                exp_data = json.loads(exp_json.read_text(encoding="utf-8"))
                if not metrics_dict:
                    metrics_dict = exp_data.get("model_metrics", {})
                if not diag_dict:
                    diag_dict = exp_data.get("diagnostics", {})

                st.subheader("Experiment Metadata")
                meta = {
                    "experiment_id": exp_data.get("experiment_id", "exp_major_final"),
                    "created_at_utc": exp_data.get("created_at_utc"),
                    "selected_model": exp_data.get("selected_model", "dense_ae_hmm_macro"),
                    "notes": exp_data.get("notes", "107 NOAA Buoy Stations • Multi-scale (6h, 24h, 72h) • 288D → 32D Latent"),
                }
                st.json(meta)
            except Exception as exc:
                st.error(f"Failed to parse experiment.json: {exc}")

        if metrics_dict:
            st.subheader("Model Benchmark Comparison (Cluster Cohesion)")
            st.plotly_chart(research_benchmark_bars(metrics_dict), width="stretch")

            st.subheader("Model Metrics Summary Table")
            metrics_df = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "model"})
            st.dataframe(metrics_df, width="stretch")

        if diag_dict:
            st.subheader("State Selection Curve (BIC Minimization)")
            st.plotly_chart(bic_selection_curve(diag_dict), width="stretch")

        if comp_csv.exists():
            st.subheader("Comparative Results Across All Architectures")
            try:
                comp_df = pd.read_csv(comp_csv)
                st.dataframe(comp_df, width="stretch")
            except Exception as exc:
                st.error(f"Failed to read comparative results: {exc}")


if __name__ == "__main__":
    main()


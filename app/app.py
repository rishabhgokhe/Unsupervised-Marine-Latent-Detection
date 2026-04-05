from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from app.inference import compute_reconstruction_errors, run_inference
from app.model_loader import InferenceModels, load_models
from app.preprocessing import preprocess_input
from app.ui_helpers import (
    build_regime_notes,
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

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(uploaded_file, columns=columns)
            return _apply_row_cap(df)
        read_kwargs = {}
        if columns is not None:
            read_kwargs["usecols"] = columns
        if row_cap is not None and not use_last_rows:
            read_kwargs["nrows"] = row_cap
        df = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")), **read_kwargs)
        return _apply_row_cap(df)
    if sample_path.exists():
        if sample_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(sample_path, columns=columns)
            return _apply_row_cap(df)
        read_kwargs = {}
        if columns is not None:
            read_kwargs["usecols"] = columns
        if row_cap is not None and not use_last_rows:
            read_kwargs["nrows"] = row_cap
        df = pd.read_csv(sample_path, **read_kwargs)
        return _apply_row_cap(df)
    return None


def main() -> None:
    st.set_page_config(layout="wide", page_title="Unsupervised Marine Hidden Regime Discovery")

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .block-container {
            padding: 0.6rem 0.8rem 0.9rem 0.8rem;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.5rem 0 0.7rem 0;
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
        section[data-testid="stSidebar"] .stRadio,
        section[data-testid="stSidebar"] .stSlider {
            padding-top: 0.1rem;
        }
        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stToggle {
            margin-bottom: 0.35rem;
        }
        .hero {
            background: linear-gradient(120deg, #032b44, #0f4c5c 55%, #1d7874);
            border-radius: 14px;
            padding: 18px 22px;
            color: #f4fbff;
            margin-bottom: 10px;
        }
        .hero h1 { margin: 0 0 4px 0; font-size: 1.8rem; }
        .hero p { margin: 0; opacity: 0.92; }
        .status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 6px 0 10px 0;
        }
        .status-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(0, 0, 0, 0.08);
        }
        .pill-primary {
            background: #e6f6ff;
            color: #0b3b52;
        }
        .pill-muted {
            background: #eef2f7;
            color: #314155;
        }
        .pill-ok {
            background: #e8f7ef;
            color: #135c2b;
            border-color: #c7ead4;
        }
        .pill-warn {
            background: #fff2e5;
            color: #7a3b00;
            border-color: #ffd6ad;
        }
        </style>
        <div class="hero">
            <h1>Hierarchical Latent State Segmentation of Marine Time-Series Data Using HMM</h1>
            <p>NOAA-style marine telemetry -> latent states -> temporal regimes -> operational regime intelligence dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("**Unsupervised Marine Regime Intelligence**")
        st.caption("Production-grade dashboard for latent regime discovery and operational insight.")
        st.divider()
        st.subheader("Deployment")
        st.caption("Artifacts and runtime configuration")
        artifacts_dir = st.text_input("Artifacts directory", value="artifacts/latest")
        cfg_path = st.text_input("Config path", value="configs/config.yml")
        st.divider()
        st.subheader("Run Mode")
        st.caption("Choose the inference workflow")
        mode = st.radio("Processing mode", ["Offline", "Simulated Streaming"], horizontal=True)

    try:
        models = load_all_models(artifacts_dir)
        cfg = load_config(cfg_path)
    except Exception as exc:
        st.error(f"Failed to load models/config: {exc}")
        st.info("Required artifacts: feature_scaler.pkl and hmm.pkl. Optional for hierarchical mode: autoencoder_dense.pt, macro_mapping.pkl, dense_autoencoder_config.json.")
        return

    device_label = "GPU (cuda)" if str(models.device).lower().startswith("cuda") else f"CPU ({models.device})"
    st.markdown(
        f"""
        <div class="status-row">
            <span class="status-pill pill-primary">Device: {device_label}</span>
            <span class="status-pill pill-muted">Mode: {models.mode}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.divider()
        st.subheader("Model Status")
        st.caption("Auto-detected from artifacts")
        inf_cfg = models.inference_config or {}
        dense_cfg = inf_cfg.get("dense_autoencoder") or {}
        st.markdown(f"**Inference mode:** {models.mode}")
        st.markdown(f"**Torch device:** {models.device}")
        st.markdown(f"**Dense AE latent dim:** {dense_cfg.get('latent_dim', 'n/a')}")
        st.markdown(f"**HMM states:** {getattr(models.hmm_model, 'n_components', 'n/a')}")
        macro_count = (
            len(set(models.macro_mapping.values()))
            if models.macro_mapping
            else int(getattr(models.hmm_model, "n_components", 0))
        )
        st.markdown(f"**Macro regimes:** {macro_count}")
        if models.mode == "hmm_only":
            st.warning("Fallback mode active: macro states mirror micro states.")
        st.divider()
        st.subheader("Artifacts Health")
        art_dir = Path(artifacts_dir)
        required = ["feature_scaler.pkl", "hmm.pkl"]
        optional = ["autoencoder_dense.pt", "macro_mapping.pkl", "dense_autoencoder_config.json"]
        for name in required:
            exists = (art_dir / name).exists()
            pill_class = "pill-ok" if exists else "pill-warn"
            status = "OK" if exists else "MISSING"
            st.markdown(
                f'<span class="status-pill {pill_class}">{name}: {status}</span>',
                unsafe_allow_html=True,
            )
        st.caption("Optional for hierarchical mode")
        for name in optional:
            exists = (art_dir / name).exists()
            pill_class = "pill-ok" if exists else "pill-warn"
            status = "OK" if exists else "MISSING"
            st.markdown(
                f'<span class="status-pill {pill_class}">{name}: {status}</span>',
                unsafe_allow_html=True,
            )

    sample_path = Path("data/raw/merged final.parquet")
    uploaded = st.file_uploader("Upload Marine Dataset", type=["csv", "parquet"])
    use_sample = st.toggle("Use default NOAA sample dataset (data/raw/merged final.parquet)", value=(uploaded is None))

    with st.sidebar:
        st.divider()
        st.subheader("Data Controls")
        st.caption("Tuning for performance and latency")
        max_rows = st.slider(
            "Row cap (max rows to process)",
            min_value=50_000,
            max_value=1_000_000,
            value=300_000,
            step=50_000,
        )
        use_last_rows = st.toggle("Use last N rows (recommended)", value=True)
        st.caption("For CSV uploads, using last N rows may be slower because the full file must be read.")

    inferred_numeric = list(inf_cfg.get("numeric_columns", cfg.data.numeric_columns))
    inferred_directional = list(inf_cfg.get("directional_columns", cfg.data.directional_columns))
    needed_columns = [cfg.data.station_col, cfg.data.timestamp_col, *inferred_numeric, *inferred_directional]

    spinner_text = f"Loading + processing dataset on {device_label}..."
    if uploaded is not None:
        load_hint = "CSV detected (parsing may be slower)." if uploaded.name.lower().endswith(".csv") else "Parquet detected (fast load)."
    elif use_sample:
        load_hint = "Using sample Parquet dataset (fast load)."
    else:
        load_hint = "Awaiting dataset selection."
    st.caption(load_hint)
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
    st.success(f"Dataset loaded successfully. Device: {device_label}")
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
    else:
        out_view = out_df

    station_col = "station" if "station" in out_view.columns else cfg.data.station_col
    early_warning_df = station_early_warning(out_view, station_col, models.hmm_model, models.macro_mapping)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Windows", f"{len(out_view):,}")
    c3.metric("Micro States", int(out_view["micro_state"].nunique()))
    c4.metric("Macro Regimes", int(out_view["macro_state_name"].nunique()))
    c5.metric("Avg Duration (hr)", f"{out_view['duration_hours'].mean():.2f}")
    risk_level, risk_note = risk_snapshot(out_view)
    st.info(f"Operational Risk Meter: **{risk_level}** | {risk_note}")

    share_pivot, dominant_months, station_dominant = monthly_regime_shares(out_view, station_col)
    health_df = sensor_health_report(
        prep.processed,
        station_col=cfg.data.station_col,
        timestamp_col=cfg.data.timestamp_col,
        numeric_columns=cfg.data.numeric_columns,
    )
    overall_plan, station_plan = operational_planning_summary(out_view, station_col)

    with st.sidebar.expander("Tools", expanded=False):
        st.markdown("Detected columns")
        st.write(sorted(df.columns))
        st.markdown("Downloads")
        st.download_button(
            label="Download regime_labels.csv",
            data=out_view.to_csv(index=False).encode("utf-8"),
            file_name="regime_labels.csv",
            mime="text/csv",
            key="dl_regime_labels",
        )
        if not early_warning_df.empty:
            st.download_button(
                label="Download early_warning.csv",
                data=early_warning_df.to_csv(index=False).encode("utf-8"),
                file_name="early_warning.csv",
                mime="text/csv",
                key="dl_early_warning",
            )
        if not dominant_months.empty:
            st.download_button(
                label="Download seasonal_summary.csv",
                data=dominant_months.to_csv(index=False).encode("utf-8"),
                file_name="seasonal_summary.csv",
                mime="text/csv",
                key="dl_seasonal_summary",
            )
        if not overall_plan.empty:
            st.download_button(
                label="Download operational_planning.csv",
                data=overall_plan.to_csv(index=False).encode("utf-8"),
                file_name="operational_planning.csv",
                mime="text/csv",
                key="dl_operational_planning",
            )
        if not health_df.empty:
            st.download_button(
                label="Download sensor_health.csv",
                data=health_df.to_csv(index=False).encode("utf-8"),
                file_name="sensor_health.csv",
                mime="text/csv",
                key="dl_sensor_health",
            )

    share_pivot, dominant_months, station_dominant = monthly_regime_shares(out_view, station_col)
    health_df = sensor_health_report(
        prep.processed,
        station_col=cfg.data.station_col,
        timestamp_col=cfg.data.timestamp_col,
        numeric_columns=cfg.data.numeric_columns,
    )
    overall_plan, station_plan = operational_planning_summary(out_view, station_col)

    with st.sidebar.expander("Tools", expanded=False):
        st.markdown("Detected columns")
        st.write(sorted(df.columns))
        st.markdown("Downloads")
        st.download_button(
            label="Download regime_labels.csv",
            data=out_view.to_csv(index=False).encode("utf-8"),
            file_name="regime_labels.csv",
            mime="text/csv",
        )
        if not early_warning_df.empty:
            st.download_button(
                label="Download early_warning.csv",
                data=early_warning_df.to_csv(index=False).encode("utf-8"),
                file_name="early_warning.csv",
                mime="text/csv",
            )
        if not dominant_months.empty:
            st.download_button(
                label="Download seasonal_summary.csv",
                data=dominant_months.to_csv(index=False).encode("utf-8"),
                file_name="seasonal_summary.csv",
                mime="text/csv",
            )
        if not overall_plan.empty:
            st.download_button(
                label="Download operational_planning.csv",
                data=overall_plan.to_csv(index=False).encode("utf-8"),
                file_name="operational_planning.csv",
                mime="text/csv",
            )
        if not health_df.empty:
            st.download_button(
                label="Download sensor_health.csv",
                data=health_df.to_csv(index=False).encode("utf-8"),
                file_name="sensor_health.csv",
                mime="text/csv",
            )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Overview",
            "Regime Intelligence",
            "Early Warning",
            "Seasonal Insights",
            "Sensor Health",
            "Operational Planning",
            "Research Results",
        ]
    )

    with tab1:
        st.subheader("Overview")
        st.markdown(
            "This tab gives the big picture: how regimes evolve over time and how frequently each regime appears. "
            "Use it to explain the dataset behavior quickly to any audience."
        )

        st.subheader("Regime Timeline")
        wave_col = find_wave_column(out_view)
        if wave_col is not None:
            st.plotly_chart(
                timeline_scatter(out_view, time_col="end_time", value_col=wave_col, regime_col="macro_state_name"),
                width="stretch",
            )
        else:
            st.warning("No WAVE_HGT mean feature found for timeline chart.")

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
        st.subheader("Regime Intelligence")
        st.markdown(
            "This tab answers: what is likely to happen next and how stable the regime transitions are. "
            "It supports short‑term operational decisions."
        )

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

        st.subheader("Top Anomaly Windows")
        st.markdown(
            "Anomalies are windows with high reconstruction error (autoencoder). "
            "These can indicate unusual events or sensor issues."
        )
        if "reconstruction_error" in out_view.columns:
            cols = ["start_time", "end_time", "macro_state_name", "micro_state", "reconstruction_error"]
            wave_col = first_mean_col(out_view, "WAVE_HGT")
            wind_col = first_mean_col(out_view, "WIND_SPEED")
            if wave_col is not None:
                cols.append(wave_col)
            if wind_col is not None:
                cols.append(wind_col)
            anom = out_view.nlargest(10, "reconstruction_error")[cols].reset_index(drop=True)
            st.dataframe(anom, width="stretch")
        else:
            st.info("Anomaly panel requires dense autoencoder artifacts.")

        st.subheader("Transitions & Stability")
        st.markdown("These plots show how often the system switches between regimes.")
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
        st.subheader("Early Warning")
        st.markdown(
            "This section flags stations that may enter high‑risk regimes. "
            "Risk score = 0.7 x current regime severity + 0.3 x probability of switching into a high‑risk regime."
        )
        if early_warning_df.empty:
            st.info("Not enough wave/wind features to compute early warning signals.")
        else:
            st.dataframe(early_warning_df.head(25), width="stretch")
            st.subheader("Macro Regime Severity Map")
            severity_map = macro_severity_map(out_view)
            st.dataframe(severity_map, width="stretch")

    with tab4:
        st.subheader("Seasonal Regime Insights")
        st.markdown(
            "We summarize monthly regime behavior across all stations and highlight the dominant regime per month. "
            "This is tailored for 100 stations per month across 6 months."
        )
        if share_pivot.empty:
            st.info("Seasonal summary requires end_time and macro_state_name columns.")
        else:
            st.subheader("Monthly Regime Share (%)")
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
            dominant_months["dominant_share"] = (dominant_months["dominant_share"] * 100.0).round(2)
            st.dataframe(dominant_months, width="stretch")

            st.subheader("Station-Level Monthly Dominant Regime")
            stations = sorted(out_view[station_col].dropna().unique().tolist())
            selected_station = st.selectbox("Select station", options=stations, index=0 if stations else None)
            if selected_station is not None:
                view = station_dominant[station_dominant[station_col] == selected_station].copy()
                view["dominant_share"] = (view["dominant_share"] * 100.0).round(2)
                st.dataframe(view, width="stretch")

    with tab5:
        st.subheader("Sensor Health Monitoring")
        st.markdown(
            "Health score combines missing data, flatline behavior, spikes, and timestamp gaps. "
            "Lower score means the sensor needs attention."
        )
        if health_df.empty:
            st.info("Not enough numeric columns to compute sensor health.")
        else:
            st.dataframe(health_df.head(50), width="stretch")
            st.caption("Tips: high missing_rate or flatline_rate usually indicates sensor failure or transmission issues.")

    with tab6:
        st.subheader("Operational Planning")
        st.markdown(
            "Goal: identify months that are safest for field operations and maintenance. "
            "We use regime severity derived from wave/wind behavior."
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

            st.caption("Recommended = low regime share >= 60% and high regime share <= 15%.")

    with tab7:
        st.subheader("Research Results")
        st.markdown(
            "This tab shows the saved experiment logs used for the research paper. "
            "These are real results generated by the pipeline, not demo data."
        )

        exp_dir = Path("experiments/exp_major_final")
        exp_json = exp_dir / "experiment.json"
        comp_csv = exp_dir / "comparative_rows.csv"

        if exp_json.exists():
            try:
                exp_data = json.loads(exp_json.read_text())
                st.subheader("Experiment Metadata")
                meta = {
                    "experiment_id": exp_data.get("experiment_id"),
                    "created_at_utc": exp_data.get("created_at_utc"),
                    "selected_model": exp_data.get("selected_model"),
                    "notes": exp_data.get("notes"),
                }
                st.json(meta)

                st.subheader("Model Metrics")
                metrics = exp_data.get("model_metrics", {})
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
                    st.dataframe(metrics_df, width="stretch")
                else:
                    st.info("No model_metrics found in experiment.json.")
            except Exception as exc:
                st.error(f"Failed to parse experiment.json: {exc}")
        else:
            st.info("experiment.json not found under experiments/exp_major_final.")

        if comp_csv.exists():
            st.subheader("Comparative Results (per model run)")
            try:
                comp_df = pd.read_csv(comp_csv)
                st.dataframe(comp_df, width="stretch")
            except Exception as exc:
                st.error(f"Failed to read comparative_rows.csv: {exc}")
        else:
            st.info("comparative_rows.csv not found under experiments/exp_major_final.")


if __name__ == "__main__":
    main()

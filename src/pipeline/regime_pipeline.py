from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core.config import ProjectConfig
from src.data.ingest import read_csv_dataset, resample_by_station
from src.data.preprocess import quality_preprocess, replace_special_missing
from src.evaluation.framework import (
    changepoint_regime_alignment,
    hmm_seed_stability,
    pressure_drop_analysis,
    summarize_state_sensitivity,
    transition_entropy,
)
from src.evaluation.metrics import cluster_quality_scores
from src.evaluation.temporal_diagnostics import duration_statistics, label_transition_matrix
from src.features.window_engine import WindowedFeatures, generate_multiscale_window_features
from src.features.windows import build_sliding_windows
from src.models.changepoint import ChangePointResult, detect_changepoints
from src.models.clustering import ClusterRunOutput, run_clustering_baselines
from src.models.deep_autoencoder import run_lstm_autoencoder_segmentation
from src.models.hierarchy import (
    build_hierarchical_latent_states,
    build_hierarchical_regimes,
    characterize_regimes,
)
from src.models.hmm_model import HMMResult, run_hmm
from src.models.kmeans_model import best_kmeans_by_silhouette
from src.models.label_utils import smooth_labels
from src.models.train_autoencoder import train_dense_autoencoder
from src.models.vae_model import run_vae_ablation

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    processed_data: pd.DataFrame
    windowed: WindowedFeatures
    model_labels: Dict[str, np.ndarray]
    model_metrics: Dict[str, Dict[str, float]]
    quality_report: Dict
    changepoints: Optional[ChangePointResult]
    feature_scaler: Optional[StandardScaler] = None
    pca_projection: Optional[pd.DataFrame] = None
    diagnostics: Optional[Dict] = None
    dense_autoencoder_state: Optional[Dict] = None
    dense_autoencoder_config: Optional[Dict] = None
    dense_latent_projection: Optional[pd.DataFrame] = None
    hierarchical_mapping: Optional[Dict[str, Dict[int, int]]] = None
    macro_regime_characterization: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    trained_models: Optional[Dict[str, Any]] = None
    deployment_config: Optional[Dict] = None


def _ensure_features(cfg: ProjectConfig, df: pd.DataFrame) -> list[str]:
    if cfg.data.numeric_columns:
        present = [c for c in cfg.data.numeric_columns if c in df.columns]
        if len(present) < 4:
            raise ValueError("Too few configured numeric features found in data; expected at least 4 present columns")
        return present
    auto_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    auto_cols = [c for c in auto_cols if c not in [cfg.data.station_col]]
    return auto_cols


def run_pipeline(cfg: ProjectConfig) -> PipelineResult:
    logger.info("Loading dataset from %s", cfg.data.input_path)
    base_df = read_csv_dataset(
        path=cfg.data.input_path,
        timestamp_col=cfg.data.timestamp_col,
        station_col=cfg.data.station_col,
        expected_columns=[*cfg.data.numeric_columns, *cfg.data.directional_columns],
        strict_columns=False,
    )
    all_feature_candidates = [*cfg.data.numeric_columns, *cfg.data.directional_columns]
    base_df = replace_special_missing(base_df, all_feature_candidates, sentinels=[99, 999, 9999, 99999])

    features = _ensure_features(cfg, base_df)
    logger.info("Using %d numeric columns", len(features))

    aligned = resample_by_station(
        base_df,
        station_col=cfg.data.station_col,
        timestamp_col=cfg.data.timestamp_col,
        numeric_columns=features,
        rule=cfg.data.resample_rule,
        directional_columns=cfg.data.directional_columns,
    )

    processed, quality = quality_preprocess(
        aligned,
        group_col=cfg.data.station_col,
        timestamp_col=cfg.data.timestamp_col,
        numeric_columns=features,
        directional_columns=cfg.data.directional_columns,
        small_gap_limit=cfg.preprocess.small_gap_limit,
        medium_gap_limit=cfg.preprocess.medium_gap_limit,
        drop_large_gap_rows=cfg.preprocess.drop_large_gap_rows,
        low_q=cfg.preprocess.clip_quantile_low,
        high_q=cfg.preprocess.clip_quantile_high,
    )

    directional_encoded = [f"{c}_SIN" for c in cfg.data.directional_columns] + [f"{c}_COS" for c in cfg.data.directional_columns]
    model_feature_cols = list(dict.fromkeys([*features, *[c for c in directional_encoded if c in processed.columns]]))

    if cfg.features.use_multiscale:
        windowed = generate_multiscale_window_features(
            processed,
            station_col=cfg.data.station_col,
            timestamp_col=cfg.data.timestamp_col,
            feature_columns=model_feature_cols,
            window_sizes=cfg.features.multi_window_sizes,
            stride=cfg.features.multi_stride,
        )
    else:
        windowed = build_sliding_windows(
            processed,
            station_col=cfg.data.station_col,
            timestamp_col=cfg.data.timestamp_col,
            feature_columns=model_feature_cols,
            window_size=cfg.features.window_size,
            step_size=cfg.features.step_size,
            stats=cfg.features.rolling_features,
        )

    cluster_output: ClusterRunOutput = run_clustering_baselines(
        windowed.X,
        candidate_states=cfg.models.candidate_states,
        random_state=cfg.models.random_state,
    )

    labels: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict = {
        "model_selection": cluster_output.diagnostics,
        "duration_stats": {},
        "transition_matrices": {},
        "transition_entropy": {},
        "hmm_bic_by_states": {},
        "hierarchical": {},
        "stability": {},
        "changepoint_alignment": {},
        "domain_validation": {},
    }
    trained_models: Dict[str, Any] = {}

    for model_result in cluster_output.results:
        denoised = smooth_labels(model_result.labels, cfg.models.min_segment_length)
        labels[model_result.name] = denoised
        quality_score = cluster_quality_scores(cluster_output.transformed, denoised)
        durations = duration_statistics(denoised)
        diagnostics["duration_stats"][model_result.name] = durations
        trans_mat = label_transition_matrix(denoised, int(model_result.n_states))
        diagnostics["transition_matrices"][model_result.name] = trans_mat.tolist()
        diagnostics["transition_entropy"][model_result.name] = float(transition_entropy(trans_mat))
        metrics[model_result.name] = {
            **model_result.score_summary,
            "n_states": float(model_result.n_states),
            "silhouette_post": float(quality_score.silhouette),
            "davies_bouldin": float(quality_score.davies_bouldin),
            "mean_regime_duration": float(durations["mean_duration"]),
        }

    hmm_result: Optional[HMMResult] = run_hmm(
        cluster_output.transformed,
        candidate_states=cfg.models.candidate_states,
        covariance_type=cfg.models.hmm_covariance_type,
        random_state=cfg.models.random_state,
    )
    if hmm_result is not None:
        hmm_labels = smooth_labels(hmm_result.labels, cfg.models.min_segment_length)
        labels["hmm"] = hmm_labels
        hmm_quality = cluster_quality_scores(cluster_output.transformed, hmm_labels)
        hmm_durations = duration_statistics(hmm_labels)
        diagnostics["duration_stats"]["hmm"] = hmm_durations
        diagnostics["transition_matrices"]["hmm"] = hmm_result.transition_matrix.tolist()
        diagnostics["transition_entropy"]["hmm"] = float(transition_entropy(hmm_result.transition_matrix))
        diagnostics["hmm_bic_by_states"] = {int(k): float(v) for k, v in hmm_result.bic_by_states.items()}
        metrics["hmm"] = {
            "n_states": float(hmm_result.n_states),
            "log_likelihood": float(hmm_result.log_likelihood),
            "bic": float(hmm_result.bic),
            "silhouette_post": float(hmm_quality.silhouette),
            "davies_bouldin": float(hmm_quality.davies_bouldin),
            "mean_regime_duration": float(hmm_durations["mean_duration"]),
        }
        trained_models["hmm"] = hmm_result.model
    else:
        logger.warning("hmmlearn not installed or HMM fit failed; skipping HMM")

    cp_result = detect_changepoints(cluster_output.transformed)
    if cp_result is not None:
        metrics["changepoint"] = {"n_breaks": float(cp_result.n_breaks)}
        for name, lbl in labels.items():
            diagnostics["changepoint_alignment"][name] = changepoint_regime_alignment(
                regime_labels=lbl,
                changepoints=cp_result.break_indices,
                tolerance=cfg.evaluation.boundary_tolerance_steps,
            )

    baseline_stability = hmm_seed_stability(
        cluster_output.transformed,
        n_states=(hmm_result.n_states if hmm_result is not None else max(cfg.models.candidate_states)),
        seeds=[0, cfg.models.random_state, 100],
        covariance_type=cfg.models.hmm_covariance_type,
    )
    if baseline_stability is not None:
        diagnostics["stability"]["hmm_raw_features"] = baseline_stability

    dense_autoencoder_state: Optional[Dict] = None
    dense_autoencoder_config: Optional[Dict] = None
    dense_latent_projection: Optional[pd.DataFrame] = None
    hierarchical_mapping: Dict[str, Dict[int, int]] = {}
    macro_regime_characterization: Dict[str, Dict[str, float]] = {}
    if cfg.deep.enable_dense_ae:
        dense_output = train_dense_autoencoder(
            cluster_output.transformed,
            latent_dim=cfg.deep.dense_latent_dim,
            epochs=cfg.deep.dense_epochs,
            batch_size=cfg.deep.dense_batch_size,
            lr=cfg.deep.dense_learning_rate,
            random_state=cfg.models.random_state,
        )
        if dense_output is not None:
            dense_autoencoder_state = dense_output.state_dict
            dense_autoencoder_config = {
                "input_dim": int(cluster_output.transformed.shape[1]),
                "latent_dim": int(cfg.deep.dense_latent_dim),
                "epochs": int(cfg.deep.dense_epochs),
                "batch_size": int(cfg.deep.dense_batch_size),
                "learning_rate": float(cfg.deep.dense_learning_rate),
                "random_state": int(cfg.models.random_state),
            }

            _, dense_km_labels, dense_km_k, dense_km_sil, _ = best_kmeans_by_silhouette(
                dense_output.latent_embeddings,
                candidate_ks=cfg.models.candidate_states,
                random_state=cfg.models.random_state,
            )
            dense_km_labels = smooth_labels(dense_km_labels, cfg.models.min_segment_length)
            labels["dense_ae_kmeans"] = dense_km_labels
            dense_km_quality = cluster_quality_scores(dense_output.latent_embeddings, dense_km_labels)
            dense_km_durations = duration_statistics(dense_km_labels)
            diagnostics["duration_stats"]["dense_ae_kmeans"] = dense_km_durations
            dense_km_trans = label_transition_matrix(dense_km_labels, int(dense_km_k))
            diagnostics["transition_matrices"]["dense_ae_kmeans"] = dense_km_trans.tolist()
            diagnostics["transition_entropy"]["dense_ae_kmeans"] = float(transition_entropy(dense_km_trans))
            metrics["dense_ae_kmeans"] = {
                "n_states": float(dense_km_k),
                "silhouette_embed": float(dense_km_sil),
                "silhouette_post": float(dense_km_quality.silhouette),
                "davies_bouldin": float(dense_km_quality.davies_bouldin),
                "mean_regime_duration": float(dense_km_durations["mean_duration"]),
                "reconstruction_mse": float(dense_output.reconstruction_mse),
                "final_train_loss": float(dense_output.final_train_loss),
            }

            dense_hmm_result: Optional[HMMResult] = run_hmm(
                dense_output.latent_embeddings,
                candidate_states=cfg.models.candidate_states,
                covariance_type=cfg.models.hmm_covariance_type,
                random_state=cfg.models.random_state,
            )
            if dense_hmm_result is not None:
                dense_hmm_labels = smooth_labels(dense_hmm_result.labels, cfg.models.min_segment_length)
                labels["dense_ae_hmm"] = dense_hmm_labels
                dense_hmm_quality = cluster_quality_scores(dense_output.latent_embeddings, dense_hmm_labels)
                dense_hmm_durations = duration_statistics(dense_hmm_labels)
                diagnostics["duration_stats"]["dense_ae_hmm"] = dense_hmm_durations
                diagnostics["transition_matrices"]["dense_ae_hmm"] = dense_hmm_result.transition_matrix.tolist()
                diagnostics["transition_entropy"]["dense_ae_hmm"] = float(transition_entropy(dense_hmm_result.transition_matrix))
                diagnostics["hmm_bic_by_states"]["dense_ae"] = {
                    int(k): float(v) for k, v in dense_hmm_result.bic_by_states.items()
                }
                metrics["dense_ae_hmm"] = {
                    "n_states": float(dense_hmm_result.n_states),
                    "log_likelihood": float(dense_hmm_result.log_likelihood),
                    "bic": float(dense_hmm_result.bic),
                    "silhouette_post": float(dense_hmm_quality.silhouette),
                    "davies_bouldin": float(dense_hmm_quality.davies_bouldin),
                    "mean_regime_duration": float(dense_hmm_durations["mean_duration"]),
                    "reconstruction_mse": float(dense_output.reconstruction_mse),
                    "final_train_loss": float(dense_output.final_train_loss),
                }
                trained_models["hmm"] = dense_hmm_result.model

                hier_latent = build_hierarchical_latent_states(
                    micro_states=dense_hmm_labels,
                    state_means=dense_hmm_result.state_means,
                    n_macro=cfg.models.n_super_regimes,
                    random_state=cfg.models.random_state,
                )
                if hier_latent is not None:
                    labels["dense_ae_hmm_macro"] = hier_latent.macro_states
                    hierarchical_mapping["dense_ae_hmm"] = hier_latent.macro_mapping
                    macro_quality = cluster_quality_scores(dense_output.latent_embeddings, hier_latent.macro_states)
                    macro_durations = duration_statistics(hier_latent.macro_states)
                    diagnostics["duration_stats"]["dense_ae_hmm_macro"] = macro_durations
                    diagnostics["transition_matrices"]["dense_ae_hmm_macro"] = hier_latent.macro_transition_matrix.tolist()
                    diagnostics["transition_entropy"]["dense_ae_hmm_macro"] = float(
                        transition_entropy(hier_latent.macro_transition_matrix)
                    )
                    diagnostics["hierarchical"]["dense_ae_hmm"] = {
                        "n_macro": int(hier_latent.n_macro),
                        "micro_to_macro": {int(k): int(v) for k, v in hier_latent.macro_mapping.items()},
                    }
                    metrics["dense_ae_hmm_macro"] = {
                        "n_states": float(hier_latent.n_macro),
                        "silhouette_post": float(macro_quality.silhouette),
                        "davies_bouldin": float(macro_quality.davies_bouldin),
                        "mean_regime_duration": float(macro_durations["mean_duration"]),
                    }

                    macro_summary = characterize_regimes(windowed.X, hier_latent.macro_states)
                    macro_regime_characterization["dense_ae_hmm_macro"] = {
                        str(idx): {col: float(val) for col, val in row.items()}
                        for idx, row in macro_summary.iterrows()
                    }
                    diagnostics["domain_validation"]["dense_ae_hmm_macro"] = {
                        "pressure_drop": pressure_drop_analysis(windowed.X, hier_latent.macro_states)
                    }
                else:
                    logger.warning("Failed to build hierarchical latent states from dense AE HMM output")

                dense_stability = hmm_seed_stability(
                    dense_output.latent_embeddings,
                    n_states=dense_hmm_result.n_states,
                    seeds=[0, cfg.models.random_state, 100],
                    covariance_type=cfg.models.hmm_covariance_type,
                )
                if dense_stability is not None:
                    diagnostics["stability"]["hmm_dense_latent"] = dense_stability

            dense_latent_projection = pd.DataFrame(
                {
                    "latent_pc1": dense_output.latent_pca_2d[:, 0],
                    "latent_pc2": dense_output.latent_pca_2d[:, 1],
                }
            )
            dense_latent_projection = pd.concat([windowed.meta.reset_index(drop=True), dense_latent_projection], axis=1)
            diagnostics["reconstruction_error"] = {
                "mean": float(np.mean(dense_output.reconstruction_errors)),
                "median": float(np.median(dense_output.reconstruction_errors)),
                "p95": float(np.percentile(dense_output.reconstruction_errors, 95)),
                "per_window": dense_output.reconstruction_errors.astype(float).tolist(),
            }
        else:
            logger.warning("Dense autoencoder unavailable or failed; skipping dense autoencoder branch")

    if cfg.deep.enabled:
        deep_result = run_lstm_autoencoder_segmentation(
            cluster_output.transformed,
            candidate_states=cfg.models.candidate_states,
            random_state=cfg.models.random_state,
            seq_len=cfg.deep.seq_len,
            hidden_dim=cfg.deep.hidden_dim,
            latent_dim=cfg.deep.latent_dim,
            epochs=cfg.deep.epochs,
            batch_size=cfg.deep.batch_size,
            learning_rate=cfg.deep.learning_rate,
        )
        if deep_result is not None:
            deep_labels = smooth_labels(deep_result.labels, cfg.models.min_segment_length)
            labels["deep_lstm_ae"] = deep_labels
            deep_quality = cluster_quality_scores(cluster_output.transformed, deep_labels)
            deep_trans = label_transition_matrix(deep_labels, int(deep_result.n_states))
            diagnostics["transition_matrices"]["deep_lstm_ae"] = deep_trans.tolist()
            diagnostics["transition_entropy"]["deep_lstm_ae"] = float(transition_entropy(deep_trans))
            metrics["deep_lstm_ae"] = {
                "n_states": float(deep_result.n_states),
                "train_loss": float(deep_result.train_loss),
                "silhouette_embed": float(deep_result.silhouette),
                "silhouette_post": float(deep_quality.silhouette),
                "davies_bouldin": float(deep_quality.davies_bouldin),
            }
        else:
            logger.warning("Deep LSTM autoencoder unavailable or failed; skipping deep model")

    if cfg.deep.enable_vae:
        vae_result = run_vae_ablation(
            cluster_output.transformed,
            candidate_states=cfg.models.candidate_states,
            random_state=cfg.models.random_state,
            latent_dim=cfg.deep.vae_latent_dim,
            hidden_dim=cfg.deep.hidden_dim,
            epochs=cfg.deep.epochs,
            batch_size=cfg.deep.batch_size,
            learning_rate=cfg.deep.learning_rate,
            beta=cfg.deep.vae_beta,
        )
        if vae_result is not None:
            vae_labels = smooth_labels(vae_result.labels, cfg.models.min_segment_length)
            labels["vae_ablation"] = vae_labels
            vae_quality = cluster_quality_scores(cluster_output.transformed, vae_labels)
            vae_trans = label_transition_matrix(vae_labels, int(vae_result.n_states))
            diagnostics["transition_matrices"]["vae_ablation"] = vae_trans.tolist()
            diagnostics["transition_entropy"]["vae_ablation"] = float(transition_entropy(vae_trans))
            metrics["vae_ablation"] = {
                "n_states": float(vae_result.n_states),
                "recon_loss": float(vae_result.recon_loss),
                "kl_loss": float(vae_result.kl_loss),
                "silhouette_embed": float(vae_result.silhouette),
                "silhouette_post": float(vae_quality.silhouette),
                "davies_bouldin": float(vae_quality.davies_bouldin),
            }
        else:
            logger.warning("VAE ablation unavailable or failed; skipping VAE")

    hierarchy_source = None
    for candidate in ("dense_ae_hmm", "dense_ae_kmeans", "hmm", "vae_ablation", "deep_lstm_ae", "gmm", "kmeans"):
        if candidate in labels:
            hierarchy_source = candidate
            break
    if hierarchy_source is not None:
        hierarchy = build_hierarchical_regimes(
            cluster_output.transformed,
            labels[hierarchy_source],
            n_super_regimes=cfg.models.n_super_regimes,
        )
        if hierarchy is not None:
            labels["hierarchy_super"] = hierarchy.super_regimes
            metrics["hierarchy_super"] = {
                "source_model": float(list(labels.keys()).index(hierarchy_source)),
                "n_super_regimes": float(hierarchy.n_super_regimes),
            }

    pca_df: Optional[pd.DataFrame] = None
    if cfg.features.save_pca_sanity and len(cluster_output.transformed) >= 3:
        pca = PCA(n_components=2, random_state=cfg.models.random_state)
        p = pca.fit_transform(cluster_output.transformed)
        pca_df = pd.DataFrame({"pc1": p[:, 0], "pc2": p[:, 1]})
        pca_df = pd.concat([windowed.meta.reset_index(drop=True), pca_df], axis=1)

    if cp_result is not None:
        for name, lbl in labels.items():
            diagnostics["changepoint_alignment"][name] = changepoint_regime_alignment(
                regime_labels=lbl,
                changepoints=cp_result.break_indices,
                tolerance=cfg.evaluation.boundary_tolerance_steps,
            )

    model_selection_diag = diagnostics.get("model_selection", {})
    hmm_bic_any = diagnostics.get("hmm_bic_by_states", {})
    hmm_bic_flat = {int(k): float(v) for k, v in hmm_bic_any.items() if np.isscalar(v)} if isinstance(hmm_bic_any, dict) else {}
    diagnostics["sensitivity"] = summarize_state_sensitivity(
        kmeans_silhouette_by_k={int(k): float(v) for k, v in model_selection_diag.get("kmeans_silhouette_by_k", {}).items()},
        gmm_bic_by_k={int(k): float(v) for k, v in model_selection_diag.get("gmm_bic_by_k", {}).items()},
        hmm_bic_by_k=(hmm_bic_flat or None),
    )

    deployment_config = {
        "timestamp_col": cfg.data.timestamp_col,
        "station_col": cfg.data.station_col,
        "numeric_columns": list(features),
        "directional_columns": [c for c in cfg.data.directional_columns if c in processed.columns],
        "model_feature_cols": model_feature_cols,
        "windowing": {
            "use_multiscale": bool(cfg.features.use_multiscale),
            "window_size": int(cfg.features.window_size),
            "step_size": int(cfg.features.step_size),
            "rolling_features": list(cfg.features.rolling_features),
            "multi_window_sizes": list(cfg.features.multi_window_sizes),
            "multi_stride": int(cfg.features.multi_stride),
        },
        "dense_autoencoder": dense_autoencoder_config,
    }

    return PipelineResult(
        processed_data=processed,
        windowed=windowed,
        model_labels=labels,
        model_metrics=metrics,
        quality_report={
            "missing_ratio_before": quality.missing_ratio_before,
            "missing_ratio_after": quality.missing_ratio_after,
            "outlier_clip_bounds": quality.outlier_clip_bounds,
            "selected_numeric_features": features,
            "selected_directional_features": [c for c in cfg.data.directional_columns if c in processed.columns],
            "final_model_feature_count": len(model_feature_cols),
            "window_feature_dimension": int(windowed.X.shape[1]),
            "window_count": int(windowed.X.shape[0]),
            "final_has_nan": bool(processed[features].isna().any().any()),
            "dense_ae_enabled": bool(cfg.deep.enable_dense_ae),
        },
        changepoints=cp_result,
        feature_scaler=cluster_output.scaler,
        pca_projection=pca_df,
        diagnostics=diagnostics,
        dense_autoencoder_state=dense_autoencoder_state,
        dense_autoencoder_config=dense_autoencoder_config,
        dense_latent_projection=dense_latent_projection,
        hierarchical_mapping=hierarchical_mapping or None,
        macro_regime_characterization=macro_regime_characterization or None,
        trained_models=trained_models or None,
        deployment_config=deployment_config,
    )


def save_artifacts(result: PipelineResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.processed_data.to_csv(out / "processed_data.csv", index=False)
    result.windowed.X.to_csv(out / "window_features.csv", index=False)
    joblib.dump(result.feature_scaler, out / "feature_scaler.pkl")

    label_df = result.windowed.meta.copy()
    for model_name, model_labels in result.model_labels.items():
        label_df[f"regime_{model_name}"] = model_labels
    label_df.to_csv(out / "window_regimes.csv", index=False)

    with (out / "model_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(result.model_metrics, fp, indent=2)

    with (out / "quality_report.json").open("w", encoding="utf-8") as fp:
        json.dump(result.quality_report, fp, indent=2)

    if result.changepoints is not None:
        with (out / "changepoints.json").open("w", encoding="utf-8") as fp:
            json.dump(asdict(result.changepoints), fp, indent=2)

    if result.pca_projection is not None:
        result.pca_projection.to_csv(out / "pca_projection.csv", index=False)

    if result.diagnostics is not None:
        with (out / "model_diagnostics.json").open("w", encoding="utf-8") as fp:
            json.dump(result.diagnostics, fp, indent=2)

    if result.dense_autoencoder_config is not None:
        with (out / "dense_autoencoder_config.json").open("w", encoding="utf-8") as fp:
            json.dump(result.dense_autoencoder_config, fp, indent=2)

    if result.dense_latent_projection is not None:
        result.dense_latent_projection.to_csv(out / "dense_latent_projection.csv", index=False)

    if result.dense_autoencoder_state is not None:
        try:
            import torch

            torch.save(result.dense_autoencoder_state, out / "autoencoder_dense.pt")
        except Exception:
            joblib.dump(result.dense_autoencoder_state, out / "autoencoder_dense_state.joblib")

    if result.hierarchical_mapping is not None:
        joblib.dump(result.hierarchical_mapping, out / "macro_mapping.pkl")

    if result.macro_regime_characterization is not None:
        with (out / "macro_regime_characterization.json").open("w", encoding="utf-8") as fp:
            json.dump(result.macro_regime_characterization, fp, indent=2)

    if result.deployment_config is not None:
        with (out / "inference_config.json").open("w", encoding="utf-8") as fp:
            json.dump(result.deployment_config, fp, indent=2)

    if result.trained_models is not None and "hmm" in result.trained_models:
        joblib.dump(result.trained_models["hmm"], out / "hmm.pkl")

    joblib.dump(result.model_labels, out / "labels.joblib")

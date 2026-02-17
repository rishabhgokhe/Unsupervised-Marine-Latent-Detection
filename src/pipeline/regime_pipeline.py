from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core.config import ProjectConfig
from src.data.ingest import read_csv_dataset, resample_by_station
from src.data.preprocess import quality_preprocess, replace_special_missing
from src.evaluation.metrics import cluster_quality_scores
from src.evaluation.temporal_diagnostics import duration_statistics, label_transition_matrix
from src.features.window_engine import WindowedFeatures, generate_multiscale_window_features
from src.features.windows import build_sliding_windows
from src.models.changepoint import ChangePointResult, detect_changepoints
from src.models.clustering import ClusterRunOutput, run_clustering_baselines
from src.models.deep_autoencoder import run_lstm_autoencoder_segmentation
from src.models.hierarchy import build_hierarchical_regimes
from src.models.hmm_model import HMMResult, run_hmm
from src.models.label_utils import smooth_labels
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
        "hmm_bic_by_states": {},
    }

    for model_result in cluster_output.results:
        denoised = smooth_labels(model_result.labels, cfg.models.min_segment_length)
        labels[model_result.name] = denoised
        quality_score = cluster_quality_scores(cluster_output.transformed, denoised)
        durations = duration_statistics(denoised)
        diagnostics["duration_stats"][model_result.name] = durations
        diagnostics["transition_matrices"][model_result.name] = label_transition_matrix(denoised, int(model_result.n_states)).tolist()
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
        diagnostics["hmm_bic_by_states"] = {int(k): float(v) for k, v in hmm_result.bic_by_states.items()}
        metrics["hmm"] = {
            "n_states": float(hmm_result.n_states),
            "log_likelihood": float(hmm_result.log_likelihood),
            "bic": float(hmm_result.bic),
            "silhouette_post": float(hmm_quality.silhouette),
            "davies_bouldin": float(hmm_quality.davies_bouldin),
            "mean_regime_duration": float(hmm_durations["mean_duration"]),
        }
    else:
        logger.warning("hmmlearn not installed or HMM fit failed; skipping HMM")

    cp_result = detect_changepoints(cluster_output.transformed)
    if cp_result is not None:
        metrics["changepoint"] = {"n_breaks": float(cp_result.n_breaks)}

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
    for candidate in ("hmm", "vae_ablation", "deep_lstm_ae", "gmm", "kmeans"):
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
        },
        changepoints=cp_result,
        feature_scaler=cluster_output.scaler,
        pca_projection=pca_df,
        diagnostics=diagnostics,
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

    joblib.dump(result.model_labels, out / "labels.joblib")

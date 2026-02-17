from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from src.evaluation.metrics import boundary_precision_recall, labels_to_boundaries


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return -1.0
    return float(silhouette_score(X, labels))


def compute_db_index(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("inf")
    return float(davies_bouldin_score(X, labels))


def compute_hmm_bic(model: object, X: np.ndarray) -> float:
    logL = float(model.score(X))
    n_states = int(model.n_components)
    n_params = n_states**2 + 2 * n_states * X.shape[1]
    return float(-2 * logL + n_params * np.log(max(len(X), 2)))


def transition_entropy(matrix: np.ndarray, epsilon: float = 1e-12) -> float:
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return 0.0
    entropies = []
    for row in m:
        row = np.asarray(row, dtype=float)
        s = row.sum()
        if s <= 0:
            entropies.append(0.0)
            continue
        p = row / s
        p = np.clip(p, epsilon, 1.0)
        ent = -np.sum(p * np.log(p))
        entropies.append(float(ent))
    return float(np.mean(entropies)) if entropies else 0.0


def hmm_seed_stability(
    X: np.ndarray,
    n_states: int,
    seeds: Iterable[int],
    covariance_type: str = "diag",
    n_iter: int = 300,
) -> Optional[Dict]:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return None

    seed_labels: Dict[int, np.ndarray] = {}
    for seed in sorted(set(int(s) for s in seeds)):
        model = GaussianHMM(
            n_components=int(n_states),
            covariance_type=covariance_type,
            random_state=seed,
            n_iter=n_iter,
        )
        try:
            model.fit(X)
            seed_labels[seed] = model.predict(X)
        except Exception:
            continue

    if len(seed_labels) < 2:
        return None

    pairwise: Dict[str, float] = {}
    scores: List[float] = []
    for a, b in combinations(sorted(seed_labels.keys()), 2):
        ari = float(adjusted_rand_score(seed_labels[a], seed_labels[b]))
        pairwise[f"{a}_{b}"] = ari
        scores.append(ari)

    return {
        "seeds": sorted(seed_labels.keys()),
        "ari_pairwise": pairwise,
        "ari_mean": float(np.mean(scores)) if scores else 0.0,
        "ari_min": float(np.min(scores)) if scores else 0.0,
        "unique_states_by_seed": {int(s): int(len(np.unique(lbl))) for s, lbl in seed_labels.items()},
    }


def changepoint_regime_alignment(
    regime_labels: np.ndarray,
    changepoints: Iterable[int],
    tolerance: int,
) -> Dict[str, float]:
    boundaries = labels_to_boundaries(regime_labels)
    return boundary_precision_recall(changepoints, boundaries, tolerance=tolerance)


def regime_profiles(df_features: pd.DataFrame, regime_labels: np.ndarray) -> pd.DataFrame:
    out = df_features.copy().reset_index(drop=True)
    out["REGIME"] = np.asarray(regime_labels, dtype=int)
    numeric_cols = [c for c in out.columns if c != "REGIME" and pd.api.types.is_numeric_dtype(out[c])]
    return out.groupby("REGIME")[numeric_cols].mean().sort_index()


def pressure_drop_analysis(
    window_features: pd.DataFrame,
    regime_labels: np.ndarray,
    pressure_prefix: str = "SEA_LVL_PRES",
    lookback: int = 3,
) -> Dict[str, float]:
    cols = [c for c in window_features.columns if c.startswith(f"{pressure_prefix}_") and c.endswith("_mean")]
    if not cols:
        return {}

    pressure = window_features[sorted(cols)[0]].to_numpy(dtype=float)
    labels = np.asarray(regime_labels, dtype=int)
    transitions = np.where(labels[1:] != labels[:-1])[0] + 1
    if len(transitions) == 0:
        return {}

    drops = []
    for idx in transitions:
        start = max(0, idx - lookback)
        pre = pressure[start:idx]
        post = pressure[idx : min(len(pressure), idx + lookback)]
        if len(pre) == 0 or len(post) == 0:
            continue
        drops.append(float(np.mean(post) - np.mean(pre)))

    if not drops:
        return {}

    arr = np.asarray(drops, dtype=float)
    return {
        "n_transitions": float(len(transitions)),
        "mean_pressure_delta_post_minus_pre": float(np.mean(arr)),
        "median_pressure_delta_post_minus_pre": float(np.median(arr)),
        "p10_pressure_delta_post_minus_pre": float(np.percentile(arr, 10)),
    }


def summarize_state_sensitivity(
    kmeans_silhouette_by_k: Dict[int, float],
    gmm_bic_by_k: Dict[int, float],
    hmm_bic_by_k: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if kmeans_silhouette_by_k:
        best_km = max(kmeans_silhouette_by_k, key=kmeans_silhouette_by_k.get)
        summary["kmeans_best_k"] = float(best_km)
        summary["kmeans_best_silhouette"] = float(kmeans_silhouette_by_k[best_km])
    if gmm_bic_by_k:
        best_gmm = min(gmm_bic_by_k, key=gmm_bic_by_k.get)
        summary["gmm_best_k"] = float(best_gmm)
        summary["gmm_best_bic"] = float(gmm_bic_by_k[best_gmm])
    if hmm_bic_by_k:
        best_hmm = min(hmm_bic_by_k, key=hmm_bic_by_k.get)
        summary["hmm_best_k"] = float(best_hmm)
        summary["hmm_best_bic"] = float(hmm_bic_by_k[best_hmm])
    return summary

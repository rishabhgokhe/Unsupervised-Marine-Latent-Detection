from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    labels: np.ndarray
    n_states: int
    score_summary: Dict[str, float]


@dataclass
class ClusterRunOutput:
    scaler: StandardScaler
    transformed: np.ndarray
    results: List[ModelResult]


def _best_k_by_silhouette(x_scaled: np.ndarray, ks: List[int], random_state: int) -> tuple[int, float]:
    best_k, best_score = ks[0], -1.0
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(x_scaled)
        if len(np.unique(labels)) == 1:
            continue
        score = silhouette_score(x_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def run_clustering_baselines(
    X: pd.DataFrame,
    candidate_states: List[int],
    random_state: int,
) -> ClusterRunOutput:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X.values)

    best_k, best_k_sil = _best_k_by_silhouette(x_scaled, candidate_states, random_state)
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    km_labels = km.fit_predict(x_scaled)

    best_gmm = None
    best_bic = float("inf")
    gmm_summaries = {}
    for k in candidate_states:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gmm.fit(x_scaled)
        bic = gmm.bic(x_scaled)
        aic = gmm.aic(x_scaled)
        gmm_summaries[k] = {"bic": float(bic), "aic": float(aic)}
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    assert best_gmm is not None
    gmm_labels = best_gmm.predict(x_scaled)
    gmm_k = int(best_gmm.n_components)
    gmm_sil = silhouette_score(x_scaled, gmm_labels) if len(np.unique(gmm_labels)) > 1 else -1.0

    return ClusterRunOutput(
        scaler=scaler,
        transformed=x_scaled,
        results=[
            ModelResult(
                name="kmeans",
                labels=km_labels,
                n_states=int(best_k),
                score_summary={"silhouette": float(best_k_sil)},
            ),
            ModelResult(
                name="gmm",
                labels=gmm_labels,
                n_states=gmm_k,
                score_summary={
                    "silhouette": float(gmm_sil),
                    "bic": float(gmm_summaries[gmm_k]["bic"]),
                    "aic": float(gmm_summaries[gmm_k]["aic"]),
                },
            ),
        ],
    )

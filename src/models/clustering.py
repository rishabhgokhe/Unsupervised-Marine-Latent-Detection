from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.models.gmm_model import best_gmm_by_bic
from src.models.kmeans_model import best_kmeans_by_silhouette


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
    diagnostics: Dict[str, Dict[int, float]]


def run_clustering_baselines(
    X: pd.DataFrame,
    candidate_states: List[int],
    random_state: int,
) -> ClusterRunOutput:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X.values)

    _, km_labels, km_k, km_sil, km_sils = best_kmeans_by_silhouette(
        x_scaled,
        candidate_ks=candidate_states,
        random_state=random_state,
    )

    _, gmm_labels, gmm_k, gmm_best_bic, gmm_bics, gmm_aics = best_gmm_by_bic(
        x_scaled,
        candidate_components=candidate_states,
        random_state=random_state,
    )
    gmm_sil = silhouette_score(x_scaled, gmm_labels) if len(np.unique(gmm_labels)) > 1 else -1.0

    return ClusterRunOutput(
        scaler=scaler,
        transformed=x_scaled,
        results=[
            ModelResult(
                name="kmeans",
                labels=km_labels,
                n_states=int(km_k),
                score_summary={"silhouette": float(km_sil)},
            ),
            ModelResult(
                name="gmm",
                labels=gmm_labels,
                n_states=int(gmm_k),
                score_summary={
                    "silhouette": float(gmm_sil),
                    "bic": float(gmm_best_bic),
                    "aic": float(gmm_aics[gmm_k]),
                },
            ),
        ],
        diagnostics={
            "kmeans_silhouette_by_k": {int(k): float(v) for k, v in km_sils.items()},
            "gmm_bic_by_k": {int(k): float(v) for k, v in gmm_bics.items()},
            "gmm_aic_by_k": {int(k): float(v) for k, v in gmm_aics.items()},
        },
    )

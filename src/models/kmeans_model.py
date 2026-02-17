from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(X)
    return model, labels


def evaluate_kmeans(X: np.ndarray, candidate_ks: Iterable[int], random_state: int = 42) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for k in sorted(set(int(v) for v in candidate_ks if int(v) >= 2)):
        _, labels = train_kmeans(X, n_clusters=k, random_state=random_state)
        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
            continue
        scores[k] = float(silhouette_score(X, labels))
    return scores


def best_kmeans_by_silhouette(X: np.ndarray, candidate_ks: Iterable[int], random_state: int = 42) -> Tuple[KMeans, np.ndarray, int, float, Dict[int, float]]:
    scores = evaluate_kmeans(X, candidate_ks, random_state=random_state)
    if not scores:
        raise ValueError("No valid candidate k values for KMeans")
    best_k = max(scores, key=scores.get)
    model, labels = train_kmeans(X, n_clusters=best_k, random_state=random_state)
    return model, labels, int(best_k), float(scores[best_k]), scores

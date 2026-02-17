from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture


def train_gmm(X: np.ndarray, n_components: int, random_state: int = 42) -> tuple[GaussianMixture, np.ndarray]:
    model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    model.fit(X)
    labels = model.predict(X)
    return model, labels


def evaluate_gmm(X: np.ndarray, candidate_components: Iterable[int], random_state: int = 42) -> Tuple[Dict[int, float], Dict[int, float]]:
    bic_scores: Dict[int, float] = {}
    aic_scores: Dict[int, float] = {}
    for k in sorted(set(int(v) for v in candidate_components if int(v) >= 2)):
        model, _ = train_gmm(X, n_components=k, random_state=random_state)
        bic_scores[k] = float(model.bic(X))
        aic_scores[k] = float(model.aic(X))
    return bic_scores, aic_scores


def best_gmm_by_bic(X: np.ndarray, candidate_components: Iterable[int], random_state: int = 42) -> Tuple[GaussianMixture, np.ndarray, int, float, Dict[int, float], Dict[int, float]]:
    bic_scores, aic_scores = evaluate_gmm(X, candidate_components, random_state=random_state)
    if not bic_scores:
        raise ValueError("No valid candidate component values for GMM")
    best_k = min(bic_scores, key=bic_scores.get)
    model, labels = train_gmm(X, n_components=best_k, random_state=random_state)
    return model, labels, int(best_k), float(bic_scores[best_k]), bic_scores, aic_scores

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class HMMResult:
    labels: np.ndarray
    n_states: int
    log_likelihood: float
    bic: float
    transition_matrix: np.ndarray
    state_means: np.ndarray
    bic_by_states: Dict[int, float]


def _bic_approx(log_likelihood: float, n_states: int, n_features: int, n_samples: int) -> float:
    n_params = n_states * n_states + 2 * n_states * n_features
    return float(-2 * log_likelihood + n_params * np.log(max(n_samples, 2)))


def run_hmm(
    x_scaled: np.ndarray,
    candidate_states: List[int],
    covariance_type: str,
    random_state: int,
) -> Optional[HMMResult]:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return None

    best_model = None
    best_bic = float("inf")
    best_logl = -np.inf
    bic_by_states: Dict[int, float] = {}

    n_samples, n_features = x_scaled.shape
    for n in sorted(set(int(v) for v in candidate_states if int(v) >= 2)):
        model = GaussianHMM(
            n_components=n,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=300,
        )
        try:
            model.fit(x_scaled)
            logl = float(model.score(x_scaled))
            bic = _bic_approx(logl, n_states=n, n_features=n_features, n_samples=n_samples)
            bic_by_states[n] = bic
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_logl = logl
        except Exception:
            continue

    if best_model is None:
        return None

    labels = best_model.predict(x_scaled)
    trans = np.asarray(best_model.transmat_, dtype=float)
    return HMMResult(
        labels=labels,
        n_states=int(best_model.n_components),
        log_likelihood=float(best_logl),
        bic=float(best_bic),
        transition_matrix=trans,
        state_means=np.asarray(best_model.means_, dtype=float),
        bic_by_states=bic_by_states,
    )

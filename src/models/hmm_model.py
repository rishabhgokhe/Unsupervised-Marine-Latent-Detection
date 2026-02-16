from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class HMMResult:
    labels: np.ndarray
    n_states: int
    log_likelihood: float


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
    best_score = -np.inf
    for n in candidate_states:
        model = GaussianHMM(
            n_components=n,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=300,
        )
        try:
            model.fit(x_scaled)
            score = model.score(x_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return None

    labels = best_model.predict(x_scaled)
    return HMMResult(labels=labels, n_states=int(best_model.n_components), log_likelihood=float(best_score))

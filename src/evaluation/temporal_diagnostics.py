from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.models.label_utils import regime_durations


def duration_statistics(labels: np.ndarray) -> Dict[str, float]:
    durs = regime_durations(labels)
    if not durs:
        return {"n_segments": 0.0, "mean_duration": 0.0, "median_duration": 0.0, "p90_duration": 0.0, "min_duration": 0.0, "max_duration": 0.0}

    arr = np.asarray(durs, dtype=float)
    return {
        "n_segments": float(len(arr)),
        "mean_duration": float(np.mean(arr)),
        "median_duration": float(np.median(arr)),
        "p90_duration": float(np.percentile(arr, 90)),
        "min_duration": float(np.min(arr)),
        "max_duration": float(np.max(arr)),
    }


def label_transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    mat = np.zeros((n_states, n_states), dtype=float)
    if len(labels) < 2:
        return mat

    for i in range(1, len(labels)):
        a, b = int(labels[i - 1]), int(labels[i])
        if 0 <= a < n_states and 0 <= b < n_states:
            mat[a, b] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums

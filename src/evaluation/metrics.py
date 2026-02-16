from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


@dataclass
class ClusterQuality:
    silhouette: float
    davies_bouldin: float


def cluster_quality_scores(x_scaled: np.ndarray, labels: np.ndarray) -> ClusterQuality:
    if len(np.unique(labels)) < 2:
        return ClusterQuality(silhouette=-1.0, davies_bouldin=float("inf"))
    return ClusterQuality(
        silhouette=float(silhouette_score(x_scaled, labels)),
        davies_bouldin=float(davies_bouldin_score(x_scaled, labels)),
    )


def label_alignment_ari(reference_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    return float(adjusted_rand_score(reference_labels, predicted_labels))


def labels_to_boundaries(labels: Sequence[int]) -> List[int]:
    boundaries: List[int] = []
    prev = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != prev:
            boundaries.append(i)
        prev = labels[i]
    return boundaries


def boundary_precision_recall(
    true_boundaries: Iterable[int],
    pred_boundaries: Iterable[int],
    tolerance: int,
) -> Dict[str, float]:
    true_set = list(true_boundaries)
    pred_set = list(pred_boundaries)

    if not true_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_true = set()
    tp = 0
    for p in pred_set:
        for ti, t in enumerate(true_set):
            if ti in matched_true:
                continue
            if abs(p - t) <= tolerance:
                matched_true.add(ti)
                tp += 1
                break

    precision = tp / len(pred_set)
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

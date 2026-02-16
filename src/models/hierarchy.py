from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


@dataclass
class HierarchyResult:
    super_regimes: np.ndarray
    n_super_regimes: int
    mapping: Dict[int, int]


def build_hierarchical_regimes(
    x_scaled: np.ndarray,
    base_labels: np.ndarray,
    n_super_regimes: int = 2,
) -> Optional[HierarchyResult]:
    unique_states = np.unique(base_labels)
    if len(unique_states) < 2:
        return None

    centroids = []
    states = []
    for st in unique_states:
        mask = base_labels == st
        if mask.sum() == 0:
            continue
        centroids.append(x_scaled[mask].mean(axis=0))
        states.append(int(st))

    if len(states) < 2:
        return None

    n_clusters = min(n_super_regimes, len(states))
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    super_ids = agg.fit_predict(np.asarray(centroids))

    mapping = {states[i]: int(super_ids[i]) for i in range(len(states))}
    remapped = np.asarray([mapping[int(lbl)] for lbl in base_labels], dtype=int)

    return HierarchyResult(
        super_regimes=remapped,
        n_super_regimes=int(n_clusters),
        mapping=mapping,
    )

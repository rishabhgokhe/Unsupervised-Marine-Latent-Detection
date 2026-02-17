from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


@dataclass
class HierarchyResult:
    super_regimes: np.ndarray
    n_super_regimes: int
    mapping: Dict[int, int]


@dataclass
class HierarchicalLatentResult:
    micro_states: np.ndarray
    macro_states: np.ndarray
    macro_mapping: Dict[int, int]
    n_macro: int
    macro_transition_matrix: np.ndarray


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


def build_macro_regimes(state_means: np.ndarray, n_macro: int = 4, random_state: int = 42) -> np.ndarray:
    if state_means.ndim != 2 or len(state_means) < 2:
        raise ValueError("state_means must be 2D with at least 2 micro states")
    n_clusters = min(int(n_macro), int(len(state_means)))
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    return model.fit_predict(state_means).astype(int)


def map_micro_to_macro(micro_states: np.ndarray, macro_labels: np.ndarray) -> np.ndarray:
    micro_arr = np.asarray(micro_states, dtype=int)
    macro_arr = np.asarray(macro_labels, dtype=int)
    if micro_arr.size == 0:
        return np.asarray([], dtype=int)
    if macro_arr.ndim != 1:
        raise ValueError("macro_labels must be 1D")
    if micro_arr.min() < 0 or micro_arr.max() >= len(macro_arr):
        raise ValueError("micro state ids must index macro_labels")
    return macro_arr[micro_arr]


def transition_matrix(labels: np.ndarray) -> np.ndarray:
    labs = np.asarray(labels, dtype=int)
    if labs.size == 0:
        return np.zeros((0, 0), dtype=float)
    n = int(labs.max() + 1)
    mat = np.zeros((n, n), dtype=float)
    for i in range(len(labs) - 1):
        a, b = labs[i], labs[i + 1]
        if 0 <= a < n and 0 <= b < n:
            mat[a, b] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def characterize_regimes(df_features: pd.DataFrame, macro_sequence: np.ndarray) -> pd.DataFrame:
    out = df_features.copy().reset_index(drop=True)
    out["REGIME"] = np.asarray(macro_sequence, dtype=int)
    numeric_cols = [c for c in out.columns if c != "REGIME" and pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_cols:
        return pd.DataFrame(index=sorted(out["REGIME"].unique()))
    return out.groupby("REGIME")[numeric_cols].mean().sort_index()


def build_hierarchical_latent_states(
    micro_states: np.ndarray,
    state_means: np.ndarray,
    n_macro: int = 4,
    random_state: int = 42,
) -> Optional[HierarchicalLatentResult]:
    micro = np.asarray(micro_states, dtype=int)
    means = np.asarray(state_means, dtype=float)
    if micro.size == 0 or means.ndim != 2 or len(means) < 2:
        return None

    macro_labels = build_macro_regimes(means, n_macro=n_macro, random_state=random_state)
    macro_seq = map_micro_to_macro(micro, macro_labels)
    mapping = {int(i): int(macro_labels[i]) for i in range(len(macro_labels))}

    return HierarchicalLatentResult(
        micro_states=micro,
        macro_states=macro_seq,
        macro_mapping=mapping,
        n_macro=int(len(np.unique(macro_labels))),
        macro_transition_matrix=transition_matrix(macro_seq),
    )

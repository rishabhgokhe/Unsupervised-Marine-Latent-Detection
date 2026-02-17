import numpy as np
import pandas as pd

from src.evaluation.framework import (
    changepoint_regime_alignment,
    pressure_drop_analysis,
    summarize_state_sensitivity,
    transition_entropy,
)


def test_transition_entropy_lower_for_deterministic_transitions():
    near_deterministic = np.array([[0.95, 0.05], [0.02, 0.98]])
    diffuse = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert transition_entropy(near_deterministic) < transition_entropy(diffuse)


def test_pressure_drop_analysis_returns_expected_keys():
    x = pd.DataFrame({"SEA_LVL_PRES_6_mean": [1010, 1009, 1008, 1005, 1003, 1002]})
    labels = np.array([0, 0, 0, 1, 1, 1])
    out = pressure_drop_analysis(x, labels, lookback=2)
    assert "mean_pressure_delta_post_minus_pre" in out
    assert out["n_transitions"] >= 1


def test_changepoint_regime_alignment_basic():
    labels = np.array([0, 0, 1, 1, 2, 2])
    scores = changepoint_regime_alignment(labels, changepoints=[2, 4], tolerance=0)
    assert scores["f1"] == 1.0


def test_summarize_state_sensitivity_picks_best():
    summary = summarize_state_sensitivity(
        kmeans_silhouette_by_k={2: 0.2, 3: 0.4},
        gmm_bic_by_k={2: 100.0, 3: 90.0},
        hmm_bic_by_k={2: 120.0, 3: 80.0},
    )
    assert summary["kmeans_best_k"] == 3.0
    assert summary["gmm_best_k"] == 3.0
    assert summary["hmm_best_k"] == 3.0


import numpy as np

from src.evaluation.metrics import (
    boundary_precision_recall,
    cluster_quality_scores,
    labels_to_boundaries,
)


def test_labels_to_boundaries():
    labels = [0, 0, 1, 1, 2, 2]
    assert labels_to_boundaries(labels) == [2, 4]


def test_boundary_precision_recall_tolerance():
    scores = boundary_precision_recall(true_boundaries=[10, 20], pred_boundaries=[9, 23], tolerance=1)
    assert round(scores["precision"], 3) == 0.5
    assert round(scores["recall"], 3) == 0.5


def test_cluster_quality_scores_two_clusters():
    x = np.array([[0.0], [0.1], [10.0], [10.2]])
    labels = np.array([0, 0, 1, 1])
    q = cluster_quality_scores(x, labels)
    assert q.silhouette > 0.0

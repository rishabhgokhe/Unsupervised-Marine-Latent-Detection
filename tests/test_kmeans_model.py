import numpy as np

from src.models.kmeans_model import best_kmeans_by_silhouette, evaluate_kmeans


def test_evaluate_kmeans_skips_invalid_candidate_counts():
    x = np.array([[0.0], [1.0], [10.0]])

    scores = evaluate_kmeans(x, candidate_ks=[2, 3, 4], random_state=42)

    assert list(scores) == [2]


def test_best_kmeans_by_silhouette_on_separable_data():
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ]
    )

    _, labels, best_k, best_score, scores = best_kmeans_by_silhouette(
        x,
        candidate_ks=[2, 3],
        random_state=42,
    )

    assert best_k == 3
    assert best_score == scores[3]
    assert len(np.unique(labels)) == 3

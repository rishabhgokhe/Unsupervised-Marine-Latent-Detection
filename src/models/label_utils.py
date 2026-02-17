from __future__ import annotations

import numpy as np


def smooth_labels(labels: np.ndarray, min_duration: int = 3) -> np.ndarray:
    smoothed = labels.copy()
    if len(smoothed) == 0:
        return smoothed

    start = 0
    for i in range(1, len(smoothed) + 1):
        boundary = i == len(smoothed) or smoothed[i] != smoothed[start]
        if boundary:
            if (i - start) < min_duration:
                replacement = smoothed[start - 1] if start > 0 else (smoothed[i] if i < len(smoothed) else smoothed[start])
                smoothed[start:i] = replacement
            start = i
    return smoothed


def regime_durations(labels: np.ndarray) -> list[int]:
    if len(labels) == 0:
        return []

    durations: list[int] = []
    count = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            count += 1
        else:
            durations.append(count)
            count = 1
    durations.append(count)
    return durations

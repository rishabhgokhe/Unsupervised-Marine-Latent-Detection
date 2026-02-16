from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChangePointResult:
    break_indices: list[int]
    n_breaks: int


def detect_changepoints(x_scaled: np.ndarray, penalty: float = 8.0) -> Optional[ChangePointResult]:
    try:
        import ruptures as rpt
    except Exception:
        return None

    model = rpt.Pelt(model="rbf").fit(x_scaled)
    break_points = model.predict(pen=penalty)
    clean = sorted({int(i) for i in break_points if i < len(x_scaled)})
    return ChangePointResult(break_indices=clean, n_breaks=len(clean))

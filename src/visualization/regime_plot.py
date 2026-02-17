from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_regimes(time_values: np.ndarray | pd.Series, labels: np.ndarray, title: str = "Regime Segmentation"):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_values, alpha=0.5, color="black")

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i in range(len(labels)):
        color_idx = int(np.where(unique_labels == labels[i])[0][0])
        ax.axvspan(i, i + 1, alpha=0.12, color=cmap(color_idx))

    ax.set_title(title)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Signal")
    return fig

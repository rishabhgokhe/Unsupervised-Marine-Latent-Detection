from __future__ import annotations

import numpy as np

from src.models.train_autoencoder import get_latent_embeddings
from src.models.train_autoencoder import reconstruction_per_window


def run_inference(
    x_features_scaled: np.ndarray,
    ae_model: object | None,
    hmm_model: object,
    macro_mapping: dict[int, int] | None,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ae_model is not None:
        latent = get_latent_embeddings(ae_model, x_features_scaled, device=device)
        if latent is None:
            raise RuntimeError("Failed to compute latent embeddings")
        hmm_input = latent
    else:
        latent = x_features_scaled
        hmm_input = x_features_scaled

    micro_states = np.asarray(hmm_model.predict(hmm_input), dtype=int)
    if macro_mapping is None:
        macro_states = micro_states.copy()
    else:
        macro_states = np.asarray([macro_mapping.get(int(s), int(s)) for s in micro_states], dtype=int)
    return latent, micro_states, macro_states


def compute_reconstruction_errors(
    ae_model: object | None,
    x_features_scaled: np.ndarray,
    device: str | None = None,
) -> np.ndarray | None:
    if ae_model is None:
        return None
    errors = reconstruction_per_window(ae_model, x_features_scaled, device=device)
    if errors is None:
        return None
    return np.asarray(errors, dtype=float)

from __future__ import annotations

import numpy as np

from src.models.train_autoencoder import get_latent_embeddings


def run_inference(
    x_features_scaled: np.ndarray,
    ae_model: object,
    hmm_model: object,
    macro_mapping: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent = get_latent_embeddings(ae_model, x_features_scaled)
    if latent is None:
        raise RuntimeError("Failed to compute latent embeddings")

    micro_states = np.asarray(hmm_model.predict(latent), dtype=int)
    macro_states = np.asarray([macro_mapping.get(int(s), -1) for s in micro_states], dtype=int)
    return latent, micro_states, macro_states

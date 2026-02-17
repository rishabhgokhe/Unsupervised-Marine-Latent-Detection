from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import torch

from src.models.autoencoder_dense import DenseAutoencoder


@dataclass
class InferenceModels:
    scaler: object
    ae_model: DenseAutoencoder
    hmm_model: object
    macro_mapping: Dict[int, int]
    inference_config: Dict


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_macro_mapping(raw: object) -> Dict[int, int]:
    if isinstance(raw, dict) and "dense_ae_hmm" in raw and isinstance(raw["dense_ae_hmm"], dict):
        raw = raw["dense_ae_hmm"]
    if not isinstance(raw, dict):
        raise ValueError("macro_mapping.pkl must store a mapping dictionary")
    return {int(k): int(v) for k, v in raw.items()}


def load_models(artifacts_dir: str | Path) -> InferenceModels:
    art = Path(artifacts_dir)

    scaler = joblib.load(art / "feature_scaler.pkl")
    hmm_model = joblib.load(art / "hmm.pkl")
    macro_mapping = _normalize_macro_mapping(joblib.load(art / "macro_mapping.pkl"))

    dense_cfg = _load_json(art / "dense_autoencoder_config.json")
    inference_cfg = _load_json(art / "inference_config.json")

    input_dim = int(dense_cfg.get("input_dim", 0))
    latent_dim = int(dense_cfg.get("latent_dim", 32))
    if input_dim <= 0:
        raise ValueError("dense_autoencoder_config.json missing valid input_dim")

    ae_model = DenseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    state = torch.load(art / "autoencoder_dense.pt", map_location="cpu")
    ae_model.load_state_dict(state)
    ae_model.eval()

    return InferenceModels(
        scaler=scaler,
        ae_model=ae_model,
        hmm_model=hmm_model,
        macro_mapping=macro_mapping,
        inference_config=inference_cfg,
    )

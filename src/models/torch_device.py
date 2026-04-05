from __future__ import annotations

import os
from typing import Optional


def resolve_torch_device(device: Optional[str] = None):
    import torch

    if device is None or str(device).strip() == "":
        device = os.getenv("UMDL_DEVICE", "auto")

    device_norm = str(device).strip().lower()
    if device_norm in {"auto", "cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_norm == "cpu":
        return torch.device("cpu")

    return torch.device(device)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class DenseAEOutput:
    latent_embeddings: np.ndarray
    reconstruction_mse: float
    final_train_loss: float
    state_dict: Any
    latent_pca_2d: np.ndarray


def train_autoencoder(
    model: Any,
    X: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[Any, float] | tuple[None, None]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return None, None

    if len(X) < 32:
        return None, None

    torch.manual_seed(random_state)
    x_np = X.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = TensorDataset(torch.tensor(x_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    final_train_loss = 0.0
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_steps = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1
        final_train_loss = total_loss / max(n_steps, 1)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model, float(final_train_loss)


def get_latent_embeddings(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        import torch
    except Exception:
        return None

    x_np = X.astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        _, latent = model(x_tensor)
    return latent.cpu().numpy()


def reconstruction_error(model: Any, X: np.ndarray) -> Optional[float]:
    try:
        import torch
    except Exception:
        return None

    x_np = X.astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        recon, _ = model(x_tensor)
        mse = ((recon - x_tensor) ** 2).mean().item()
    return float(mse)


def visualize_latent(latent: np.ndarray) -> Optional[np.ndarray]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latent)
    plt.scatter(reduced[:, 0], reduced[:, 1], s=5)
    plt.title("Latent Space PCA")
    plt.show()
    return reduced


def save_autoencoder(model: Any, path: str | Path) -> bool:
    try:
        import torch
    except Exception:
        return False
    torch.save(model.state_dict(), str(path))
    return True


def train_dense_autoencoder(
    X: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    random_state: int = 42,
) -> Optional[DenseAEOutput]:
    try:
        from src.models.autoencoder_dense import DenseAutoencoder
    except Exception:
        return None

    model = DenseAutoencoder(input_dim=X.shape[1], latent_dim=latent_dim)
    trained_model, final_train_loss = train_autoencoder(
        model=model,
        X=X,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        random_state=random_state,
        verbose=False,
    )
    if trained_model is None or final_train_loss is None:
        return None

    latent_np = get_latent_embeddings(trained_model, X)
    reconstruction_mse = reconstruction_error(trained_model, X)
    if latent_np is None or reconstruction_mse is None:
        return None

    latent_scaled = StandardScaler().fit_transform(latent_np)
    pca = PCA(n_components=2, random_state=random_state)
    latent_pca = pca.fit_transform(latent_scaled)

    return DenseAEOutput(
        latent_embeddings=latent_np,
        reconstruction_mse=reconstruction_mse,
        final_train_loss=final_train_loss,
        state_dict=trained_model.state_dict(),
        latent_pca_2d=latent_pca,
    )

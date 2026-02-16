from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class VAEModelResult:
    labels: np.ndarray
    n_states: int
    recon_loss: float
    kl_loss: float
    silhouette: float


def run_vae_ablation(
    x_scaled: np.ndarray,
    candidate_states: List[int],
    random_state: int,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
) -> Optional[VAEModelResult]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return None

    x_np = x_scaled.astype(np.float32)
    if len(x_np) < 16:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class VAE(nn.Module):
        def __init__(self, input_dim: int, hidden: int, latent: int) -> None:
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.mu = nn.Linear(hidden, latent)
            self.logvar = nn.Linear(hidden, latent)
            self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.enc(x)
            return self.mu(h), self.logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.dec(z)
            return recon, mu, logvar

    model = VAE(input_dim=x_np.shape[1], hidden=hidden_dim, latent=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss(reduction="mean")

    loader = DataLoader(TensorDataset(torch.from_numpy(x_np)), batch_size=batch_size, shuffle=True)

    final_recon, final_kl = 0.0, 0.0
    model.train()
    for _ in range(epochs):
        total_recon, total_kl, count = 0.0, 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = mse(recon, batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            loss.backward()
            optimizer.step()
            total_recon += float(recon_loss.item())
            total_kl += float(kl.item())
            count += 1
        final_recon = total_recon / max(count, 1)
        final_kl = total_kl / max(count, 1)

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_np).to(device)
        mu, _ = model.encode(x_t)
        emb = mu.cpu().numpy()

    best_k = candidate_states[0]
    best_sil = -1.0
    best_labels = None
    for k in candidate_states:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(emb)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(emb, labels)
        if sil > best_sil:
            best_sil = float(sil)
            best_k = k
            best_labels = labels

    if best_labels is None:
        return None

    return VAEModelResult(
        labels=best_labels,
        n_states=int(best_k),
        recon_loss=float(final_recon),
        kl_loss=float(final_kl),
        silhouette=float(best_sil),
    )

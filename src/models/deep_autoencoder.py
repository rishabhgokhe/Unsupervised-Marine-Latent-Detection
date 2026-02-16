from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class DeepModelResult:
    labels: np.ndarray
    n_states: int
    train_loss: float
    silhouette: float


def _build_sequences(x: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if seq_len <= 1:
        idx = np.arange(len(x))
        return x[:, None, :], idx

    sequences = []
    end_indices = []
    for end in range(seq_len - 1, len(x)):
        start = end - seq_len + 1
        sequences.append(x[start : end + 1])
        end_indices.append(end)
    return np.asarray(sequences, dtype=np.float32), np.asarray(end_indices, dtype=np.int64)


def run_lstm_autoencoder_segmentation(
    x_scaled: np.ndarray,
    candidate_states: List[int],
    random_state: int,
    seq_len: int = 8,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> Optional[DeepModelResult]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return None

    x_seq, end_idx = _build_sequences(x_scaled.astype(np.float32), seq_len=seq_len)
    if len(x_seq) < max(16, seq_len * 2):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMAE(nn.Module):
        def __init__(self, input_dim: int, hidden: int, latent: int) -> None:
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder = nn.LSTM(input_dim, hidden, batch_first=True)
            self.output_proj = nn.Linear(hidden, input_dim)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            _, (h_n, _) = self.encoder(x)
            return self.to_latent(h_n[-1])

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encode(x)
            hidden = self.from_latent(z).unsqueeze(0)
            cell = torch.zeros_like(hidden)
            dec_in = torch.zeros_like(x)
            dec_out, _ = self.decoder(dec_in, (hidden, cell))
            rec = self.output_proj(dec_out)
            return rec, z

    input_dim = x_seq.shape[-1]
    model = LSTMAE(input_dim=input_dim, hidden=hidden_dim, latent=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(x_seq))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            rec, _ = model(batch)
            loss = criterion(rec, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            count += 1
        final_loss = epoch_loss / max(count, 1)

    model.eval()
    with torch.no_grad():
        emb = model.encode(torch.from_numpy(x_seq).to(device)).cpu().numpy()

    best_k = candidate_states[0]
    best_sil = -1.0
    best_labels = None
    for k in candidate_states:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        lbl = km.fit_predict(emb)
        if len(np.unique(lbl)) < 2:
            continue
        sil = silhouette_score(emb, lbl)
        if sil > best_sil:
            best_sil = float(sil)
            best_k = k
            best_labels = lbl

    if best_labels is None:
        return None

    out_labels = np.full(len(x_scaled), fill_value=-1, dtype=int)
    out_labels[end_idx] = best_labels

    valid = out_labels[out_labels >= 0]
    if len(valid) > 0:
        first = valid[0]
        for i in range(len(out_labels)):
            if out_labels[i] == -1:
                out_labels[i] = first
            else:
                break

    return DeepModelResult(
        labels=out_labels,
        n_states=int(best_k),
        train_loss=float(final_loss),
        silhouette=float(best_sil),
    )

"""Supervised training for betli hand evaluator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from trickster.betli.model import BetliNet, save_model


def train(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    save_path: Path | str = "models/betli/betli_eval.pt",
) -> BetliNet:
    """Train BetliNet on (features, labels) pairs.

    Labels should be in [-1, 1] range (MSE loss).
    Returns the trained model.
    """
    n = len(features)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    idx = np.random.permutation(n)
    features = features[idx]
    labels = labels[idx]

    X_train = torch.from_numpy(features[:n_train])
    y_train = torch.from_numpy(labels[:n_train])
    X_val = torch.from_numpy(features[n_train:])
    y_val = torch.from_numpy(labels[n_train:])

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    net = BetliNet(input_dim=features.shape[1])
    optimizer = Adam(net.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        net.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_dl:
            pred = net(xb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_n += len(xb)
        scheduler.step()
        train_loss /= train_n

        net.eval()
        with torch.no_grad():
            val_pred = net(X_val)
            val_loss = F.huber_loss(val_pred, y_val, delta=1.0).item()
            # Accuracy: sign agreement (both positive or both negative)
            val_acc = ((val_pred > 0) == (y_val > 0)).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}"
            )

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    save_model(net, save_path)
    print(f"  Saved to {save_path} (best val_loss={best_val_loss:.4f})")
    return net

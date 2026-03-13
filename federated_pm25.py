"""
Shared utilities for PM2.5 federated learning with Flower + FedProx local training.

Provides:
- Data loading from CSV splits (train/val/test) with per-client standardization
- MLP regressor model
- FedProx-aware local training loop
- Metric evaluation (MAE/MSE)
"""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TARGET_COL = "pm25"
TIMESTAMP_COL = "timestamp_utc"


@dataclass
class ClientDataSpec:
    client_id: str
    city: str
    sensor: str
    train_path: str
    val_path: str
    test_path: str


def list_client_specs(data_root: str) -> List[ClientDataSpec]:
    """Enumerate clients based on *_train.csv naming under city folders."""
    specs: List[ClientDataSpec] = []
    for city_dir in sorted(os.listdir(data_root)):
        city_path = os.path.join(data_root, city_dir)
        if not os.path.isdir(city_path):
            continue
        for train_path in glob.glob(os.path.join(city_path, "*_train.csv")):
            base = os.path.basename(train_path).replace("_train.csv", "")
            val_path = os.path.join(city_path, f"{base}_val.csv")
            test_path = os.path.join(city_path, f"{base}_test.csv")
            if not (os.path.exists(val_path) and os.path.exists(test_path)):
                continue
            parts = base.split("_")
            sensor = parts[-1] if parts else "unknown"
            specs.append(
                ClientDataSpec(
                    client_id=base,
                    city=city_dir,
                    sensor=sensor,
                    train_path=train_path,
                    val_path=val_path,
                    test_path=test_path,
                )
            )
    return specs


def load_split(path: str, feature_cols: List[str] | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (TARGET_COL, TIMESTAMP_COL)]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y, feature_cols


def make_datasets(
    train_path: str, val_path: str, test_path: str
) -> Tuple[Dict[str, TensorDataset], List[str], Dict[str, np.ndarray]]:
    X_train, y_train, feature_cols = load_split(train_path)
    X_val, y_val, _ = load_split(val_path, feature_cols)
    X_test, y_test, _ = load_split(test_path, feature_cols)

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std == 0] = 1.0

    def norm(x: np.ndarray) -> np.ndarray:
        return (x - x_mean) / x_std

    X_train = norm(X_train)
    X_val = norm(X_val)
    X_test = norm(X_test)

    def norm_y(y: np.ndarray) -> np.ndarray:
        return (y - y_mean) / y_std

    y_train = norm_y(y_train)
    y_val = norm_y(y_val)
    y_test = norm_y(y_test)

    datasets = {
        "train": TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        "val": TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        "test": TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    }
    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return datasets, feature_cols, stats


class PM25Regressor(nn.Module):
    """Configurable MLP regressor."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: List[nn.Module] = []
        in_dim = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            # Keep old behavior: dropout only once after first hidden layer by default.
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRURegressor(nn.Module):
    """Lightweight GRU regressor treating feature vector as length-1 sequence."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        head_hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(1)  # (batch, seq=1, feat)
        out, _ = self.gru(x_seq)
        last = out[:, -1, :]
        return self.head(last)


class TCNRegressor(nn.Module):
    """Simple 1D Conv/TCN-style regressor treating feature vector as a 1D signal."""

    def __init__(self, input_dim: int, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1d = x.unsqueeze(1)  # (batch, 1, length=input_dim)
        feats = self.net(x_1d)
        return self.head(feats)


def create_model(model_type: str, input_dim: int) -> nn.Module:
    mt = model_type.lower()
    if mt == "mlp":
        return PM25Regressor(input_dim)
    if mt == "mlp_large":
        return PM25Regressor(input_dim, hidden_dims=(256, 128, 64), dropout=0.1)
    if mt == "gru":
        return GRURegressor(input_dim)
    if mt == "gru_large":
        return GRURegressor(input_dim, hidden_size=256, num_layers=2, head_hidden=128, dropout=0.1)
    if mt == "tcn":
        return TCNRegressor(input_dim)
    raise ValueError(f"Unsupported model type: {model_type}")


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    mu: float,
    global_params: List[torch.Tensor],
    loss: str = "mse",
) -> float:
    """Local training with FedProx proximal term."""
    if loss == "mse":
        criterion = nn.MSELoss()
    elif loss == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=1.0)
    else:
        raise ValueError(f"Unsupported loss: {loss}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    global_params = [p.detach().to(device) for p in global_params]
    total_loss = 0.0

    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            if mu > 0:
                prox = 0.0
                for w, w0 in zip(model.parameters(), global_params):
                    prox += torch.sum((w - w0) ** 2)
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        total_loss = epoch_loss / len(loader.dataset)
    return total_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
) -> Dict[str, float]:
    model.eval()
    mse = 0.0
    mae = 0.0
    count = 0

    def denorm(y: torch.Tensor) -> torch.Tensor:
        if y_mean is None or y_std is None:
            return y
        y_m = torch.tensor(y_mean, device=y.device, dtype=y.dtype)
        y_s = torch.tensor(y_std, device=y.device, dtype=y.dtype)
        return y * y_s + y_m

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            preds = denorm(preds)
            yb = denorm(yb)
            diff = preds - yb
            mse += torch.sum(diff ** 2).item()
            mae += torch.sum(torch.abs(diff)).item()
            count += xb.size(0)
    return {"mse": mse / count, "mae": mae / count}

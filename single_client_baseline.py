"""
Single-client baseline trainer (no FL) to sanity-check model/feature quality.

Usage example:
    python single_client_baseline.py \
        --client-name client_Bhaktapur_90653 \
        --data-root processed \
        --epochs 20 \
        --batch-size 64 \
        --lr 5e-4

This trains only on one client's train split and reports MAE/MSE/RMSE on train/val/test.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from federated_pm25 import create_model, evaluate, make_datasets


def find_client_files(data_root: str, client_name: str) -> Tuple[str, str, str]:
    pattern = os.path.join(data_root, "**", f"{client_name}_train.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No train file found for {client_name} under {data_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple train files found for {client_name}: {matches}")
    train_path = matches[0]
    base = train_path.replace("_train.csv", "")
    val_path = f"{base}_val.csv"
    test_path = f"{base}_test.csv"
    for p in (val_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file missing: {p}")
    return train_path, val_path, test_path


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    loss_fn: nn.Module,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        total_loss = epoch_loss / len(loader.dataset)
    return total_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-client baseline training (no FL).")
    parser.add_argument("--client-name", required=True, help="Base name like client_City_Sensor (without split suffix).")
    parser.add_argument("--data-root", default="processed", help="Root folder containing processed data.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of local epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--model", choices=["mlp", "mlp_large", "gru", "gru_large", "tcn"], default="mlp", help="Model architecture.")
    parser.add_argument("--loss", choices=["mse", "smoothl1"], default="mse", help="Loss function.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_path, val_path, test_path = find_client_files(args.data_root, args.client_name)
    datasets, feature_cols, stats = make_datasets(train_path, val_path, test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, input_dim=len(feature_cols)).to(device)

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    loss_fn: nn.Module = nn.MSELoss() if args.loss == "mse" else nn.SmoothL1Loss(beta=1.0)
    train_loss = train_local(model, train_loader, device, epochs=args.epochs, lr=args.lr, loss_fn=loss_fn)

    train_metrics = evaluate(model, train_loader, device, y_mean=stats["y_mean"], y_std=stats["y_std"])
    val_metrics = evaluate(model, val_loader, device, y_mean=stats["y_mean"], y_std=stats["y_std"])
    test_metrics = evaluate(model, test_loader, device, y_mean=stats["y_mean"], y_std=stats["y_std"])

    # Add RMSE for readability.
    for m in (train_metrics, val_metrics, test_metrics):
        m["rmse"] = float(np.sqrt(m["mse"]))

    print(f"Trained on client {args.client_name}")
    print(f"Train loss ({args.loss}): {train_loss:.4f}")
    print(f"Train metrics: mse={train_metrics['mse']:.4f} rmse={train_metrics['rmse']:.4f} mae={train_metrics['mae']:.4f}")
    print(f"Val   metrics: mse={val_metrics['mse']:.4f} rmse={val_metrics['rmse']:.4f} mae={val_metrics['mae']:.4f}")
    print(f"Test  metrics: mse={test_metrics['mse']:.4f} rmse={test_metrics['rmse']:.4f} mae={test_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()

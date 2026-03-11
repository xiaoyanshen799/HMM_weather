"""
Local simulation of federated PM2.5 training with Flower + FedProx (for testing).

Use separate server/client scripts (fedprox_server.py, fedprox_client.py) for real
multi-machine runs. This file remains as a single-process simulator for quick
validation.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl

from federated_pm25 import (
    create_model,
    evaluate,
    get_parameters,
    list_client_specs,
    make_datasets,
    set_parameters,
    train_local,
)


class PM25Client(fl.client.NumPyClient):
    def __init__(
        self,
        spec: ClientDataSpec,
        datasets,
        feature_cols: List[str],
        y_mean,
        y_std,
        args: argparse.Namespace,
    ):
        self.spec = spec
        self.datasets = datasets
        self.feature_cols = feature_cols
        self.y_mean = y_mean
        self.y_std = y_std
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(args.model, input_dim=len(feature_cols)).to(self.device)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def fit(self, parameters, config=None):
        set_parameters(self.model, parameters)
        global_params = [p.detach().clone() for p in self.model.parameters()]
        train_loader = DataLoader(self.datasets["train"], batch_size=self.args.batch_size, shuffle=True)
        loss = train_local(
            self.model,
            train_loader,
            self.device,
            epochs=self.args.local_epochs,
            lr=self.args.lr,
            mu=self.args.mu,
            global_params=global_params,
            loss=self.args.loss,
        )
        metrics = {"train_loss": loss, "city": self.spec.city, "sensor": self.spec.sensor}
        return get_parameters(self.model), len(self.datasets["train"]), metrics

    def evaluate(self, parameters, config=None):
        set_parameters(self.model, parameters)
        val_loader = DataLoader(self.datasets["val"], batch_size=self.args.batch_size, shuffle=False)
        metrics = evaluate(self.model, val_loader, self.device, y_mean=self.y_mean, y_std=self.y_std)
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics.update({"city": self.spec.city, "sensor": self.spec.sensor})
        return float(metrics["mse"]), len(self.datasets["val"]), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower FedProx training for PM2.5 forecasting.")
    parser.add_argument("--data-root", default="processed", help="Root folder containing processed city data.")
    parser.add_argument("--num-rounds", type=int, default=30, help="Federated training rounds.")
    parser.add_argument("--clients-per-round", type=int, default=5, help="Number of clients sampled per round.")
    parser.add_argument("--local-epochs", type=int, default=10, help="Local epochs per round.")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Local learning rate.")
    parser.add_argument("--mu", type=float, default=0, help="FedProx mu coefficient.")
    parser.add_argument("--loss", choices=["mse", "smoothl1"], default="smoothl1", help="Local loss function.")
    parser.add_argument("--model", choices=["mlp", "gru", "tcn"], default="mlp", help="Model architecture.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    specs = list_client_specs(args.data_root)
    if not specs:
        raise RuntimeError(f"No clients found under {args.data_root}")

    # Preload datasets for each client spec.
    client_data: List[Tuple[ClientDataSpec, Dict[str, TensorDataset], List[str]]] = []
    for spec in specs:
        datasets, feature_cols, stats = make_datasets(spec)
        client_data.append((spec, datasets, feature_cols, stats["y_mean"], stats["y_std"]))

    def client_fn(cid: str):
        idx = int(cid)
        spec, datasets, feature_cols, y_mean, y_std = client_data[idx]
        return PM25Client(spec, datasets, feature_cols, y_mean, y_std, args)

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min(args.clients_per_round, len(specs)),
        min_available_clients=len(specs),
        min_evaluate_clients=min(args.clients_per_round, len(specs)),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(specs),
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

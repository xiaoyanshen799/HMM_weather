"""
Flower client for PM2.5 FedProx training.

Run one process per sensor/client on its machine. The client will:
- Load its train/val/test CSVs (standardized using train stats)
- Train locally with FedProx proximal term
- Report validation metrics to server

Example:
    python fedprox_client.py \
        --server-address 127.0.0.1:8080 \
        --data-root processed \
        --client-name client_Bhaktapur_90653 \
        --local-epochs 3 \
        --batch-size 64 \
        --lr 1e-3 \
        --mu 1e-3
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Sequence, Tuple

import flwr as fl
import grpc
import numpy as np
import torch
from torch.utils.data import DataLoader

from federated_pm25 import create_model, evaluate, get_parameters, make_datasets, set_parameters, train_local


def find_client_files(data_root: str, client_name: str):
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


class _ClientCallDetails(grpc.ClientCallDetails):
    def __init__(
        self,
        method: str,
        timeout: Optional[float],
        metadata: Optional[Sequence[Tuple[str, str]]],
        credentials,
        wait_for_ready: Optional[bool],
        compression,
    ) -> None:
        self.method = method
        self.timeout = timeout
        self.metadata = metadata
        self.credentials = credentials
        self.wait_for_ready = wait_for_ready
        self.compression = compression


class _HeaderInjector(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    def __init__(self, header: str, value: str) -> None:
        self._header = header
        self._value = value

    def _inject(self, client_call_details: grpc.ClientCallDetails) -> grpc.ClientCallDetails:
        metadata = list(client_call_details.metadata or [])
        metadata = [(key, val) for key, val in metadata if key.lower() != self._header]
        metadata.append((self._header, self._value))
        return _ClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=metadata,
            credentials=client_call_details.credentials,
            wait_for_ready=client_call_details.wait_for_ready,
            compression=client_call_details.compression,
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._inject(client_call_details), request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._inject(client_call_details), request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return continuation(self._inject(client_call_details), request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return continuation(self._inject(client_call_details), request_iterator)


def install_grpc_router_header(header: str, value: str) -> None:
    header_key = header.strip().lower()
    if not header_key:
        raise ValueError("router header name must be non-empty")
    if "\n" in value or "\r" in value:
        raise ValueError("router header value must not contain newlines")

    from flwr.common import grpc as flwr_grpc
    from flwr.client.grpc_client import connection as grpc_connection
    from flwr.client.grpc_rere_client import connection as grpc_rere_connection

    base_create_channel = flwr_grpc.create_channel
    injector = _HeaderInjector(header_key, value)

    def create_channel_with_header(*args, **kwargs):
        channel = base_create_channel(*args, **kwargs)
        return grpc.intercept_channel(channel, injector)

    flwr_grpc.create_channel = create_channel_with_header
    grpc_connection.create_channel = create_channel_with_header
    grpc_rere_connection.create_channel = create_channel_with_header


class PM25Client(fl.client.NumPyClient):
    def __init__(self, train_path: str, val_path: str, test_path: str, args: argparse.Namespace):
        datasets, feature_cols, stats = make_datasets(train_path, val_path, test_path)
        self.datasets = datasets
        self.feature_cols = feature_cols
        self.y_mean = stats["y_mean"]
        self.y_std = stats["y_std"]
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(args.model, input_dim=len(feature_cols)).to(self.device)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def fit(self, parameters, config=None):
        set_parameters(self.model, parameters)
        global_params: List[torch.Tensor] = [p.detach().clone() for p in self.model.parameters()]
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
        metrics = {"train_loss": loss}
        return get_parameters(self.model), len(self.datasets["train"]), metrics

    def evaluate(self, parameters, config=None):
        set_parameters(self.model, parameters)
        val_loader = DataLoader(self.datasets["val"], batch_size=self.args.batch_size, shuffle=False)
        metrics = evaluate(self.model, val_loader, self.device, y_mean=self.y_mean, y_std=self.y_std)
        # Add RMSE for convenience.
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        # Attach client identifier for server-side logging.
        metrics["client"] = self.args.client_name
        return float(metrics["mse"]), len(self.datasets["val"]), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower client for PM2.5 FedProx.")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server address host:port.")
    parser.add_argument("--data-root", default="processed", help="Root folder containing processed data.")
    parser.add_argument("--client-name", required=True, help="Base name like client_City_Sensor (without split suffix).")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local epochs per round.")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Local learning rate.")
    parser.add_argument("--mu", type=float, default=1e-3, help="FedProx mu coefficient.")
    parser.add_argument("--loss", choices=["mse", "smoothl1"], default="smoothl1", help="Local loss function.")
    parser.add_argument("--model", choices=["mlp", "gru", "tcn"], default="mlp", help="Model architecture.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--router-header",
        default=None,
        help="gRPC metadata header key for accelerator routing (default: router if router-value is set).",
    )
    parser.add_argument(
        "--router-value",
        default=None,
        help="gRPC metadata header value for accelerator routing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.router_value:
        install_grpc_router_header(args.router_header or "router", args.router_value)

    train_path, val_path, test_path = find_client_files(args.data_root, args.client_name)
    client = PM25Client(train_path, val_path, test_path, args)
    # Use the recommended API to avoid deprecation warnings.
    fl.client.start_client(server_address=args.server_address, client=client.to_client())


if __name__ == "__main__":
    main()

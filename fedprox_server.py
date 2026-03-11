"""
Flower server for PM2.5 FedProx training.

Run on the server machine, then start clients separately with fedprox_client.py.

Example:
    python fedprox_server.py --server-address 0.0.0.0:8080 --num-rounds 30 --clients-per-round 5
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server.strategy import FedAdam
from federated_pm25 import create_model, get_parameters, list_client_specs, make_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for PM2.5 FedProx.")
    parser.add_argument("--server-address", default="0.0.0.0:8080", help="gRPC server address host:port.")
    parser.add_argument("--num-rounds", type=int, default=30, help="Federated training rounds.")
    parser.add_argument("--clients-per-round", type=int, default=5, help="Clients sampled per round (min_fit_clients).")
    parser.add_argument("--min-available-clients", type=int, default=5, help="Minimum clients that must be connected.")
    parser.add_argument("--fraction-fit", type=float, default=0.0, help="Fraction of clients sampled each round; 0 means use clients-per-round.")
    parser.add_argument("--fraction-eval", type=float, default=0.0, help="Fraction of clients for evaluation; 0 uses clients-per-round.")
    parser.add_argument("--log-file", default="server.log", help="Path to server log file.")
    parser.add_argument("--strategy", choices=["fedavg", "fedadam"], default="fedavg", help="Federated optimization strategy.")
    parser.add_argument("--eta", type=float, default=0.01, help="Server learning rate for FedAdam.")
    parser.add_argument("--beta1", type=float, default=0.9, help="FedAdam beta1.")
    parser.add_argument("--beta2", type=float, default=0.99, help="FedAdam beta2.")
    parser.add_argument("--tau", type=float, default=1e-4, help="FedAdam tau (L2).")
    parser.add_argument("--data-root", default="processed", help="Root folder containing processed data (for building initial params).")
    parser.add_argument("--model", choices=["mlp", "gru", "tcn"], default="mlp", help="Model architecture for initial params.")
    # Optional logging of client-side hyperparameters (for bookkeeping only).
    parser.add_argument("--client-local-epochs", type=str, default=None, help="Client local epochs (for logging).")
    parser.add_argument("--client-lr", type=str, default=None, help="Client learning rate (for logging).")
    parser.add_argument("--client-batch-size", type=str, default=None, help="Client batch size (for logging).")
    parser.add_argument("--client-mu", type=str, default=None, help="Client mu (for logging).")
    parser.add_argument("--client-loss", type=str, default=None, help="Client loss (for logging).")
    parser.add_argument("--client-model", type=str, default=None, help="Client model (for logging).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Flower requires fraction_fit/fraction_evaluate to be floats, not None.
    fit_frac = args.fraction_fit if args.fraction_fit > 0 else args.clients_per_round / args.min_available_clients
    eval_frac = args.fraction_eval if args.fraction_eval > 0 else args.clients_per_round / args.min_available_clients

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    logging.info(
        "RUN_CONFIG strategy=%s server_model=%s data_root=%s num_rounds=%s clients_per_round=%s min_available=%s "
        "fraction_fit=%.4f fraction_eval=%.4f client_epochs=%s client_lr=%s client_batch=%s client_mu=%s client_loss=%s client_model=%s",
        args.strategy,
        args.model,
        args.data_root,
        args.num_rounds,
        args.clients_per_round,
        args.min_available_clients,
        fit_frac,
        eval_frac,
        args.client_local_epochs or "-",
        args.client_lr or "-",
        args.client_batch_size or "-",
        args.client_mu or "-",
        args.client_loss or "-",
        args.client_model or "-",
    )

    def build_initial_params():
        specs = list_client_specs(args.data_root)
        if not specs:
            logging.warning("No clients found under %s; cannot build initial parameters.", args.data_root)
            return None
        spec = specs[0]
        datasets, feature_cols, _ = make_datasets(spec.train_path, spec.val_path, spec.test_path)
        model = create_model(args.model, input_dim=len(feature_cols))
        return ndarrays_to_parameters(get_parameters(model))

    def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        total_examples = sum(num_examples for num_examples, _ in metrics)
        if total_examples == 0:
            return {}
        # Optional per-client logging if client id is provided.
        for num_examples, m in metrics:
            cid = m.get("client") if isinstance(m, dict) else None
            if cid:
                logging.info("Client %s metrics: mse=%.4f rmse=%.4f mae=%.4f n=%s", cid, m.get("mse"), m.get("rmse"), m.get("mae"), num_examples)
        mse = sum(num_examples * m["mse"] for num_examples, m in metrics) / total_examples
        mae = sum(num_examples * m["mae"] for num_examples, m in metrics) / total_examples
        rmse = math.sqrt(mse)
        logging.info("Round aggregated metrics -> mse: %.4f  rmse: %.4f  mae: %.4f", mse, rmse, mae)
        return {"mse": mse, "rmse": rmse, "mae": mae}

    if args.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=fit_frac,
            fraction_evaluate=eval_frac,
            min_fit_clients=args.clients_per_round,
            min_available_clients=args.min_available_clients,
            min_evaluate_clients=args.clients_per_round,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
        )
    else:
        init_params = build_initial_params()
        if init_params is None:
            logging.warning("Falling back to FedAvg because initial parameters could not be built.")
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=fit_frac,
                fraction_evaluate=eval_frac,
                min_fit_clients=args.clients_per_round,
                min_available_clients=args.min_available_clients,
                min_evaluate_clients=args.clients_per_round,
                evaluate_metrics_aggregation_fn=aggregate_metrics,
            )
        else:
            strategy = FedAdam(
                fraction_fit=fit_frac,
                fraction_evaluate=eval_frac,
                min_fit_clients=args.clients_per_round,
                min_available_clients=args.min_available_clients,
                min_evaluate_clients=args.clients_per_round,
                evaluate_metrics_aggregation_fn=aggregate_metrics,
                eta=args.eta,
                eta_l=None,
                beta_1=args.beta1,
                beta_2=args.beta2,
                tau=args.tau,
                initial_parameters=init_params,
            )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

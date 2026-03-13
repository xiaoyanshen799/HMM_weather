"""
Flower server for PM2.5 FedProx training.

Run on the server machine, then start clients separately with fedprox_client.py.

Example:
    python fedprox_server.py --server-address 0.0.0.0:8080 --num-rounds 30 --clients-per-round 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import Code, FitIns, FitRes, Metrics, Scalar, ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server
from flwr.server.strategy import FedAdam
from federated_pm25 import create_model, get_parameters, list_client_specs, make_datasets


def _to_float(metric_value) -> Optional[float]:
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return None


def timed_fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    group_id: int,
) -> Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    Dict[str, float],
]:
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    durations_by_cid: Dict[str, float] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted: Dict[concurrent.futures.Future, ClientProxy] = {}
        started_at: Dict[concurrent.futures.Future, float] = {}
        for client_proxy, ins in client_instructions:
            future = executor.submit(client_proxy.fit, ins, timeout=timeout, group_id=group_id)
            submitted[future] = client_proxy
            started_at[future] = time.perf_counter()

        for future in concurrent.futures.as_completed(submitted):
            client_proxy = submitted[future]
            durations_by_cid[client_proxy.cid] = time.perf_counter() - started_at[future]
            try:
                fit_res = future.result()
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)
                continue

            result = (client_proxy, fit_res)
            if fit_res.status.code == Code.OK:
                results.append(result)
            else:
                failures.append(result)

    return results, failures, durations_by_cid


class TimedServer(Server):
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ):
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            logging.info("configure_fit: no clients selected, cancel")
            return None
        logging.info(
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        results, failures, durations_by_cid = timed_fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        logging.info(
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            round_time_s = durations_by_cid.get(cid)
            compute_time_s = _to_float(fit_res.metrics.get("compute_time_s"))
            fit_time_s = _to_float(fit_res.metrics.get("fit_time_s"))
            if round_time_s is not None:
                fit_res.metrics["round_time_s"] = round_time_s
            logging.info(
                "Round %s client %s time: round_time_s=%s compute_time_s=%s fit_time_s=%s n=%s",
                server_round,
                cid,
                f"{round_time_s:.3f}" if round_time_s is not None else "-",
                f"{compute_time_s:.3f}" if compute_time_s is not None else "-",
                f"{fit_time_s:.3f}" if fit_time_s is not None else "-",
                fit_res.num_examples,
            )

        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        return parameters_aggregated, metrics_aggregated, (results, failures)


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
    parser.add_argument("--model", choices=["mlp", "mlp_large", "gru", "gru_large", "tcn"], default="mlp", help="Model architecture for initial params.")
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

    def aggregate_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        total_examples = sum(num_examples for num_examples, _ in metrics)
        if total_examples == 0:
            return {}

        total_train_loss = 0.0
        train_loss_examples = 0
        weighted_compute_time = 0.0
        compute_examples = 0
        weighted_fit_time = 0.0
        fit_examples = 0
        weighted_round_time = 0.0
        round_examples = 0

        for num_examples, m in metrics:
            cid = m.get("client") if isinstance(m, dict) else None
            train_loss = _to_float(m.get("train_loss")) if isinstance(m, dict) else None
            compute_time_s = _to_float(m.get("compute_time_s")) if isinstance(m, dict) else None
            fit_time_s = _to_float(m.get("fit_time_s")) if isinstance(m, dict) else None
            round_time_s = _to_float(m.get("round_time_s")) if isinstance(m, dict) else None

            if cid:
                logging.info(
                    "Client %s fit metrics: train_loss=%s compute_time_s=%s fit_time_s=%s round_time_s=%s n=%s",
                    cid,
                    f"{train_loss:.6f}" if train_loss is not None else "-",
                    f"{compute_time_s:.3f}" if compute_time_s is not None else "-",
                    f"{fit_time_s:.3f}" if fit_time_s is not None else "-",
                    f"{round_time_s:.3f}" if round_time_s is not None else "-",
                    num_examples,
                )

            if train_loss is not None:
                total_train_loss += num_examples * train_loss
                train_loss_examples += num_examples
            if compute_time_s is not None:
                weighted_compute_time += num_examples * compute_time_s
                compute_examples += num_examples
            if fit_time_s is not None:
                weighted_fit_time += num_examples * fit_time_s
                fit_examples += num_examples
            if round_time_s is not None:
                weighted_round_time += num_examples * round_time_s
                round_examples += num_examples

        aggregated: Dict[str, Scalar] = {}
        if train_loss_examples > 0:
            aggregated["train_loss"] = total_train_loss / train_loss_examples
        if compute_examples > 0:
            aggregated["compute_time_s"] = weighted_compute_time / compute_examples
        if fit_examples > 0:
            aggregated["fit_time_s"] = weighted_fit_time / fit_examples
        if round_examples > 0:
            aggregated["round_time_s"] = weighted_round_time / round_examples
        if aggregated:
            logging.info("Round aggregated fit metrics -> %s", aggregated)
        return aggregated

    if args.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=fit_frac,
            fraction_evaluate=eval_frac,
            min_fit_clients=args.clients_per_round,
            min_available_clients=args.min_available_clients,
            min_evaluate_clients=args.clients_per_round,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
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
                fit_metrics_aggregation_fn=aggregate_fit_metrics,
                evaluate_metrics_aggregation_fn=aggregate_metrics,
            )
        else:
            strategy = FedAdam(
                fraction_fit=fit_frac,
                fraction_evaluate=eval_frac,
                min_fit_clients=args.clients_per_round,
                min_available_clients=args.min_available_clients,
                min_evaluate_clients=args.clients_per_round,
                fit_metrics_aggregation_fn=aggregate_fit_metrics,
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
        server=TimedServer(client_manager=SimpleClientManager(), strategy=strategy),
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )


if __name__ == "__main__":
    main()

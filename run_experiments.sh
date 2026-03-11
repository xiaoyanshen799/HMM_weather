#!/usr/bin/env bash
# Simple experiment runner to sweep a few hyperparameter combos.
# It starts the server per config, then launches all local clients, waits for completion,
# and saves the server log to logs/run_<tag>.log.
#
# Adjust SERVER_ADDRESS/ROUNDS/CLIENTS_PER_ROUND/MIN_AVAILABLE as needed.
# Ensure no old server/clients are running before starting (stop with pkill -f fedprox_server.py/fedprox_client.py).

set -euo pipefail

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8081}"
DATA_ROOT="${DATA_ROOT:-processed}"
ROUNDS="${ROUNDS:-30}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-20}"
MIN_AVAILABLE="${MIN_AVAILABLE:-20}"
STRATEGY="${STRATEGY:-fedavg}"  # fedavg or fedadam

# Hyper grids
LRS=(5e-4)
MUS=(1e-3)
LOCAL_EPOCHS=(3 5)
MODELS=(mlp)
LOSS="smoothl1"
BATCH_SIZE=64

mkdir -p logs

run_idx=0
for lr in "${LRS[@]}"; do
  for mu in "${MUS[@]}"; do
    for le in "${LOCAL_EPOCHS[@]}"; do
      for model in "${MODELS[@]}"; do
        run_idx=$((run_idx + 1))
        tag="r${run_idx}_strat${STRATEGY}_lr${lr}_mu${mu}_ep${le}_model${model}"
        log_file="logs/${tag}.log"

        echo "=== Running ${tag} ==="
        # Start server
        python fedprox_server.py \
          --server-address "${SERVER_ADDRESS}" \
          --strategy "${STRATEGY}" \
          --data-root "${DATA_ROOT}" \
          --model "${model}" \
          --num-rounds "${ROUNDS}" \
          --clients-per-round "${CLIENTS_PER_ROUND}" \
          --min-available-clients "${MIN_AVAILABLE}" \
          --log-file "${log_file}" \
          --client-local-epochs "${le}" \
          --client-lr "${lr}" \
          --client-batch-size "${BATCH_SIZE}" \
          --client-mu "${mu}" \
          --client-loss "${LOSS}" \
          --client-model "${model}" &
        SERVER_PID=$!
        sleep 2

        # Launch all clients with matching hyperparams
        LOCAL_EPOCHS="${le}" LR="${lr}" MU="${mu}" LOSS="${LOSS}" MODEL="${model}" BATCH_SIZE="${BATCH_SIZE}" SERVER_ADDRESS="${SERVER_ADDRESS}" ./start_all_clients.sh

        # Wait for server to finish
        wait "${SERVER_PID}"
        echo "Finished ${tag}. Log: ${log_file}"
      done
    done
  done
done

echo "All experiments completed. Logs in logs/ directory."

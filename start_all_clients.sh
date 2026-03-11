#!/usr/bin/env bash
# Start all local Flower FedProx clients found under a data root.
# By default, enumerates processed/<city>/client_*_train.csv and launches one client per file.
#
# Usage:
#   ./start_all_clients.sh [data_root]
# Environment overrides:
#   SERVER_ADDRESS (default 127.0.0.1:8081)
#   JOBS           (default 0 = no limit; set to N to cap concurrent clients; set to 1 for sequential)
#   LOCAL_EPOCHS, BATCH_SIZE, LR, MU, LOSS, SEED, PYTHON_BIN passed through to start_client.sh
#
# Note: Run the Flower server separately (fedprox_server.py) before launching clients.

set -euo pipefail

DATA_ROOT="${1:-${DATA_ROOT:-processed}}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8081}"
# Default: no concurrency limit; each client remains single-threaded internally.
JOBS="${JOBS:-0}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root not found: ${DATA_ROOT}" >&2
  exit 1
fi

clients=()
while IFS= read -r train_path; do
  base="$(basename "${train_path}" _train.csv)"
  clients+=("${base}")
done < <(find "${DATA_ROOT}" -name '*_train.csv' -print | sort)

if [[ ${#clients[@]} -eq 0 ]]; then
  echo "No clients found under ${DATA_ROOT}" >&2
  exit 1
fi

echo "Server address: ${SERVER_ADDRESS}"
echo "Data root: ${DATA_ROOT}"
echo "Found ${#clients[@]} clients. Launching..."

launch_count=0
for client in "${clients[@]}"; do
  echo "-> Starting ${client}"
  SERVER_ADDRESS="${SERVER_ADDRESS}" DATA_ROOT="${DATA_ROOT}" ./start_client.sh "${client}" "${SERVER_ADDRESS}" "${DATA_ROOT}" &
  launch_count=$((launch_count + 1))

  if [[ "${JOBS}" -gt 0 ]]; then
    # Throttle to JOBS concurrent processes.
    while [[ "$(jobs -pr | wc -l)" -ge "${JOBS}" ]]; do
      sleep 0.5
    done
  fi
done

echo "Launched ${launch_count} clients. Waiting for them to finish (Ctrl+C to stop)."
wait

#!/usr/bin/env bash
# Convenience launcher for a single Flower FedProx client.
# Usage:
#   ./start_client.sh <client_name> [server_address] [data_root]
# Examples:
#   ./start_client.sh client_Bhaktapur_90653 10.0.0.1:8080 processed
#   SERVER_ADDRESS=10.0.0.1:8090 DATA_ROOT=/data/processed ./start_client.sh client_London_12345

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <client_name> [server_address] [data_root]" >&2
  exit 1
fi

CLIENT_NAME="$1"
SERVER_ADDRESS="${2:-${SERVER_ADDRESS:-127.0.0.1:8081}}"
DATA_ROOT="${3:-${DATA_ROOT:-processed}}"

LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
MU="${MU:-1e-3}"
LOSS="${LOSS:-smoothl1}"
MODEL="${MODEL:-gru}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# ROUTER_VALUE="${ROUTER_VALUE:-{"routing":[{"hops":"34.174.125.203:8095,127.0.0.1:8081"}]}}"
# Force single-threaded math for consistent local training (override via env if needed).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "Starting client ${CLIENT_NAME}"
echo "Server address: ${SERVER_ADDRESS}"
echo "Data root: ${DATA_ROOT}"
echo "Local epochs: ${LOCAL_EPOCHS}, batch size: ${BATCH_SIZE}, lr: ${LR}, mu: ${MU}, loss: ${LOSS}, model: ${MODEL}, seed: ${SEED}"

exec "${PYTHON_BIN}" fedprox_client.py \
  --server-address "${SERVER_ADDRESS}" \
  --data-root "${DATA_ROOT}" \
  --client-name "${CLIENT_NAME}" \
  --local-epochs "${LOCAL_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --mu "${MU}" \
  --model "${MODEL}" \
  --loss "${LOSS}" \
  --seed "${SEED}"\
  # --router-value "${ROUTER_VALUE}"

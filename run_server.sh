#!/usr/bin/env bash
# Launch the PAAS_ensemble_v3_inf FastAPI liveness service.
#
#   bash run_server.sh                 # default: 5-detector mean (config/experiments/paas5_mean.json)
#   PAAS_CONFIG=config/experiments/paas4_fast.json bash run_server.sh   # fast, no 7B MLLM
#   PAAS_COMPONENTS="gsd,selop" PAAS_THRESHOLD=0.15 bash run_server.sh   # ad-hoc subset/threshold
set -euo pipefail
cd "$(dirname "$0")"

export PAAS_CONFIG="${PAAS_CONFIG:-config/experiments/paas5_mean.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"     # keep 1: each worker loads the full model set onto the GPU

echo "[run_server] PAAS_CONFIG=$PAAS_CONFIG  GPU=$CUDA_VISIBLE_DEVICES  ${HOST}:${PORT}"
exec python3.12 -m uvicorn app_fastapi_json_v5_safe_jsonfmt:app \
    --host "$HOST" --port "$PORT" --workers "$WORKERS"

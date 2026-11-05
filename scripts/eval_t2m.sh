#!/usr/bin/env bash
# T2M -> T2M: the FlowMDM checkpoint trained without history text, evaluated with plain
# per-segment captions on ELMA test_anaphora.
#   DATA_ROOT=/path/to/ELMA/HML3D GPU_ID=0 EVAL_MODE=final bash scripts/eval_t2m.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the extracted ELMA HML3D directory}"
GPU_ID="${GPU_ID:-0}"
EVAL_MODE="${EVAL_MODE:-final}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

cd "${ROOT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" MPLBACKEND=Agg "${PYTHON_BIN}" -m elmabench.evaluate \
  --flowmdm_dir "${ROOT_DIR}/FlowMDM" \
  --model_path "${ROOT_DIR}/FlowMDM/results/humanml/ELMAv3_anap_hml3d/model000550000.pt" \
  --data_root "${DATA_ROOT}" \
  --eval_mode "${EVAL_MODE}" \
  --guidance_param 2.5 \
  --device 0 \
  "$@"

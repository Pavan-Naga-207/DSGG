#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEST_ONLY=0

usage() {
  echo "Usage: $0 [--test-only]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-only)
      TEST_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

DATA_PATH="${DATA_PATH:-$HOME/special_topics/datasets/action_genome}"
MODE="${MODE:-sgdet}"
DATASIZE="${DATASIZE:-large}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/sttran_${MODE}_$(date +%Y%m%d_%H%M%S)}"
NEPOCH="${NEPOCH:-10}"
LR="${LR:-1e-5}"
ENC_LAYER="${ENC_LAYER:-1}"
DEC_LAYER="${DEC_LAYER:-3}"
OPTIMIZER="${OPTIMIZER:-adamw}"
BCE_LOSS="${BCE_LOSS:-0}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.7}"

SBATCH_ARGS=(
  --export="ALL,DATA_PATH=${DATA_PATH},MODE=${MODE},DATASIZE=${DATASIZE},SAVE_PATH=${SAVE_PATH},NEPOCH=${NEPOCH},LR=${LR},ENC_LAYER=${ENC_LAYER},DEC_LAYER=${DEC_LAYER},OPTIMIZER=${OPTIMIZER},BCE_LOSS=${BCE_LOSS},VENV_PATH=${VENV_PATH},PYTHON_MODULE=${PYTHON_MODULE}"
  "scripts/train_sttran_h100.sbatch"
)

if [[ "${TEST_ONLY}" == "1" ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}"
else
  sbatch "${SBATCH_ARGS[@]}"
fi

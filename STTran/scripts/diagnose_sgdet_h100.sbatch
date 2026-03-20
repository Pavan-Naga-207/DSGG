#!/bin/bash
#SBATCH --partition=nsfqs
#SBATCH --job-name=sttran_diag
#SBATCH --output=logs/sttran_diag_%j.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/diagnose_sgdet_pipeline.py" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diagnose_sgdet_pipeline.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${PWD}/diagnose_sgdet_pipeline.py" ]]; then
  PROJECT_ROOT="${PWD}"
elif [[ -f "${SCRIPT_DIR}/../diagnose_sgdet_pipeline.py" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "Could not determine PROJECT_ROOT containing diagnose_sgdet_pipeline.py"
  exit 2
fi

PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.5}"
PYTHON_MODULE_FALLBACK="${PYTHON_MODULE_FALLBACK:-python3/3.11.7}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"

DATA_PATH="${DATA_PATH:-$HOME/special_topics/datasets/action_genome}"
MODEL_PATH="${MODEL_PATH:-}"
DATASIZE="${DATASIZE:-large}"
BACKBONE="${BACKBONE:-vitdet}"
MAX_DIAG_STEPS="${MAX_DIAG_STEPS:-25}"
DIAG_PRINT_EVERY="${DIAG_PRINT_EVERY:-5}"
DIAG_SCORE_THRESH="${DIAG_SCORE_THRESH:-0.1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-1}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required. Example:"
  echo "  sbatch --export=ALL,MODEL_PATH=/path/to/sttran_epoch_9.pth scripts/diagnose_sgdet_h100.sbatch"
  exit 3
fi

module purge
if ! module load "${PYTHON_MODULE}"; then
  echo "Requested module unavailable: ${PYTHON_MODULE}"
  echo "Trying fallback module: ${PYTHON_MODULE_FALLBACK}"
  module load "${PYTHON_MODULE_FALLBACK}"
  PYTHON_MODULE="${PYTHON_MODULE_FALLBACK}"
fi

MODULE_BIN_DIR="$(dirname "$(command -v python3)")"
if [[ -f "${MODULE_BIN_DIR}/activate" ]]; then
  # shellcheck disable=SC1090
  source "${MODULE_BIN_DIR}/activate"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual env missing: ${VENV_PATH}"
  echo "Run scripts/setup_sttran_env.sh first."
  exit 4
fi
source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Checkpoint not found: ${MODEL_PATH}"
  exit 5
fi

if [[ ! -f "${DATA_PATH}/annotations/object_bbox_and_relationship.pkl" ]]; then
  echo "Missing Action Genome annotations in ${DATA_PATH}/annotations"
  exit 6
fi

if [[ ! -d "${DATA_PATH}/frames" ]]; then
  echo "Missing frame directory: ${DATA_PATH}/frames"
  exit 7
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export STTRAN_OFFLINE_WORDVEC="${STTRAN_OFFLINE_WORDVEC:-1}"
export STTRAN_WV_DIR="${STTRAN_WV_DIR:-$PROJECT_ROOT/data}"
export VITDET_FREEZE="${VITDET_FREEZE:-1}"

echo "Running on host: $(hostname)"
echo "CUDA device(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Data: ${DATA_PATH}"
echo "Model: ${MODEL_PATH}"
echo "Project root: ${PROJECT_ROOT}"
echo "Backbone: ${BACKBONE}"
echo "Python module: ${PYTHON_MODULE}"
echo "Max diagnostic steps: ${MAX_DIAG_STEPS}"
echo "Diagnostic score threshold: ${DIAG_SCORE_THRESH}"
echo "VITDET_FREEZE: ${VITDET_FREEZE}"

EXTRA_ARGS=()
if [[ "${PIN_MEMORY}" == "1" ]]; then
  EXTRA_ARGS+=(--pin-memory)
fi

python -u diagnose_sgdet_pipeline.py \
  --data_path "${DATA_PATH}" \
  --model_path "${MODEL_PATH}" \
  --datasize "${DATASIZE}" \
  --backbone "${BACKBONE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --max-steps "${MAX_DIAG_STEPS}" \
  --print-every "${DIAG_PRINT_EVERY}" \
  --det-score-thresh "${DIAG_SCORE_THRESH}" \
  "${EXTRA_ARGS[@]}"

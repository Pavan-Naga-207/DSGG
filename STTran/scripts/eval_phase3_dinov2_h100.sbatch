#!/bin/bash
#SBATCH --partition=condo
#SBATCH --job-name=sttran_p3_eval
#SBATCH --output=logs/sttran_p3_eval_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/evaluate_only_phase3.py" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/evaluate_only_phase3.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.5}"
PYTHON_MODULE_FALLBACK="${PYTHON_MODULE_FALLBACK:-python3/3.11.7}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"

DATA_PATH="${DATA_PATH:-$HOME/special_topics/datasets/action_genome}"
MODE="${MODE:-sgdet}"
DATASIZE="${DATASIZE:-large}"
MODEL_PATH="${MODEL_PATH:-}"
ENC_LAYER="${ENC_LAYER:-1}"
DEC_LAYER="${DEC_LAYER:-3}"
BACKBONE="${BACKBONE:-dinov2}"
DET_THRESHOLD="${DET_THRESHOLD:-0.1}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required"
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
  source "${MODULE_BIN_DIR}/activate"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual env missing: ${VENV_PATH}"
  exit 4
fi
source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export STTRAN_OFFLINE_WORDVEC="${STTRAN_OFFLINE_WORDVEC:-1}"
export STTRAN_WV_DIR="${STTRAN_WV_DIR:-$PROJECT_ROOT/data}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export EVAL_CONSTRAINTS="${EVAL_CONSTRAINTS:-with,semi,no}"
export MAX_TEST_STEPS="${MAX_TEST_STEPS:--1}"
export DETECTOR_BN_MODE="${DETECTOR_BN_MODE:-groupnorm}"
export DETECTOR_GN_GROUPS="${DETECTOR_GN_GROUPS:-32}"
export PHASE3_DINOV2_MODEL="${PHASE3_DINOV2_MODEL:-vit_base_patch14_dinov2}"
export PHASE3_FREEZE_BACKBONE="${PHASE3_FREEZE_BACKBONE:-1}"
export PHASE3_INPUT_SIZE="${PHASE3_INPUT_SIZE:-1024}"
export PHASE3_DYNAMIC_IMG_SIZE="${PHASE3_DYNAMIC_IMG_SIZE:-1}"
export PHASE3_DYNAMIC_IMG_PAD="${PHASE3_DYNAMIC_IMG_PAD:-1}"
export PHASE3_DETECTOR_STRIDE="${PHASE3_DETECTOR_STRIDE:-16}"
export PHASE3_OUT_CHANNELS="${PHASE3_OUT_CHANNELS:-256}"
export PHASE3_ALIGN_CHANNELS="${PHASE3_ALIGN_CHANNELS:-1024}"
export PHASE3_ASSERT_DINOV2_PATH="${PHASE3_ASSERT_DINOV2_PATH:-1}"
export SGCLS_DUPLICATE_POLICY="${SGCLS_DUPLICATE_POLICY:-iou}"
export SGCLS_LABEL_SOURCE="${SGCLS_LABEL_SOURCE:-detector}"

python -u evaluate_only_phase3.py \
  -mode "${MODE}" \
  -datasize "${DATASIZE}" \
  -data_path "${DATA_PATH}" \
  -model_path "${MODEL_PATH}" \
  --backbone "${BACKBONE}" \
  --det_threshold "${DET_THRESHOLD}" \
  -enc_layer "${ENC_LAYER}" \
  -dec_layer "${DEC_LAYER}"

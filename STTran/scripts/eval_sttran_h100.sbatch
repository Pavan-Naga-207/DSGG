#!/bin/bash
#SBATCH --partition=nsfqs
#SBATCH --job-name=sttran_eval
#SBATCH --output=logs/sttran_eval_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/evaluate_only.py" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/evaluate_only.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${PWD}/evaluate_only.py" ]]; then
  PROJECT_ROOT="${PWD}"
elif [[ -f "${SCRIPT_DIR}/../evaluate_only.py" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "Could not determine PROJECT_ROOT containing evaluate_only.py"
  echo "PWD=${PWD}"
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
  echo "SCRIPT_DIR=${SCRIPT_DIR}"
  exit 2
fi

PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.7}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"

DATA_PATH="${DATA_PATH:-$HOME/special_topics/datasets/action_genome}"
MODE="${MODE:-sgdet}"
DATASIZE="${DATASIZE:-large}"
MODEL_PATH="${MODEL_PATH:-}"
ENC_LAYER="${ENC_LAYER:-1}"
DEC_LAYER="${DEC_LAYER:-3}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required, e.g.:"
  echo "  sbatch --export=ALL,MODEL_PATH=/path/to/sttran_epoch_2.pth scripts/eval_sttran_h100.sbatch"
  exit 3
fi

module purge
module load "${PYTHON_MODULE}"

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

if [[ ! -f "fasterRCNN/models/faster_rcnn_ag.pth" ]]; then
  echo "Missing detector checkpoint: fasterRCNN/models/faster_rcnn_ag.pth"
  echo "Run scripts/download_sttran_assets.sh first."
  exit 6
fi

if [[ ! -f "dataloader/object_bbox_and_relationship_filtersmall.pkl" ]]; then
  echo "Missing dataloader/object_bbox_and_relationship_filtersmall.pkl"
  echo "Run scripts/download_sttran_assets.sh first."
  exit 7
fi

if [[ ! -f "${DATA_PATH}/annotations/object_bbox_and_relationship.pkl" ]]; then
  echo "Missing Action Genome annotations in ${DATA_PATH}/annotations"
  exit 8
fi

if [[ ! -d "${DATA_PATH}/frames" ]]; then
  echo "Missing frame directory: ${DATA_PATH}/frames"
  exit 9
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export STTRAN_OFFLINE_WORDVEC="${STTRAN_OFFLINE_WORDVEC:-1}"
export STTRAN_WV_DIR="${STTRAN_WV_DIR:-$PROJECT_ROOT/data}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export EVAL_CONSTRAINTS="${EVAL_CONSTRAINTS:-with,semi,no}"
export MAX_TEST_STEPS="${MAX_TEST_STEPS:--1}"

echo "Running on host: $(hostname)"
echo "CUDA device(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Mode: ${MODE}"
echo "Data: ${DATA_PATH}"
echo "Model: ${MODEL_PATH}"
echo "Project root: ${PROJECT_ROOT}"
echo "Eval constraints: ${EVAL_CONSTRAINTS}"
echo "Eval workers: ${EVAL_NUM_WORKERS}"

python -u evaluate_only.py \
  -mode "${MODE}" \
  -datasize "${DATASIZE}" \
  -data_path "${DATA_PATH}" \
  -model_path "${MODEL_PATH}" \
  -enc_layer "${ENC_LAYER}" \
  -dec_layer "${DEC_LAYER}"

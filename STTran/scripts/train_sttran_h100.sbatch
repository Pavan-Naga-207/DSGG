#!/bin/bash
#SBATCH --partition=nsfqs
#SBATCH --job-name=sttran_h100
#SBATCH --output=logs/sttran_h100_%j.log
#SBATCH --time=2-00:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# When submitted via sbatch, BASH_SOURCE can point to Slurm spool copy.
# Prefer submit dir if it is the STTran project root.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train.py" ]]; then
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
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/sttran_${MODE}_$(date +%Y%m%d_%H%M%S)}"
NEPOCH="${NEPOCH:-10}"
LR="${LR:-1e-5}"
ENC_LAYER="${ENC_LAYER:-1}"
DEC_LAYER="${DEC_LAYER:-3}"
OPTIMIZER="${OPTIMIZER:-adamw}"
BCE_LOSS="${BCE_LOSS:-0}"
BACKBONE="${BACKBONE:-resnet101}"
DET_THRESHOLD="${DET_THRESHOLD:-0.1}"
CKPT="${CKPT:-}"

module purge
if ! module load "${PYTHON_MODULE}"; then
  echo "Requested module unavailable: ${PYTHON_MODULE}"
  echo "Trying fallback module: ${PYTHON_MODULE_FALLBACK}"
  module load "${PYTHON_MODULE_FALLBACK}"
  PYTHON_MODULE="${PYTHON_MODULE_FALLBACK}"
fi

# Follow CoE HPC activation guidance.
MODULE_BIN_DIR="$(dirname "$(command -v python3)")"
if [[ -f "${MODULE_BIN_DIR}/activate" ]]; then
  # shellcheck disable=SC1090
  source "${MODULE_BIN_DIR}/activate"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual env missing: ${VENV_PATH}"
  echo "Run scripts/setup_sttran_env.sh first."
  exit 2
fi
source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs "${SAVE_PATH}"

if [[ ! -f "fasterRCNN/models/faster_rcnn_ag.pth" ]]; then
  echo "Missing detector checkpoint: fasterRCNN/models/faster_rcnn_ag.pth"
  echo "Run scripts/download_sttran_assets.sh first."
  exit 3
fi

if [[ ! -f "dataloader/object_bbox_and_relationship_filtersmall.pkl" ]]; then
  echo "Missing dataloader/object_bbox_and_relationship_filtersmall.pkl"
  echo "Run scripts/download_sttran_assets.sh first."
  exit 4
fi

if [[ ! -f "${DATA_PATH}/annotations/object_bbox_and_relationship.pkl" ]]; then
  echo "Missing Action Genome annotations in ${DATA_PATH}/annotations"
  exit 5
fi

if [[ ! -d "${DATA_PATH}/frames" ]]; then
  echo "Missing frame directory: ${DATA_PATH}/frames"
  exit 6
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export STTRAN_OFFLINE_WORDVEC="${STTRAN_OFFLINE_WORDVEC:-1}"
export STTRAN_WV_DIR="${STTRAN_WV_DIR:-$PROJECT_ROOT/data}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export FUSED_ADAMW="${FUSED_ADAMW:-1}"
export RUN_EVAL="${RUN_EVAL:-0}"
export CKPT_EVERY="${CKPT_EVERY:-2}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
if [[ -z "${TRAIN_DETECTOR:-}" ]]; then
  if [[ "${BACKBONE}" == "vitdet" ]]; then
    TRAIN_DETECTOR=1
  else
    TRAIN_DETECTOR=0
  fi
fi
export TRAIN_DETECTOR
export VIT_LR="${VIT_LR:-${LR}}"
export TASK_LR="${TASK_LR:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export VITDET_MAX_TRAIN_ROIS="${VITDET_MAX_TRAIN_ROIS:-256}"
export VITDET_MAX_EVAL_ROIS="${VITDET_MAX_EVAL_ROIS:-1000}"
if [[ -z "${MAX_VIDEO_FRAMES:-}" ]]; then
  if [[ "${TRAIN_DETECTOR}" == "1" ]]; then
    MAX_VIDEO_FRAMES=24
  else
    MAX_VIDEO_FRAMES=-1
  fi
fi
export MAX_VIDEO_FRAMES
export VITDET_DET_CHUNK="${VITDET_DET_CHUNK:-4}"

WORLD_SIZE="${WORLD_SIZE:-${SLURM_NTASKS:-1}}"
CPUS_PER_RANK="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPUS_PER_RANK}"
if [[ -z "${NUM_WORKERS:-}" ]]; then
  if [[ "${TRAIN_DETECTOR}" == "1" ]]; then
    # Avoid forked worker CUDA context duplication when detector is trainable.
    NUM_WORKERS=0
  elif [[ "${CPUS_PER_RANK}" -gt 2 ]]; then
    NUM_WORKERS=$((CPUS_PER_RANK - 2))
  else
    NUM_WORKERS=1
  fi
fi
export NUM_WORKERS

echo "Running on host: $(hostname)"
echo "Master node list: ${SLURM_JOB_NODELIST:-unset}"
echo "CUDA device(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo "World size: ${WORLD_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "AMP: ${AMP} (${AMP_DTYPE})"
echo "RUN_EVAL: ${RUN_EVAL}"
echo "CKPT_EVERY: ${CKPT_EVERY}"
echo "Backbone: ${BACKBONE}"
echo "Det threshold: ${DET_THRESHOLD}"
echo "TRAIN_DETECTOR: ${TRAIN_DETECTOR}"
echo "VIT_LR: ${VIT_LR}  TASK_LR: ${TASK_LR}  WEIGHT_DECAY: ${WEIGHT_DECAY}"
echo "VITDET_MAX_TRAIN_ROIS: ${VITDET_MAX_TRAIN_ROIS}  VITDET_MAX_EVAL_ROIS: ${VITDET_MAX_EVAL_ROIS}"
echo "MAX_VIDEO_FRAMES: ${MAX_VIDEO_FRAMES}  VITDET_DET_CHUNK: ${VITDET_DET_CHUNK}"
echo "Python module: ${PYTHON_MODULE}"
echo "Mode: ${MODE}"
echo "Resume CKPT: ${CKPT:-none}"
echo "Data: ${DATA_PATH}"
echo "Save: ${SAVE_PATH}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29501}"
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

TRAIN_CMD=(
  python
  -u
  train.py
  -mode "${MODE}"
  -datasize "${DATASIZE}"
  -data_path "${DATA_PATH}"
  -save_path "${SAVE_PATH}"
  -nepoch "${NEPOCH}"
  -lr "${LR}"
  -enc_layer "${ENC_LAYER}"
  -dec_layer "${DEC_LAYER}"
  -optimizer "${OPTIMIZER}"
  --backbone "${BACKBONE}"
  --det_threshold "${DET_THRESHOLD}"
)

if [[ "${BCE_LOSS}" == "1" ]]; then
  TRAIN_CMD+=(-bce_loss)
fi
if [[ -n "${CKPT}" ]]; then
  TRAIN_CMD+=(-ckpt "${CKPT}")
fi

srun --ntasks="${WORLD_SIZE}" --ntasks-per-node=1 "${TRAIN_CMD[@]}"

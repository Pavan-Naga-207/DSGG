#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=sttran_vitdet_stage1
#SBATCH --output=logs/sttran_vitdet_stage1_%j.log
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train_detector_stage1.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.5}"
PYTHON_MODULE_FALLBACK="${PYTHON_MODULE_FALLBACK:-python3/3.11.7}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"

DATA_PATH="${DATA_PATH:-$HOME/special_topics/datasets/action_genome}"
DATASIZE="${DATASIZE:-large}"
BACKBONE="${BACKBONE:-vitdet}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/detector_stage1_${BACKBONE}_$(date +%Y%m%d_%H%M%S)}"
NEPOCH="${NEPOCH:-12}"
DET_THRESHOLD="${DET_THRESHOLD:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
VIT_LR="${VIT_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1}"
PATIENCE="${PATIENCE:-2}"
MAP_DELTA="${MAP_DELTA:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:--1}"
MAP_MAX_STEPS="${MAP_MAX_STEPS:-80}"
MAP_MAX_VIDEO_FRAMES="${MAP_MAX_VIDEO_FRAMES:-12}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
RESUME="${RESUME:-}"

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
  exit 2
fi
source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs "${SAVE_PATH}"

if [[ ! -f "fasterRCNN/models/faster_rcnn_ag.pth" ]]; then
  echo "Missing detector checkpoint: fasterRCNN/models/faster_rcnn_ag.pth"
  exit 3
fi

if [[ ! -f "dataloader/object_bbox_and_relationship_filtersmall.pkl" ]]; then
  echo "Missing dataloader/object_bbox_and_relationship_filtersmall.pkl"
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
export PIN_MEMORY="${PIN_MEMORY:-1}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export DETECTOR_LOAD_AG_CKPT="${DETECTOR_LOAD_AG_CKPT:-1}"
export DETECTOR_SKIP_AG_HEAD_LOAD="${DETECTOR_SKIP_AG_HEAD_LOAD:-1}"
export DETECTOR_REINIT_ROI_HEADS="${DETECTOR_REINIT_ROI_HEADS:-1}"
export DETECTOR_BN_MODE="${DETECTOR_BN_MODE:-groupnorm}"
export DETECTOR_GN_GROUPS="${DETECTOR_GN_GROUPS:-32}"
export DETECTOR_CLS_LOSS="${DETECTOR_CLS_LOSS:-ce}"
export DETECTOR_CLASS_WEIGHT_MODE="${DETECTOR_CLASS_WEIGHT_MODE:-none}"
export DETECTOR_CLASS_WEIGHT_BG="${DETECTOR_CLASS_WEIGHT_BG:-1.0}"
export DETECTOR_SAMPLER="${DETECTOR_SAMPLER:-default}"
export DETECTOR_REPEAT_THRESHOLD="${DETECTOR_REPEAT_THRESHOLD:-0.01}"
export DETECTOR_REPEAT_MAX="${DETECTOR_REPEAT_MAX:-4.0}"
export DETECTOR_BBOX_LOSS="${DETECTOR_BBOX_LOSS:-smoothl1}"
export DETECTOR_FOCUS_CLASSES="${DETECTOR_FOCUS_CLASSES:-person,table,chair,cup/glass/bottle,phone/camera,medicine,food,paper/notebook}"
export VITDET_FREEZE="${VITDET_FREEZE:-0}"
export VITDET_DET_CHUNK="${VITDET_DET_CHUNK:-2}"
export VITDET_MODEL="${VITDET_MODEL:-vit_base_patch16_224}"
export VIT_INPUT_SIZE="${VIT_INPUT_SIZE:-1024}"
export VIT_LSJ_MIN_SCALE="${VIT_LSJ_MIN_SCALE:-0.5}"
export VIT_LSJ_MAX_SCALE="${VIT_LSJ_MAX_SCALE:-2.0}"

echo "Running on host: $(hostname)"
echo "Python module: ${PYTHON_MODULE}"
echo "Data: ${DATA_PATH}"
echo "Save: ${SAVE_PATH}"
echo "Backbone: ${BACKBONE}"
echo "Detector head reinit: ${DETECTOR_REINIT_ROI_HEADS}"
echo "BN mode: ${DETECTOR_BN_MODE}"
echo "VITDET_MODEL: ${VITDET_MODEL}"
echo "VIT_INPUT_SIZE: ${VIT_INPUT_SIZE}"
echo "VIT_LSJ_MIN_SCALE/VIT_LSJ_MAX_SCALE: ${VIT_LSJ_MIN_SCALE}/${VIT_LSJ_MAX_SCALE}"
echo "VITDET_FREEZE: ${VITDET_FREEZE}"
echo "VITDET_DET_CHUNK: ${VITDET_DET_CHUNK}"
echo "DETECTOR_CLS_LOSS: ${DETECTOR_CLS_LOSS}"
echo "DETECTOR_CLASS_WEIGHT_MODE: ${DETECTOR_CLASS_WEIGHT_MODE}"
echo "DETECTOR_SAMPLER: ${DETECTOR_SAMPLER}"
echo "DETECTOR_BBOX_LOSS: ${DETECTOR_BBOX_LOSS}"
echo "DETECTOR_FOCUS_CLASSES: ${DETECTOR_FOCUS_CLASSES}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "NUM_WORKERS/EVAL_NUM_WORKERS: ${NUM_WORKERS}/${EVAL_NUM_WORKERS}"
echo "VIT_LR/HEAD_LR/WEIGHT_DECAY: ${VIT_LR}/${HEAD_LR}/${WEIGHT_DECAY}"

TRAIN_CMD=(
  python
  -u
  train_detector_stage1.py
  -data_path "${DATA_PATH}"
  -save_path "${SAVE_PATH}"
  -datasize "${DATASIZE}"
  -nepoch "${NEPOCH}"
  --backbone "${BACKBONE}"
  --det_threshold "${DET_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --eval_workers "${EVAL_NUM_WORKERS}"
  --vit_lr "${VIT_LR}"
  --head_lr "${HEAD_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --log_every "${LOG_EVERY}"
  --eval_every "${EVAL_EVERY}"
  --patience "${PATIENCE}"
  --map_delta "${MAP_DELTA}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --map_max_steps "${MAP_MAX_STEPS}"
  --map_max_video_frames "${MAP_MAX_VIDEO_FRAMES}"
  --iou_threshold "${IOU_THRESHOLD}"
  --cls_loss "${DETECTOR_CLS_LOSS}"
  --class_weight_mode "${DETECTOR_CLASS_WEIGHT_MODE}"
  --class_weight_bg "${DETECTOR_CLASS_WEIGHT_BG}"
  --sampler "${DETECTOR_SAMPLER}"
  --repeat_factor_threshold "${DETECTOR_REPEAT_THRESHOLD}"
  --repeat_factor_max "${DETECTOR_REPEAT_MAX}"
  --bbox_loss "${DETECTOR_BBOX_LOSS}"
  --focus_classes "${DETECTOR_FOCUS_CLASSES}"
)

if [[ -n "${RESUME}" ]]; then
  TRAIN_CMD+=(--resume "${RESUME}")
fi
if [[ "${PIN_MEMORY}" == "1" ]]; then
  TRAIN_CMD+=(--pin_memory)
fi

srun --ntasks=1 --ntasks-per-node=1 "${TRAIN_CMD[@]}"

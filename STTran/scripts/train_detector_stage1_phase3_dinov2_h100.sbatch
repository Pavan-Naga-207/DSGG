#!/bin/bash
#SBATCH --partition=condo
#SBATCH --job-name=sttran_p3_det
#SBATCH --output=logs/sttran_p3_det_%j.log
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train_detector_stage1_phase3.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.5}"
PYTHON_MODULE_FALLBACK="${PYTHON_MODULE_FALLBACK:-python3/3.11.7}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"

DATA_PATH="${DATA_PATH:-$HOME/special_topics/datasets/action_genome}"
DATASIZE="${DATASIZE:-large}"
BACKBONE="${BACKBONE:-dinov2}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/detector_stage1_phase3_${BACKBONE}_$(date +%Y%m%d_%H%M%S)}"
NEPOCH="${NEPOCH:-1}"
DET_THRESHOLD="${DET_THRESHOLD:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
VIT_LR="${VIT_LR:-0.0}"
HEAD_LR="${HEAD_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-1}"
PATIENCE="${PATIENCE:-2}"
MAP_DELTA="${MAP_DELTA:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
MAP_MAX_STEPS="${MAP_MAX_STEPS:-10}"
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
  source "${MODULE_BIN_DIR}/activate"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual env missing: ${VENV_PATH}"
  exit 2
fi
source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs "${SAVE_PATH}"

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
export VITDET_DET_CHUNK="${VITDET_DET_CHUNK:-1}"
export PHASE3_DINOV2_MODEL="${PHASE3_DINOV2_MODEL:-vit_base_patch14_dinov2}"
export PHASE3_FREEZE_BACKBONE="${PHASE3_FREEZE_BACKBONE:-1}"
export PHASE3_INPUT_SIZE="${PHASE3_INPUT_SIZE:-1024}"
export PHASE3_LSJ_MIN="${PHASE3_LSJ_MIN:-0.5}"
export PHASE3_LSJ_MAX="${PHASE3_LSJ_MAX:-2.0}"
export VIT_INPUT_SIZE="${VIT_INPUT_SIZE:-${PHASE3_INPUT_SIZE}}"
export VIT_LSJ_MIN_SCALE="${VIT_LSJ_MIN_SCALE:-${PHASE3_LSJ_MIN}}"
export VIT_LSJ_MAX_SCALE="${VIT_LSJ_MAX_SCALE:-${PHASE3_LSJ_MAX}}"
export PHASE3_DYNAMIC_IMG_SIZE="${PHASE3_DYNAMIC_IMG_SIZE:-1}"
export PHASE3_DYNAMIC_IMG_PAD="${PHASE3_DYNAMIC_IMG_PAD:-1}"
export PHASE3_DETECTOR_STRIDE="${PHASE3_DETECTOR_STRIDE:-16}"
export PHASE3_OUT_CHANNELS="${PHASE3_OUT_CHANNELS:-256}"
export PHASE3_ALIGN_CHANNELS="${PHASE3_ALIGN_CHANNELS:-1024}"
export PHASE3_ASSERT_DINOV2_PATH="${PHASE3_ASSERT_DINOV2_PATH:-1}"
export SGCLS_DUPLICATE_POLICY="${SGCLS_DUPLICATE_POLICY:-iou}"
export SGCLS_LABEL_SOURCE="${SGCLS_LABEL_SOURCE:-detector}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "Running on host: $(hostname)"
echo "Phase3 backbone: ${BACKBONE}"
echo "DINOv2 model: ${PHASE3_DINOV2_MODEL}"
echo "freeze backbone: ${PHASE3_FREEZE_BACKBONE}"
echo "input size: ${PHASE3_INPUT_SIZE}"
echo "LSJ range: ${PHASE3_LSJ_MIN}/${PHASE3_LSJ_MAX}"
echo "detector stride: ${PHASE3_DETECTOR_STRIDE}"
echo "phase3 assert path: ${PHASE3_ASSERT_DINOV2_PATH}"
echo "SGCLS duplicate policy: ${SGCLS_DUPLICATE_POLICY}"
echo "SGCLS label source: ${SGCLS_LABEL_SOURCE}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "NUM_WORKERS/EVAL_NUM_WORKERS: ${NUM_WORKERS}/${EVAL_NUM_WORKERS}"

TRAIN_CMD=(
  python
  -u
  train_detector_stage1_phase3.py
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

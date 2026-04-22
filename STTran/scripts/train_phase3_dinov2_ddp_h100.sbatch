#!/bin/bash
#SBATCH --partition=condo
#SBATCH --job-name=sttran_p3_ddp
#SBATCH --output=logs/sttran_p3_ddp_%j.log
#SBATCH --time=2-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train_phase3.py" ]]; then
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
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/sttran_phase3_ddp_${MODE}_$(date +%Y%m%d_%H%M%S)}"
NEPOCH="${NEPOCH:-10}"
LR="${LR:-1e-6}"
ENC_LAYER="${ENC_LAYER:-1}"
DEC_LAYER="${DEC_LAYER:-3}"
OPTIMIZER="${OPTIMIZER:-adamw}"
BCE_LOSS="${BCE_LOSS:-0}"
BACKBONE="${BACKBONE:-dinov2}"
DET_THRESHOLD="${DET_THRESHOLD:-0.1}"
CKPT="${CKPT:-}"
DETECTOR_CKPT="${DETECTOR_CKPT:-}"
export DETECTOR_CKPT

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
export STTRAN_OFFLINE_WORDVEC="${STTRAN_OFFLINE_WORDVEC:-1}"
export STTRAN_WV_DIR="${STTRAN_WV_DIR:-$PROJECT_ROOT/data}"
export DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export FUSED_ADAMW="${FUSED_ADAMW:-1}"
export RUN_EVAL="${RUN_EVAL:-0}"
export CKPT_EVERY="${CKPT_EVERY:-1}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
export LOG_EVERY="${LOG_EVERY:-100}"
export TRAIN_DETECTOR="${TRAIN_DETECTOR:-1}"
export VIT_LR="${VIT_LR:-0.0}"
export TASK_LR="${TASK_LR:-1e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
export TWO_STAGE="${TWO_STAGE:-1}"
export STAGE1_EPOCHS="${STAGE1_EPOCHS:-0}"
export STAGE1_VIT_LR="${STAGE1_VIT_LR:-0.0}"
export STAGE1_TASK_LR="${STAGE1_TASK_LR:-1e-4}"
export STAGE1_WEIGHT_DECAY="${STAGE1_WEIGHT_DECAY:-1e-4}"
export STAGE2_VIT_LR="${STAGE2_VIT_LR:-0.0}"
export STAGE2_TASK_LR="${STAGE2_TASK_LR:-1e-5}"
export STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-1e-4}"
export STAGE2_OBJ_WEIGHT="${STAGE2_OBJ_WEIGHT:-1.0}"
export STAGE2_REL_WEIGHT="${STAGE2_REL_WEIGHT:-1.0}"
export STAGE2_UNFREEZE_LAST_VIT_BLOCK="${STAGE2_UNFREEZE_LAST_VIT_BLOCK:-0}"
export DETECTOR_BN_MODE="${DETECTOR_BN_MODE:-groupnorm}"
export DETECTOR_GN_GROUPS="${DETECTOR_GN_GROUPS:-32}"
export DETECTOR_LOAD_AG_CKPT="${DETECTOR_LOAD_AG_CKPT:-0}"
export DETECTOR_SKIP_AG_HEAD_LOAD="${DETECTOR_SKIP_AG_HEAD_LOAD:-0}"
export DETECTOR_REINIT_ROI_HEADS="${DETECTOR_REINIT_ROI_HEADS:-0}"
export DETECTOR_MAP_EVERY="${DETECTOR_MAP_EVERY:-0}"
export DETECTOR_MAP_MAX_STEPS="${DETECTOR_MAP_MAX_STEPS:--1}"
export DETECTOR_MAP_WORKERS="${DETECTOR_MAP_WORKERS:-0}"
export DETECTOR_MAP_IOU="${DETECTOR_MAP_IOU:-0.5}"
export MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-24}"
export VITDET_DET_CHUNK="${VITDET_DET_CHUNK:-12}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export PHASE3_DINOV2_MODEL="${PHASE3_DINOV2_MODEL:-vit_base_patch14_dinov2}"
export PHASE3_FREEZE_BACKBONE="${PHASE3_FREEZE_BACKBONE:-1}"
export PHASE3_INPUT_SIZE="${PHASE3_INPUT_SIZE:-1024}"
export PHASE3_LSJ_MIN="${PHASE3_LSJ_MIN:-0.5}"
export PHASE3_LSJ_MAX="${PHASE3_LSJ_MAX:-2.0}"
export VIT_INPUT_SIZE="${VIT_INPUT_SIZE:-${PHASE3_INPUT_SIZE}}"
export PHASE3_DYNAMIC_IMG_SIZE="${PHASE3_DYNAMIC_IMG_SIZE:-1}"
export PHASE3_DYNAMIC_IMG_PAD="${PHASE3_DYNAMIC_IMG_PAD:-1}"
export PHASE3_DETECTOR_STRIDE="${PHASE3_DETECTOR_STRIDE:-16}"
export PHASE3_OUT_CHANNELS="${PHASE3_OUT_CHANNELS:-256}"
export PHASE3_ALIGN_CHANNELS="${PHASE3_ALIGN_CHANNELS:-1024}"
export PHASE3_ASSERT_DINOV2_PATH="${PHASE3_ASSERT_DINOV2_PATH:-1}"
export SGCLS_DUPLICATE_POLICY="${SGCLS_DUPLICATE_POLICY:-iou}"
export SGCLS_LABEL_SOURCE="${SGCLS_LABEL_SOURCE:-detector}"

WORLD_SIZE="${WORLD_SIZE:-${SLURM_NTASKS:-1}}"
export WORLD_SIZE
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29531}"
export MASTER_ADDR MASTER_PORT

echo "Running on host: $(hostname)"
echo "Master node list: ${SLURM_JOB_NODELIST:-unset}"
echo "CUDA device(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo "World size: ${WORLD_SIZE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "Phase3 mode: ${MODE} backbone=${BACKBONE} model=${PHASE3_DINOV2_MODEL}"
echo "freeze backbone: ${PHASE3_FREEZE_BACKBONE} two_stage=${TWO_STAGE} stage1_epochs=${STAGE1_EPOCHS}"
echo "RUN_EVAL: ${RUN_EVAL}"
echo "NUM_WORKERS/EVAL_NUM_WORKERS: ${NUM_WORKERS}/${EVAL_NUM_WORKERS}"
echo "MAX_VIDEO_FRAMES: ${MAX_VIDEO_FRAMES}"
echo "VITDET_DET_CHUNK: ${VITDET_DET_CHUNK}"
echo "DETECTOR_CKPT: ${DETECTOR_CKPT:-none}"
echo "Save: ${SAVE_PATH}"

TRAIN_CMD=(
  python -u train_phase3.py
  -mode "${MODE}"
  -save_path "${SAVE_PATH}"
  -data_path "${DATA_PATH}"
  -datasize "${DATASIZE}"
  -optimizer "${OPTIMIZER}"
  -lr "${LR}"
  -nepoch "${NEPOCH}"
  -enc_layer "${ENC_LAYER}"
  -dec_layer "${DEC_LAYER}"
  --backbone "${BACKBONE}"
  --det_threshold "${DET_THRESHOLD}"
)
if [[ -n "${CKPT}" ]]; then
  TRAIN_CMD+=(-ckpt "${CKPT}")
fi
if [[ "${BCE_LOSS}" == "1" ]]; then
  TRAIN_CMD+=(-bce_loss)
fi

srun --kill-on-bad-exit=1 --ntasks="${WORLD_SIZE}" --ntasks-per-node=1 "${TRAIN_CMD[@]}"

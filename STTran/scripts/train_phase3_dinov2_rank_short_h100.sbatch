#!/bin/bash
# Short Stage 2 ranking worker for Phase 3 detector checkpoints.
# Runs a brief Stage 2 training from a detector checkpoint, then a short
# SGDET validation pass with `with` constraint only, and writes a summary.

#SBATCH --partition=condo
#SBATCH --job-name=sttran_p3_rank
#SBATCH --output=logs/sttran_p3_rank_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
DATASIZE="${DATASIZE:-large}"
BACKBONE="${BACKBONE:-dinov2}"
DETECTOR_CKPT="${DETECTOR_CKPT:-}"
RANK_TAG="${RANK_TAG:-unnamed}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/sttran_phase3_rank_${RANK_TAG}_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${DETECTOR_CKPT}" ]]; then
  echo "DETECTOR_CKPT is required"
  exit 2
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
  exit 3
fi
source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs "${SAVE_PATH}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export STTRAN_OFFLINE_WORDVEC="${STTRAN_OFFLINE_WORDVEC:-1}"
export STTRAN_WV_DIR="${STTRAN_WV_DIR:-$PROJECT_ROOT/data}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export FUSED_ADAMW="${FUSED_ADAMW:-1}"
export RUN_EVAL=0
export CKPT_EVERY=1
export EVAL_EVERY=1
export LOG_EVERY="${LOG_EVERY:-100}"
export TRAIN_DETECTOR=1
export VIT_LR=0.0
export TASK_LR="${TASK_LR:-1e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
export TWO_STAGE=1
export STAGE1_EPOCHS=0
export STAGE1_VIT_LR=0.0
export STAGE1_TASK_LR="${STAGE1_TASK_LR:-1e-4}"
export STAGE1_WEIGHT_DECAY="${STAGE1_WEIGHT_DECAY:-1e-4}"
export STAGE2_VIT_LR=0.0
export STAGE2_TASK_LR="${STAGE2_TASK_LR:-1e-5}"
export STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-1e-4}"
export STAGE2_OBJ_WEIGHT="${STAGE2_OBJ_WEIGHT:-1.0}"
export STAGE2_REL_WEIGHT="${STAGE2_REL_WEIGHT:-1.0}"
export STAGE2_UNFREEZE_LAST_VIT_BLOCK=0
export DETECTOR_BN_MODE=groupnorm
export DETECTOR_GN_GROUPS=32
export DETECTOR_LOAD_AG_CKPT=0
export DETECTOR_SKIP_AG_HEAD_LOAD=0
export DETECTOR_REINIT_ROI_HEADS=0
export DETECTOR_MAP_EVERY=0
export DETECTOR_MAP_MAX_STEPS=-1
export DETECTOR_MAP_WORKERS=0
export DETECTOR_MAP_IOU=0.5
export MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-24}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
export VITDET_DET_CHUNK="${VITDET_DET_CHUNK:-12}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
export PHASE3_DINOV2_MODEL="${PHASE3_DINOV2_MODEL:-vit_base_patch14_dinov2}"
export PHASE3_FREEZE_BACKBONE=1
export PHASE3_INPUT_SIZE="${PHASE3_INPUT_SIZE:-1024}"
export PHASE3_LSJ_MIN="${PHASE3_LSJ_MIN:-0.5}"
export PHASE3_LSJ_MAX="${PHASE3_LSJ_MAX:-2.0}"
export VIT_INPUT_SIZE="${VIT_INPUT_SIZE:-${PHASE3_INPUT_SIZE}}"
export PHASE3_DYNAMIC_IMG_SIZE=1
export PHASE3_DYNAMIC_IMG_PAD=1
export PHASE3_DETECTOR_STRIDE=16
export PHASE3_OUT_CHANNELS=256
export PHASE3_ALIGN_CHANNELS=1024
export PHASE3_ASSERT_DINOV2_PATH=1
export SGCLS_DUPLICATE_POLICY=iou
export SGCLS_LABEL_SOURCE=detector
export DETECTOR_CKPT

SHORT_MODEL_PATH="${SAVE_PATH}/sttran_epoch_0.pth"
RANK_EVAL_LOG="${SAVE_PATH}/rank_eval_with.log"
RANK_SUMMARY_JSON="${SAVE_PATH}/rank_summary.json"
RANK_EVAL_CONSTRAINTS="${RANK_EVAL_CONSTRAINTS:-with}"
RANK_MAX_TEST_STEPS="${RANK_MAX_TEST_STEPS:-300}"
RANK_MODE="${RANK_MODE:-sgdet}"

echo "Running on host: $(hostname)"
echo "Rank tag: ${RANK_TAG}"
echo "Detector checkpoint: ${DETECTOR_CKPT}"
echo "Save path: ${SAVE_PATH}"
echo "Short Stage 2 config: nepoch=1 max_train_steps=${MAX_TRAIN_STEPS} max_video_frames=${MAX_VIDEO_FRAMES}"
echo "Short eval config: mode=${RANK_MODE} constraints=${RANK_EVAL_CONSTRAINTS} max_test_steps=${RANK_MAX_TEST_STEPS}"

srun --ntasks=1 --ntasks-per-node=1 \
  python -u train_phase3.py \
    -mode sgdet \
    -save_path "${SAVE_PATH}" \
    -data_path "${DATA_PATH}" \
    -datasize "${DATASIZE}" \
    -optimizer adamw \
    -lr 1e-6 \
    -nepoch 1 \
    -enc_layer 1 \
    -dec_layer 3 \
    --backbone "${BACKBONE}" \
    --det_threshold 0.1

if [[ ! -f "${SHORT_MODEL_PATH}" ]]; then
  echo "Expected checkpoint not found: ${SHORT_MODEL_PATH}"
  exit 4
fi

(
  export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS}"
  export EVAL_CONSTRAINTS="${RANK_EVAL_CONSTRAINTS}"
  export MAX_TEST_STEPS="${RANK_MAX_TEST_STEPS}"
  python -u evaluate_only_phase3.py \
    -mode "${RANK_MODE}" \
    -datasize "${DATASIZE}" \
    -data_path "${DATA_PATH}" \
    -model_path "${SHORT_MODEL_PATH}" \
    --backbone "${BACKBONE}" \
    --det_threshold 0.1 \
    -enc_layer 1 \
    -dec_layer 3
) | tee "${RANK_EVAL_LOG}"

python - "${RANK_EVAL_LOG}" "${RANK_SUMMARY_JSON}" "${RANK_TAG}" "${DETECTOR_CKPT}" "${SHORT_MODEL_PATH}" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rank_tag = sys.argv[3]
detector_ckpt = sys.argv[4]
model_path = sys.argv[5]
text = log_path.read_text()

r20 = None
block = None
lines = text.splitlines()
for i, line in enumerate(lines):
    if line.strip() == "======================with============================":
        block = lines[i:i + 8]
        break
if block is not None:
    for line in block:
        m = re.match(r"R@20:\s+([0-9.]+)", line.strip())
        if m:
            r20 = float(m.group(1))
            break

payload = {
    "rank_tag": rank_tag,
    "detector_ckpt": detector_ckpt,
    "stage2_model": model_path,
    "rank_mode": "sgdet",
    "constraints": "with",
    "max_test_steps": None,
    "with_r20": r20,
}
summary_path.write_text(json.dumps(payload, indent=2) + "\n")
print("RANK_SUMMARY_JSON", json.dumps(payload, sort_keys=True))
PY


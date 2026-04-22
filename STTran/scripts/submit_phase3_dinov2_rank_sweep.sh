#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DETECTOR_DIR="${DETECTOR_DIR:-$PROJECT_ROOT/outputs/detector_stage1_phase3_dinov2_full_20260420_220328}"
MODELS=( "$DETECTOR_DIR"/detector_stage1_epoch_*.pth )

if [[ ! -e "${MODELS[0]}" ]]; then
  echo "No detector checkpoints found in ${DETECTOR_DIR}"
  exit 2
fi

for ckpt in "${MODELS[@]}"; do
  base="$(basename "${ckpt}" .pth)"
  rank_tag="${base#detector_stage1_}"
  save_path="${PROJECT_ROOT}/outputs/sttran_phase3_rank_${rank_tag}_$(date +%Y%m%d_%H%M%S)"

  sbatch \
    --job-name="p3rank_${rank_tag}" \
    --output="logs/sttran_p3_rank_%j.log" \
    --export="ALL,DETECTOR_CKPT=${ckpt},RANK_TAG=${rank_tag},SAVE_PATH=${save_path},MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-500},RANK_MAX_TEST_STEPS=${RANK_MAX_TEST_STEPS:-300},VITDET_DET_CHUNK=${VITDET_DET_CHUNK:-12},NUM_WORKERS=${NUM_WORKERS:-8},EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-12},MAX_VIDEO_FRAMES=${MAX_VIDEO_FRAMES:-24}" \
    "$@" \
    scripts/train_phase3_dinov2_rank_short_h100.sbatch
done

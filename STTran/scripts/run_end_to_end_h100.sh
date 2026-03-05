#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$HOME/special_topics/datasets/action_genome}"
AG_ANNOTATIONS_DIR="${AG_ANNOTATIONS_DIR:-}"
CHARADES_VIDEO_DIR="${CHARADES_VIDEO_DIR:-}"
ACTION_GENOME_REPO="${ACTION_GENOME_REPO:-$HOME/special_topics/ActionGenome}"
TEST_ONLY=0

usage() {
  echo "Usage:"
  echo "  $0 [--dataset-root PATH] [--ag-annotations PATH] [--charades-videos PATH] [--action-genome-repo PATH] [--test-only]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --ag-annotations)
      AG_ANNOTATIONS_DIR="$2"
      shift 2
      ;;
    --charades-videos)
      CHARADES_VIDEO_DIR="$2"
      shift 2
      ;;
    --action-genome-repo)
      ACTION_GENOME_REPO="$2"
      shift 2
      ;;
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

scripts/setup_sttran_env.sh
scripts/download_sttran_assets.sh

PREP_ARGS=(--dataset-root "${DATASET_ROOT}" --action-genome-repo "${ACTION_GENOME_REPO}")
if [[ -n "${AG_ANNOTATIONS_DIR}" ]]; then
  PREP_ARGS+=(--ag-annotations "${AG_ANNOTATIONS_DIR}")
fi
if [[ -n "${CHARADES_VIDEO_DIR}" ]]; then
  PREP_ARGS+=(--charades-videos "${CHARADES_VIDEO_DIR}")
fi
scripts/prepare_actiongenome_dataset.sh "${PREP_ARGS[@]}"

export DATA_PATH="${DATASET_ROOT}"
if [[ "${TEST_ONLY}" == "1" ]]; then
  scripts/submit_sttran_h100.sh --test-only
else
  scripts/submit_sttran_h100.sh
fi

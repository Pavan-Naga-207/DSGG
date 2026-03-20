#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.5}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"
ACTION_GENOME_REPO="${ACTION_GENOME_REPO:-$HOME/special_topics/ActionGenome}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/special_topics/datasets/action_genome}"
AG_ANNOTATIONS_DIR="${AG_ANNOTATIONS_DIR:-}"
CHARADES_VIDEO_DIR="${CHARADES_VIDEO_DIR:-}"
ALL_FRAMES=0

usage() {
  echo "Usage:"
  echo "  $0 [--dataset-root PATH] [--ag-annotations PATH] [--charades-videos PATH] [--action-genome-repo PATH] [--all-frames]"
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
    --all-frames)
      ALL_FRAMES=1
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

module purge
module load "${PYTHON_MODULE}"

# Follow CoE HPC activation guidance.
MODULE_BIN_DIR="$(dirname "$(command -v python3)")"
if [[ -f "${MODULE_BIN_DIR}/activate" ]]; then
  # shellcheck disable=SC1090
  source "${MODULE_BIN_DIR}/activate"
fi

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

if [[ ! -d "${ACTION_GENOME_REPO}" ]]; then
  echo "ActionGenome repo not found at ${ACTION_GENOME_REPO}"
  exit 2
fi

mkdir -p "${DATASET_ROOT}"

if [[ -n "${AG_ANNOTATIONS_DIR}" ]]; then
  for required in object_bbox_and_relationship.pkl person_bbox.pkl frame_list.txt object_classes.txt relationship_classes.txt; do
    if [[ ! -f "${AG_ANNOTATIONS_DIR}/${required}" ]]; then
      echo "Missing ${required} in ${AG_ANNOTATIONS_DIR}"
      exit 3
    fi
  done

  ln -sfn "${AG_ANNOTATIONS_DIR}" "${DATASET_ROOT}/annotations"
fi

if [[ ! -d "${DATASET_ROOT}/annotations" ]]; then
  echo "Annotations are missing."
  echo "Provide --ag-annotations PATH to Action Genome annotation files."
  exit 4
fi

if [[ -n "${CHARADES_VIDEO_DIR}" ]]; then
  if [[ ! -d "${CHARADES_VIDEO_DIR}" ]]; then
    echo "Charades video directory not found: ${CHARADES_VIDEO_DIR}"
    exit 5
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required but not available in PATH."
    exit 6
  fi

  mkdir -p "${DATASET_ROOT}/frames"
  DUMP_ARGS=(
    --video_dir "${CHARADES_VIDEO_DIR}"
    --frame_dir "${DATASET_ROOT}/frames"
    --annotation_dir "${DATASET_ROOT}/annotations"
  )
  if [[ "${ALL_FRAMES}" == "1" ]]; then
    DUMP_ARGS+=(--all_frames)
  fi

  python "${ACTION_GENOME_REPO}/tools/dump_frames.py" "${DUMP_ARGS[@]}"
fi

if [[ ! -d "${DATASET_ROOT}/frames" ]]; then
  echo "Frames directory missing: ${DATASET_ROOT}/frames"
  echo "Provide --charades-videos PATH or create/symlink frames directory manually."
  exit 7
fi

echo "Action Genome dataset ready:"
echo "  annotations: ${DATASET_ROOT}/annotations"
echo "  frames:      ${DATASET_ROOT}/frames"

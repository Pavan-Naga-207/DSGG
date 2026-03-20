#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.5}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/sttran311}"

module purge
module load "${PYTHON_MODULE}"

MODULE_BIN_DIR="$(dirname "$(command -v python3)")"
if [[ -f "${MODULE_BIN_DIR}/activate" ]]; then
  # shellcheck disable=SC1090
  source "${MODULE_BIN_DIR}/activate"
fi

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

mkdir -p "${PROJECT_ROOT}/fasterRCNN/models" "${PROJECT_ROOT}/dataloader"

FRCNN_PATH="${PROJECT_ROOT}/fasterRCNN/models/faster_rcnn_ag.pth"
FILTERSMALL_PATH="${PROJECT_ROOT}/dataloader/object_bbox_and_relationship_filtersmall.pkl"

if command -v gdown >/dev/null 2>&1; then
  GDOWN_CMD=(gdown)
elif python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("gdown") else 1)
PY
then
  GDOWN_CMD=(python -m gdown)
else
  GDOWN_CMD=()
fi

if [[ ! -f "${FRCNN_PATH}" ]]; then
  if [[ ${#GDOWN_CMD[@]} -eq 0 ]]; then
    echo "gdown is not available. Install gdown in your home environment, then rerun."
    exit 2
  fi
  "${GDOWN_CMD[@]}" --fuzzy \
    "https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing" \
    -O "${FRCNN_PATH}"
fi

if [[ ! -f "${FILTERSMALL_PATH}" ]]; then
  if [[ ${#GDOWN_CMD[@]} -eq 0 ]]; then
    echo "gdown is not available. Install gdown in your home environment, then rerun."
    exit 2
  fi
  "${GDOWN_CMD[@]}" --fuzzy \
    "https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing" \
    -O "${FILTERSMALL_PATH}"
fi

echo "Assets ready:"
echo "  ${FRCNN_PATH}"
echo "  ${FILTERSMALL_PATH}"

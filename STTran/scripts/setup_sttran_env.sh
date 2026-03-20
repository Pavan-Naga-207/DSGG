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
  source "${MODULE_BIN_DIR}/activate"
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  python -m venv --system-site-packages "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

python - <<'PY'
import importlib
required = [
    "torch", "torchvision", "numpy", "scipy", "cv2", "pandas",
    "yaml", "h5py", "dill", "tqdm", "matplotlib",
]
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    raise SystemExit(
        "Missing required Python packages in current HPC runtime: "
        + ", ".join(missing)
        + ". Install them in your home env and rerun."
    )
PY

pushd "${PROJECT_ROOT}" >/dev/null

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.0;9.0}"

python - <<'PY'
import torch
from dataloader.action_genome import AG  # noqa: F401
from lib.evaluation_recall import BasicSceneGraphEvaluator  # noqa: F401
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms  # noqa: F401

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Environment setup and imports succeeded.")
PY

popd >/dev/null

echo "STTran environment is ready at ${VENV_PATH}"

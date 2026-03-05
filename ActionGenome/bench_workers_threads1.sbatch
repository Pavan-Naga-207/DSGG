#!/bin/bash
#SBATCH -J ag_bench_threads1
#SBATCH -p gpuqs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /home/019151743/special_topics/ActionGenome/bench_threads1.%j.out

set -euo pipefail

module purge
module load python3/3.11.7
source /home/019151743/venvs/sttran311/bin/activate

VIDEO_DIR=/home/019151743/special_topics/datasets/action_genome/videos/Charades_v1_480
ANN_DIR=/home/019151743/special_topics/datasets/action_genome/annotations_bench_300
FFMPEG_BIN=/home/019151743/special_topics/tools/ffmpeg_threads1.sh
BENCH_ROOT=/home/019151743/special_topics/datasets/action_genome/bench_frames_threads1

cd /home/019151743/special_topics/ActionGenome
mkdir -p "${BENCH_ROOT}"

echo "Starting benchmark on host: $(hostname)"
echo "ffmpeg wrapper: ${FFMPEG_BIN}"
echo "video_dir: ${VIDEO_DIR}"
echo "annotation_dir: ${ANN_DIR}"

for W in 16 24 32; do
  RUN_DIR="${BENCH_ROOT}/w${W}"
  rm -rf "${RUN_DIR}"

  START_TS=$(date +%s)
  python tools/dump_frames_parallel.py \
    --video_dir "${VIDEO_DIR}" \
    --frame_dir "${RUN_DIR}" \
    --annotation_dir "${ANN_DIR}" \
    --ffmpeg_bin "${FFMPEG_BIN}" \
    --num_workers "${W}"
  END_TS=$(date +%s)

  ELAPSED_SEC=$((END_TS - START_TS))
  DONE_COUNT=$(python - <<PY
from pathlib import Path
run_dir = Path("${RUN_DIR}")
print(sum(1 for p in run_dir.iterdir() if p.is_dir()) if run_dir.exists() else 0)
PY
)
  VIDS_PER_SEC=$(python - <<PY
elapsed = ${ELAPSED_SEC}
done = ${DONE_COUNT}
print(f"{(done / elapsed):.6f}" if elapsed > 0 else "0.000000")
PY
)

  echo "BENCH_RESULT workers=${W} elapsed_sec=${ELAPSED_SEC} done=${DONE_COUNT} vids_per_sec=${VIDS_PER_SEC}"
done

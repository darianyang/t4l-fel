#!/bin/bash
# BioEmu1 equilibrium samples of apo T4L L99A (same a3m as BioEmu2), then HPacker side chains.
#   ./run_bioemu1.sh <model: v1.1|v1.2> <num_samples> [gpu]
# Sampling resumes from existing batch_*.npz in the output dir. Unphysical samples are filtered
# by bioemu (filter_samples=True, default).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${1:-v1.2}; N=${2:-10000}; GPU=${3:-1}
OUT=runs/bioemu1/$MODEL
source /opt/software/miniconda/24.5.0/etc/profile.d/conda.sh
conda activate bioemu
CUDA_VISIBLE_DEVICES=$GPU python -m bioemu.sample --sequence inputs/t4l_l99a.a3m \
  --num_samples "$N" --output_dir "$OUT" --model_name "bioemu-$MODEL" --batch_size_100 "${BATCH100:-160}"

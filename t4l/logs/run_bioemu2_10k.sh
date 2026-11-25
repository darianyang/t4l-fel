#!/bin/bash
# 10,000 BioEmu2 equilibrium samples of apo T4L L99A: 2 x 5000 on GPUs 0/1 (resumable: just rerun).
cd "$(dirname "$0")/.."
source /opt/software/miniconda/24.5.0/etc/profile.d/conda.sh
conda activate bioemu2
CUDA_VISIBLE_DEVICES=0 nohup python run_bioemu2.py --n 5000 --seed 0     --tag seed0     > logs/bioemu2_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python run_bioemu2.py --n 5000 --seed 50000 --tag seed50000 > logs/bioemu2_seed50000.log 2>&1 &
wait

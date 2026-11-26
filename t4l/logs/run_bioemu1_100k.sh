#!/bin/bash
# BioEmu1 v1.2, 100k samples total: part0 resumes from the first 10k (runs/bioemu1/v1.2 batches),
# part1 is independent. Each bioemu.sample run draws its own time-based base seed.
cd "$(dirname "$0")/.."
source /opt/software/miniconda/24.5.0/etc/profile.d/conda.sh
conda activate bioemu
B=runs/bioemu1/v1.2_100k
for p in 0 1; do
  N=$([ $p = 0 ] && echo 55000 || echo 45000)
  CUDA_VISIBLE_DEVICES=$p nohup python -m bioemu.sample --sequence inputs/t4l_l99a.a3m \
    --num_samples $N --output_dir $B/part$p --model_name bioemu-v1.2 --batch_size_100 250 \
    > logs/bioemu1_v1.2_100k_part$p.log 2>&1 &
done
wait

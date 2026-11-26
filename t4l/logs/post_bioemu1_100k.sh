#!/bin/bash
# After the 100k BioEmu1 v1.2 sampling finishes: HPacker side chains for both parts, then assemble.
cd "$(dirname "$0")/.."
while pgrep -f "bioemu.sample.*v1.2_100k" >/dev/null; do sleep 120; done
source /opt/software/miniconda/24.5.0/etc/profile.d/conda.sh
conda activate bioemu2
B=runs/bioemu1/v1.2_100k
for p in 0 1; do grep -q "Completed" logs/bioemu1_v1.2_100k_part$p.log || { echo "part$p sampling did not complete"; exit 1; }; done
for p in 0 1; do python sidechains.py prep $B/part$p; done
python sidechains.py run $B/part0 --nproc 66 --threads 2 &
python sidechains.py run $B/part1 --nproc 54 --threads 2 &
wait
for p in 0 1; do python sidechains.py assemble $B/part$p; done
echo "post-processing done"

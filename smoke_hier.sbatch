#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Validate the hierarchical (gloo-intra / NCCL-inter) all-reduce on real GPUs.
# Smallest shape that exercises co-location: 2 GPUs x 2 ranks = 4 processes.
#SBATCH --job-name=smoke_hier
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/oscar/data/stellex/glvov/wbc-consolidate-20260810/logs/smoke_hier_%j.out

set -x
REPO=/oscar/data/stellex/glvov/wbc-consolidate-20260810
PY=/oscar/data/stellex/glvov/glvov-envs/sonic-isaaclab232/bin/python
mkdir -p "$REPO/logs"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=2

nvidia-smi --query-gpu=index,name,memory.used --format=csv

# 4 ranks over 2 GPUs -> ranks 0,2 on gpu0 and 1,3 on gpu1 (rank % n_gpu).
"$PY" -m torch.distributed.run --nproc_per_node=4 --master_port=29677 \
    gear_sonic/tests/smoke_hierarchical_pg.py --n-gpu 2 --stack 2
echo "SMOKE_EXIT=$?"

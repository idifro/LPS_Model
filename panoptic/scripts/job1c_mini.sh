#!/bin/bash
#BSUB -J job1c_mini
#BSUB -q gpu
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=32000]"
#BSUB -o logs/job1c_mini_%J.out
#BSUB -e logs/job1c_mini_%J.err

set -euo pipefail

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
source ~/.bashrc
conda activate openscene

mkdir -p logs

echo "=== Job 1c mini smoke test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Verify before training
echo ""
echo "--- Verification ---"
python panoptic/scripts/verify_job1c_losses.py --device cuda

echo ""
echo "--- Smoke test (50 steps) ---"
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1c_mini.yaml \
    --smoke_test

echo "=== Done: $(date) ==="

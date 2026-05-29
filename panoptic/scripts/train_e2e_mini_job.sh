#!/bin/bash
#BSUB -J e2e_mini_panoptic
#BSUB -o panoptic/logs/e2e_mini_%J.stdout
#BSUB -e panoptic/logs/e2e_mini_%J.stderr
#BSUB -q batch_a100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -M 8192
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"

# ─── E2E Encoder Training — nuScenes Mini ───────────────────────────────────
# Unfreezes backbone decoder layers with differential LR.
# Warm-starts from frozen-backbone best checkpoint.
# Expected: ~3-5 GB GPU, ~2-3 min/epoch on mini (323 samples).

module load conda
conda activate opensc_env
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=4
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model

mkdir -p panoptic/logs
mkdir -p panoptic/checkpoints_e2e_mini

echo "=============================================="
echo "  E2E Encoder Training — nuScenes Mini"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

python -u panoptic/train/train_panoptic_e2e.py \
    --config panoptic/configs/mini_e2e_encoder.yaml

echo "Done: $(date)"

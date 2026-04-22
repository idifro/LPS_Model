#!/bin/bash
#BSUB -J panoptic_train
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 16384
#BSUB -W 24:00
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/train_%J.out
#BSUB -e panoptic/logs/train_%J.err

# ── Setup ──
module load conda
conda activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/checkpoints

# ── Config ──
CONFIG="${CONFIG:-panoptic/configs/default.yaml}"
echo "Config: $CONFIG"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── Auto-resume: find latest.pth if it exists ──
RESUME_FLAG=""
LATEST="panoptic/checkpoints/latest.pth"
INTERRUPTED="panoptic/checkpoints/interrupted.pth"
if [ -f "$INTERRUPTED" ]; then
    echo "Found interrupted checkpoint, resuming from $INTERRUPTED"
    RESUME_FLAG="--resume $INTERRUPTED"
elif [ -f "$LATEST" ]; then
    echo "Found latest checkpoint, resuming from $LATEST"
    RESUME_FLAG="--resume $LATEST"
else
    echo "No checkpoint found, training from scratch"
fi

# ── Train ──
python -u panoptic/train/train_panoptic.py \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "End: $(date)"
echo "Exit code: $?"

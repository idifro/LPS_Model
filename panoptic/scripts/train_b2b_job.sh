#!/bin/bash
#BSUB -J panoptic_b2b                         # Job name
#BSUB -o panoptic/logs/train_b2b_%J.stdout     # Stdout file
#BSUB -e panoptic/logs/train_b2b_%J.stderr     # Stderr file
#BSUB -q batch_a100                            # Queue: A100 GPUs
#BSUB -W 48:00                                 # Wall time: 48 hours
#BSUB -n 8                                     # 8 CPU cores
#BSUB -M 8192                                  # Memory per core (8GB x 8 = 64GB)
#BSUB -R "span[hosts=1]"                       # Single node
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"

# ── Environment setup ──
module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

mkdir -p panoptic/logs panoptic/checkpoints_b2b

# ── Job info ──
CONFIG="panoptic/configs/full_nuscenes_b2b.yaml"
echo "Phase 5a (B2b) Class-Agnostic Training"
echo "Config: $CONFIG"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── Auto-resume ──
RESUME_FLAG=""
LATEST="panoptic/checkpoints_b2b/latest.pth"
INTERRUPTED="panoptic/checkpoints_b2b/interrupted.pth"
if [ -f "$INTERRUPTED" ]; then
    echo "Resuming from interrupted: $INTERRUPTED"
    RESUME_FLAG="--resume $INTERRUPTED"
elif [ -f "$LATEST" ]; then
    echo "Resuming from latest: $LATEST"
    RESUME_FLAG="--resume $LATEST"
else
    echo "Training from warm-start (Phase 4 checkpoint)"
fi

# ── Run training ──
python -u panoptic/train/train_panoptic_b2b.py \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "End: $(date)"
echo "Exit code: $?"

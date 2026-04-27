#!/bin/bash
#BSUB -J panoptic_full                        # Job name
#BSUB -o panoptic/logs/train_full_%J.stdout    # Stdout file (%J = job ID)
#BSUB -e panoptic/logs/train_full_%J.stderr    # Stderr file
#BSUB -q batch_a100                            # Queue: A100 GPUs, max 168h
#BSUB -W 48:00                                 # Wall time: 48 hours
#BSUB -n 8                                     # 8 CPU cores (for num_workers=4 + overhead)
#BSUB -M 8192                                  # Memory per CPU core in MB (8GB × 8 = 64GB total)
#BSUB -R "span[hosts=1]"                       # All cores on single node
#BSUB -gpu "num=1:mode=exclusive_process:mps=no" # 1 full GPU, no MPS sharing

# ── Environment setup ──
module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

# Ensure CUDA libraries are visible (conda activate can clobber LD_LIBRARY_PATH)
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

mkdir -p panoptic/logs panoptic/checkpoints_full

# ── Job info ──
CONFIG="panoptic/configs/full_nuscenes.yaml"
echo "Config: $CONFIG"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── Auto-resume: pick up from interrupted or latest checkpoint ──
RESUME_FLAG=""
LATEST="panoptic/checkpoints_full/latest.pth"
INTERRUPTED="panoptic/checkpoints_full/interrupted.pth"
if [ -f "$INTERRUPTED" ]; then
    echo "Resuming from interrupted checkpoint: $INTERRUPTED"
    RESUME_FLAG="--resume $INTERRUPTED"
elif [ -f "$LATEST" ]; then
    echo "Resuming from latest checkpoint: $LATEST"
    RESUME_FLAG="--resume $LATEST"
else
    echo "Training from scratch"
fi

# ── Run training ──
python -u panoptic/train/train_panoptic.py \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "End: $(date)"
echo "Exit code: $?"

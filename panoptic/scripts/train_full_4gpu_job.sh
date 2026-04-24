#!/bin/bash
#BSUB -J panoptic_full_4gpu                        # Job name
#BSUB -o panoptic/logs/train_full_4gpu_%J.stdout    # Stdout file (%J = job ID)
#BSUB -e panoptic/logs/train_full_4gpu_%J.stderr    # Stderr file
#BSUB -q batch_a100                                 # Queue: A100 GPUs, max 168h
#BSUB -W 48:00                                      # Wall time: 48 hours
#BSUB -n 16                                         # 16 CPU cores (4 workers × 4 GPUs)
#BSUB -M 8192                                       # Memory per CPU core in MB (8GB × 16 = 128GB total)
#BSUB -R "span[hosts=1]"                            # All cores on single node
#BSUB -gpu "num=4:mode=exclusive_process:mps=no"    # 4 full GPUs, no MPS sharing

# ── Environment setup ──
module load conda
conda activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

mkdir -p panoptic/logs panoptic/checkpoints_full_4gpu

# ── Job info ──
CONFIG="panoptic/configs/full_nuscenes_4gpu.yaml"
echo "Config: $CONFIG"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── Auto-resume: pick up from interrupted or latest checkpoint ──
RESUME_FLAG=""
LATEST="panoptic/checkpoints_full_4gpu/latest.pth"
INTERRUPTED="panoptic/checkpoints_full_4gpu/interrupted.pth"
if [ -f "$INTERRUPTED" ]; then
    echo "Resuming from interrupted checkpoint: $INTERRUPTED"
    RESUME_FLAG="--resume $INTERRUPTED"
elif [ -f "$LATEST" ]; then
    echo "Resuming from latest checkpoint: $LATEST"
    RESUME_FLAG="--resume $LATEST"
else
    echo "Training from scratch"
fi

# ── Run training (4-GPU DDP via torchrun) ──
# Use conda env python explicitly to avoid system python picking up wrong torchrun
python -m torch.distributed.run --standalone --nproc_per_node=4 \
    panoptic/train/train_panoptic.py \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "End: $(date)"
echo "Exit code: $?"

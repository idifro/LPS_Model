#!/bin/bash
#BSUB -J job1b-v2-lo-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 36:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1b_v2_lo_full_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1b_v2_lo_full_%J.err

set -euo pipefail

# Job 1b v2 Ablation A: full run (15 epochs, v1.0-trainval, L_O only)
# Gate 9: novel PQ > 0.05 at final epoch, PQ_base >= 0.38

echo "[$(date)] === Job 1b v2 L_O Full: starting ==="
echo "[$(date)] Host: $(hostname), Job ID: $LSB_JOBID"

echo "[$(date)] Loading modules..."
module load conda
source activate openscene
echo "[$(date)] Conda env: $CONDA_DEFAULT_ENV"

module load cuda/11.0.2
module load cudnn/11.0_v8.0
echo "[$(date)] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4
echo "[$(date)] PYTHONPATH=$PYTHONPATH"

echo "[$(date)] Changing to working directory..."
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
echo "[$(date)] CWD: $(pwd)"
mkdir -p panoptic/logs panoptic/checkpoints_job1b_v2_lo

echo "[$(date)] Verifying prototype files..."
for f in panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
         panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt \
         panoptic/clip_prototypes/negative_bank_v2.pt; do
    if [ ! -f "$f" ]; then
        echo "[$(date)] ERROR: Missing $f"
        exit 1
    fi
    echo "[$(date)] OK: $f ($(du -sh "$f" | cut -f1))"
done

echo "[$(date)] Checking warm-start checkpoint..."
WARMSTART="panoptic/checkpoints_job1a_v2/best.pth"
if [ ! -f "$WARMSTART" ]; then
    echo "[$(date)] ERROR: Warm-start not found at $WARMSTART — job1a_v2 must finish first"
    exit 1
fi
echo "[$(date)] Warm-start found: $WARMSTART ($(du -sh "$WARMSTART" | cut -f1))"

echo "[$(date)] Checking precomputed CLIP features (trainval)..."
CLIP_DIR=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-trainval
N_FILES=$(find "$CLIP_DIR" -name "*.npz" 2>/dev/null | wc -l || echo 0)
if [ "$N_FILES" -lt 28000 ]; then
    echo "[$(date)] ERROR: Only $N_FILES CLIP feature files in $CLIP_DIR (need >= 28000)"
    exit 1
fi
echo "[$(date)] Found $N_FILES precomputed CLIP feature files (trainval)"

echo "[$(date)] Checking for resume checkpoint..."
RESUME_FLAG=""
LATEST="panoptic/checkpoints_job1b_v2_lo/latest.pth"
if [ -f "$LATEST" ]; then
    echo "[$(date)] latest.pth found — resuming from: $LATEST ($(du -sh "$LATEST" | cut -f1))"
    RESUME_FLAG="--resume $LATEST"
else
    echo "[$(date)] No latest.pth — starting fresh"
fi

echo "[$(date)] Disk space: $(df -sh /fs/scratch/SX_ETL3_GenAI_Team/mha2cob 2>/dev/null | tail -1)"
echo "[$(date)] Python: $(python --version)"

echo "[$(date)] === Launching training ==="
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_lo_only.yaml \
    $RESUME_FLAG
EXIT_CODE=$?

echo "[$(date)] === Training exited with code $EXIT_CODE ==="
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed — check logs"
    exit $EXIT_CODE
fi

echo "[$(date)] Training complete."

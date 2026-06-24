#!/bin/bash
#BSUB -J job1b-mini
#BSUB -q inter_a100
#BSUB -n 4
#BSUB -M 4096
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1b_mini_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1b_mini_%J.err

set -euo pipefail

# Job 1b Mini: Smoke test — Camera CLIP Distillation (L_O + L_V)
# 10 epochs on v1.0-mini, warm-start from job1a_v1 trainval checkpoint
# L_O weight=2.0, L_V weight=1.0 (increased from job1b baseline)

echo "[$(date)] === Job 1b Mini: starting ==="
echo "[$(date)] Host: $(hostname), Job ID: $LSB_JOBID"

echo "[$(date)] Loading modules..."
module load conda
source activate openscene
echo "[$(date)] Conda env: $CONDA_DEFAULT_ENV"

module load cuda/11.0.2
module load cudnn/11.0_v8.0
echo "[$(date)] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

module load proxy4server-access/
proxy_on
echo "[$(date)] Proxy enabled"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4
echo "[$(date)] PYTHONPATH=$PYTHONPATH"

echo "[$(date)] Changing to working directory..."
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
echo "[$(date)] CWD: $(pwd)"
mkdir -p panoptic/logs panoptic/checkpoints_job1b_mini

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
WARMSTART="panoptic/checkpoints_job1a_v1/best.pth"
if [ ! -f "$WARMSTART" ]; then
    echo "[$(date)] ERROR: Warm-start not found at $WARMSTART — job1a_v1 must finish first"
    exit 1
fi
echo "[$(date)] Warm-start found: $WARMSTART ($(du -sh "$WARMSTART" | cut -f1))"

echo "[$(date)] Checking precomputed CLIP features (mini)..."
CLIP_DIR=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-mini
N_FILES=$(ls "$CLIP_DIR"/*.npz 2>/dev/null | wc -l)
if [ "$N_FILES" -lt 50 ]; then
    echo "[$(date)] ERROR: Only $N_FILES CLIP feature files in $CLIP_DIR (need >= 50)"
    exit 1
fi
echo "[$(date)] Found $N_FILES precomputed CLIP feature files (mini)"

echo "[$(date)] Checking for resume checkpoint..."
RESUME_FLAG=""
LATEST="panoptic/checkpoints_job1b_mini/latest.pth"
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
    --config panoptic/configs/job1b_mini.yaml \
    $RESUME_FLAG
EXIT_CODE=$?

echo "[$(date)] === Training exited with code $EXIT_CODE ==="
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed — check logs"
    exit $EXIT_CODE
fi

echo "[$(date)] Training complete."

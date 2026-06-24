#!/bin/bash
#BSUB -J job1a-v2-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1a_v2_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1a_v2_%J.err

set -euo pipefail

# Job 1a v2: LoRA + SingleCLIPHead + C1 TokenLoss — v2 neg-bank + v2 prompts
# 12 epochs on v1.0-trainval, warm-start from checkpoints_token_full_v2/best.pth

echo "[$(date)] === Job 1a v2 Full: starting ==="
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
mkdir -p panoptic/logs panoptic/checkpoints_job1a_v2

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
WARMSTART="panoptic/checkpoints_token_full_v2/best.pth"
if [ -f "$WARMSTART" ]; then
    echo "[$(date)] Warm-start found: $WARMSTART ($(du -sh "$WARMSTART" | cut -f1))"
else
    echo "[$(date)] WARNING: Warm-start not found at $WARMSTART — training from scratch"
fi

echo "[$(date)] Checking for resume checkpoint..."
RESUME_FLAG=""
LATEST="panoptic/checkpoints_job1a_v2/latest.pth"
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
    --config panoptic/configs/job1a_v2.yaml \
    $RESUME_FLAG
EXIT_CODE=$?

echo "[$(date)] === Training exited with code $EXIT_CODE ==="
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed — check logs"
    exit $EXIT_CODE
fi

echo "[$(date)] Training complete."

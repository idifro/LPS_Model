#!/bin/bash
#BSUB -J job1a-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 38:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1a_full_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1a_full_%J.err

set -euo pipefail

# Job 1a Full: LoRA + SingleCLIPHead (A4) + C1 TokenLoss (216+) + T4 MultiPrompt
# 20 epochs on v1.0-trainval, warm-start from best.pth

echo "[$(date)] === Job 1a Full: starting ==="
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
mkdir -p /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/checkpoints_job1a_v1

echo "[$(date)] Verifying prototype files..."
for f in /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
         /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt.pt \
         /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/clip_prototypes/negative_bank_200.pt; do
    if [ ! -f "$f" ]; then
        echo "[$(date)] ERROR: Missing $f — run prereq_protos_job.sh first"
        exit 1
    fi
    echo "[$(date)] OK: $f ($(du -sh "$f" | cut -f1))"
done

echo "[$(date)] Checking warm-start checkpoint..."
WARMSTART="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/checkpoints_token_full_v2/best.pth"
if [ -f "$WARMSTART" ]; then
    echo "[$(date)] Warm-start found: $WARMSTART ($(du -sh "$WARMSTART" | cut -f1))"
else
    echo "[$(date)] WARNING: Warm-start not found at $WARMSTART — training from scratch"
fi

echo "[$(date)] Checking for resume checkpoint..."
RESUME_FLAG=""
LATEST="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/checkpoints_job1a_v1/latest.pth"
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
    --config panoptic/configs/job1a.yaml \
    $RESUME_FLAG
EXIT_CODE=$?

echo "[$(date)] === Training exited with code $EXIT_CODE ==="
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed — check logs"
    exit $EXIT_CODE
fi

echo "[$(date)] Training complete."

#!/bin/bash
#BSUB -J e2e_mini_panoptic_last_n_layers
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 8192
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/e2e_mini_%J.stdout
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/e2e_mini_%J.stderr

set -euo pipefail

echo "[$(date)] === E2E Encoder Training (nuScenes Mini): starting ==="
echo "[$(date)] Host: $(hostname), Job ID: $LSB_JOBID"

echo "[$(date)] Loading modules..."
module load conda
source activate opensc_env
echo "[$(date)] Conda env: $CONDA_DEFAULT_ENV"

module load cuda/11.0.2
module load cudnn/11.0_v8.0

echo "[$(date)] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

# Match cluster setup used by working job
module load proxy4server-access/ || true
proxy_on || true

# gcc/5.4.0 provides real libgfortran.so.3 with GFORTRAN_1.0 needed by libopenblas
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libgfortran.so.3)):$LD_LIBRARY_PATH

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=4

echo "[$(date)] PYTHONPATH=$PYTHONPATH"

echo "[$(date)] Changing to working directory..."
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
echo "[$(date)] CWD: $(pwd)"

mkdir -p \
    /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs_last_n_layers \
    /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/checkpoints_e2e_mini_last_n_layers

CONFIG="panoptic/configs/mini_e2e_encoder.yaml"

echo "[$(date)] Config: $CONFIG"

if [ ! -f "$CONFIG" ]; then
    echo "[$(date)] ERROR: Missing config file: $CONFIG"
    exit 1
fi

echo "[$(date)] Python: $(python --version)"
echo "[$(date)] Disk space: $(df -sh /fs/scratch/SX_ETL3_GenAI_Team/mha2cob 2>/dev/null | tail -1)"

echo "[$(date)] Scheduler settings from config:"
grep -Ei "warmup|epoch|lr|scheduler" "$CONFIG" || true

# Warm-start checkpoint check
WARMSTART="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/checkpoints_token_full_v2/best.pth"

if [ -f "$WARMSTART" ]; then
    echo "[$(date)] Warm-start found: $WARMSTART"
else
    echo "[$(date)] WARNING: Warm-start checkpoint not found: $WARMSTART"
fi

# Auto-resume support
RESUME_FLAG=""
LATEST="panoptic/checkpoints_e2e_mini_last_n_layers/latest.pth"
INTERRUPTED="panoptic/checkpoints_e2e_mini_last_n_layers/interrupted.pth"

if [ -f "$INTERRUPTED" ]; then
    echo "[$(date)] Resuming from interrupted checkpoint: $INTERRUPTED"
    RESUME_FLAG="--resume $INTERRUPTED"
elif [ -f "$LATEST" ]; then
    echo "[$(date)] Resuming from latest checkpoint: $LATEST"
    RESUME_FLAG="--resume $LATEST"
else
    echo "[$(date)] No checkpoint found — starting fresh"
fi

echo "[$(date)] === Launching training ==="

python -u panoptic/train/train_panoptic_e2e.py \
    --config "$CONFIG" \
    $RESUME_FLAG

EXIT_CODE=$?

echo "[$(date)] === Training exited with code $EXIT_CODE ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed"
    exit $EXIT_CODE
fi

echo "[$(date)] Training complete."
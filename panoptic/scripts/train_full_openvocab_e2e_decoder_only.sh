#!/bin/bash
#BSUB -J panoptic_full_openvocab_v2
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/decoder_only_train_full_%J.stdout
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/decoder_only_train_full_%J.stderr

set -eo pipefail

echo "[$(date)] === Panoptic Full Training v2: starting ==="
echo "[$(date)] Host: $(hostname), Job ID: $LSB_JOBID"

echo "[$(date)] Loading modules..."
set +u
module load conda
source activate opensc_env
set -u

echo "[$(date)] Conda env: $CONDA_DEFAULT_ENV"
module load cuda/11.0.2
module load cudnn/11.0_v8.0

echo "[$(date)] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libgfortran.so.3)):$LD_LIBRARY_PATH
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=4
export WANDB_MODE=offline

echo "[$(date)] PYTHONPATH=$PYTHONPATH"
echo "[$(date)] W&B mode: offline"

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
echo "[$(date)] CWD: $(pwd)"

SAVE_DIR="panoptic/checkpoints/full_openvocab_decoder_only"
mkdir -p \
    /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs \
    "$SAVE_DIR"

CONFIG="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/scripts/full_openvocab_e2e_decoder_only.yaml"
echo "[$(date)] Config: $CONFIG"
if [ ! -f "$CONFIG" ]; then
    echo "[$(date)] ERROR: Missing config file: $CONFIG"
    exit 1
fi

echo "[$(date)] Python: $(python --version)"
echo "[$(date)] Disk space: $(df -sh /fs/scratch/SX_ETL3_GenAI_Team/mha2cob 2>/dev/null | tail -1)"

# ── Resume logic: best.pth > interrupted.pth > latest.pth > fresh start ──
# NOTE: best.pth is preferred because latest.pth may be from a diverged epoch.
# Pass --reset-adam always when resuming to clear corrupted Adam momentum.
RESUME_FLAG=""
BEST="$SAVE_DIR/best.pth"
INTERRUPTED="$SAVE_DIR/interrupted.pth"
LATEST="$SAVE_DIR/latest.pth"

if [ -f "$INTERRUPTED" ]; then
    echo "[$(date)] Resuming from interrupted checkpoint: $INTERRUPTED"
    RESUME_FLAG="--resume $INTERRUPTED --reset-adam"
elif [ -f "$BEST" ]; then
    echo "[$(date)] Resuming from best checkpoint: $BEST"
    RESUME_FLAG="--resume $BEST --reset-adam"
elif [ -f "$LATEST" ]; then
    echo "[$(date)] Resuming from latest checkpoint: $LATEST"
    RESUME_FLAG="--resume $LATEST --reset-adam"
else
    echo "[$(date)] No checkpoint found — starting fresh"
fi

echo "[$(date)] === Launching training ==="
python -u panoptic/train/train_openvocab_e2e.py \
    --config "$CONFIG" \
    $RESUME_FLAG

EXIT_CODE=$?
echo "[$(date)] === Training exited with code $EXIT_CODE ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "[$(date)] Training complete."
echo "[$(date)] To sync W&B: wandb sync panoptic/wandb/offline-run-*/"
#!/bin/bash
#BSUB -J panoptic_full_last_n
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 72:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/last_n_train_full_%J.stdout
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/last_n_train_full_%J.stderr

set -eo pipefail

echo "[$(date)] === Panoptic Full Training last_n: starting ==="

set +u
module load conda
source activate opensc_env
set -u

module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libgfortran.so.3)):$LD_LIBRARY_PATH
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=4
export WANDB_MODE=offline

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model

SAVE_DIR="panoptic/checkpoints/full_openvocab_last_n"
mkdir -p panoptic/logs "$SAVE_DIR"

CONFIG="panoptic/scripts/full_openvocab_e2e_last_n.yaml"

# Resume logic
RESUME_FLAG=""
INTERRUPTED="$SAVE_DIR/interrupted.pth"
BEST="$SAVE_DIR/best.pth"
LATEST="$SAVE_DIR/latest.pth"

if [ -f "$INTERRUPTED" ]; then
    echo "[$(date)] Resuming from interrupted: $INTERRUPTED"
    RESUME_FLAG="--resume $INTERRUPTED --reset-adam"
elif [ -f "$BEST" ]; then
    echo "[$(date)] Resuming from best: $BEST"
    RESUME_FLAG="--resume $BEST --reset-adam"
elif [ -f "$LATEST" ]; then
    echo "[$(date)] Resuming from latest: $LATEST"
    RESUME_FLAG="--resume $LATEST --reset-adam"
else
    echo "[$(date)] Fresh start with decoder_only warmstart via config"
fi

python -u panoptic/train/train_openvocab_e2e.py \
    --config "$CONFIG" \
    $RESUME_FLAG

EXIT_CODE=$?
echo "[$(date)] Exited with code $EXIT_CODE"
#!/bin/bash
#BSUB -J ov_full_decoder_only
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 32000
#BSUB -W 72:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/ov_full_decoder_only_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model/panoptic/logs/ov_full_decoder_only_%J.err

set -euo pipefail

echo "[$(date)] === Open-Vocab FULL decoder_only: starting ==="
echo "[$(date)] Host: $(hostname), Job ID: $LSB_JOBID"

echo "[$(date)] Loading modules..."
module load conda
source activate opensc_env
echo "[$(date)] Conda env: $CONDA_DEFAULT_ENV"

module load cuda/11.0.2
module load cudnn/11.0_v8.0
echo "[$(date)] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

module load proxy4server-access/
proxy_on
echo "[$(date)] Proxy enabled"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libgfortran.so.3)):$LD_LIBRARY_PATH
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=4

echo "[$(date)] PYTHONPATH=$PYTHONPATH"
echo "[$(date)] Changing to working directory..."
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
echo "[$(date)] CWD: $(pwd)"

mkdir -p panoptic/logs
mkdir -p panoptic/checkpoints/full_openvocab_decoder_only

echo "[$(date)] Verifying prototype file..."
PROTO=panoptic/clip_prototypes/text_prototypes_diverse_16class.pt
if [ ! -f "$PROTO" ]; then
    echo "[$(date)] ERROR: Prototypes not found at $PROTO — run generate_prototypes.py first"
    exit 1
fi
echo "[$(date)] OK: $PROTO ($(du -sh $PROTO | cut -f1))"

echo "[$(date)] Checking for resume checkpoint..."
RESUME_FLAG=""
LATEST=panoptic/checkpoints/full_openvocab_decoder_only/latest.pth
if [ -f "$LATEST" ]; then
    echo "[$(date)] latest.pth found — resuming from: $LATEST ($(du -sh $LATEST | cut -f1))"
    RESUME_FLAG="--resume $LATEST"
else
    echo "[$(date)] No latest.pth — starting fresh"
fi

echo "[$(date)] Disk space: $(df -sh /fs/scratch/SX_ETL3_GenAI_Team/mha2cob 2>/dev/null | tail -1)"
echo "[$(date)] Python: $(python --version)"

echo "[$(date)] === Launching training ==="
python -u panoptic/train/train_openvocab_e2e.py     --config panoptic/configs/full_openvocab_e2e_decoder_only.yaml     $RESUME_FLAG

EXIT_CODE=$?
echo "[$(date)] === Training exited with code $EXIT_CODE ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed — check logs"
    exit $EXIT_CODE
fi

echo "[$(date)] Training complete."
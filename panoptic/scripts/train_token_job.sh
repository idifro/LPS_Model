#!/bin/bash
#BSUB -J panoptic-token-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/token_full_%J.out
#BSUB -e panoptic/logs/token_full_%J.err

# ── Environment setup ──
module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

# Ensure CUDA libraries are visible
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

mkdir -p panoptic/logs panoptic/checkpoints_token_full_v2

# ── Generate text prototypes (if not already cached) ──
if [ ! -f panoptic/clip_prototypes/text_prototypes_vitl14_336.pt ]; then
    echo "Generating text prototypes..."
    python -u panoptic/scripts/generate_text_prototypes.py
fi

# ── Auto-resume ──
RESUME_FLAG=""
LATEST="panoptic/checkpoints_token_full_v2/latest.pth"
if [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_FLAG="--resume $LATEST"
fi

# ── Train (single GPU, full nuScenes, v2 config) ──
python -u panoptic/train/train_panoptic_token.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    $RESUME_FLAG

echo "Training complete."

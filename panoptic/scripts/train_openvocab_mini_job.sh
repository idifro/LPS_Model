#!/bin/bash
#BSUB -J openvocab_mini
#BSUB -o panoptic/logs/openvocab_mini_%J.stdout
#BSUB -e panoptic/logs/openvocab_mini_%J.stderr
#BSUB -q batch_a100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -M 8192
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"

# ─── Open-Vocabulary E2E Panoptic Training — nuScenes Mini ──────────────────
# Class-agnostic masks + CLIP-aligned region features.
# Enables arbitrary text queries at inference time.

module load conda
conda activate opensc_env
module load cuda/11.0.2
module load cudnn/11.0_v8.0
module load gcc/5.4.0
# gcc/5.4.0 provides real libgfortran.so.3 with GFORTRAN_1.0 needed by libopenblas
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libgfortran.so.3)):$LD_LIBRARY_PATH

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=4
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model

mkdir -p panoptic/logs
mkdir -p panoptic/checkpoints_openvocab_mini

echo "=============================================="
echo "  Open-Vocab E2E Panoptic — nuScenes Mini"
echo "  $(date)"
echo "=============================================="

python -u panoptic/train/train_openvocab_e2e.py \
    --config panoptic/configs/mini_openvocab_e2e.yaml

echo "Done: $(date)"

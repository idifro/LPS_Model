#!/bin/bash
#BSUB -J ov_mini_all
#BSUB -o panoptic/logs/ov_mini_all_%J.stdout
#BSUB -e panoptic/logs/ov_mini_all_%J.stderr
#BSUB -q batch_a100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -M 8192
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"

module load conda
conda activate opensc_env
module load cuda/11.0.2
module load cudnn/11.0_v8.0
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libgfortran.so.3)):$LD_LIBRARY_PATH
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=4

module load proxy4server-access/ || true
proxy_on || true

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
mkdir -p panoptic/logs
mkdir -p panoptic/checkpoints/mini_openvocab_all

echo "=============================================="
echo "  Open-Vocab E2E — nuScenes Mini — all"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

python -u panoptic/train/train_openvocab_e2e.py \
    --config panoptic/configs/mini_openvocab_e2e_all.yaml

echo "Done: $(date)"
#!/bin/bash
#BSUB -J precompute-clip
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 16384
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/precompute_clip_%J.out
#BSUB -e panoptic/logs/precompute_clip_%J.err

# Precompute per-voxel CLIP features from camera images
# Estimated: ~6s/sample × 28K samples = ~47 hours (with --resume for restarts)

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs

echo "=== Precomputing CLIP Voxel Features ==="
python -u panoptic/scripts/precompute_clip_voxel_features.py \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --version v1.0-trainval \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features \
    --device cuda \
    --split both \
    --resume

echo "Done."

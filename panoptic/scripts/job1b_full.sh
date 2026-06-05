#!/bin/bash
#BSUB -J job1b-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 36:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/job1b_full_%J.out
#BSUB -e panoptic/logs/job1b_full_%J.err

# Job 1b: Camera CLIP Distillation (L_O + L_V)
# 15 epochs on v1.0-trainval, warm-start from Job 1a best

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/checkpoints_job1b

# Verify prerequisites
for f in panoptic/checkpoints_job1a/best.pth \
         panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
         panoptic/clip_prototypes/negative_bank_200.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

# Check clip features exist
CLIP_DIR=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features/v1.0-trainval
N_FILES=$(ls "$CLIP_DIR"/*.npz 2>/dev/null | wc -l)
if [ "$N_FILES" -lt 1000 ]; then
    echo "ERROR: Only $N_FILES precomputed CLIP files found (need >1000)"
    exit 1
fi
echo "Found $N_FILES precomputed CLIP feature files"

# Auto-resume
RESUME_FLAG=""
LATEST="panoptic/checkpoints_job1b/latest.pth"
if [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_FLAG="--resume $LATEST"
fi

echo "=== Job 1b Full Training ==="
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b.yaml \
    $RESUME_FLAG

echo "Training complete."

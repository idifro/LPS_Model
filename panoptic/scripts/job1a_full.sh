#!/bin/bash
#BSUB -J job1a-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 36:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/job1a_full_%J.out
#BSUB -e panoptic/logs/job1a_full_%J.err

# Job 1a Full: LoRA + SingleCLIPHead (A4) + C1 TokenLoss (216+) + T4 MultiPrompt
# 20 epochs on v1.0-trainval, warm-start from best.pth

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/checkpoints_job1a

# Verify prototypes exist
for f in panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
         panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt.pt \
         panoptic/clip_prototypes/negative_bank_200.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f — run prereq_protos_job.sh first"
        exit 1
    fi
done

# Auto-resume
RESUME_FLAG=""
LATEST="panoptic/checkpoints_job1a/latest.pth"
if [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_FLAG="--resume $LATEST"
fi

echo "=== Job 1a Full Training ==="
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1a.yaml \
    $RESUME_FLAG

echo "Training complete."

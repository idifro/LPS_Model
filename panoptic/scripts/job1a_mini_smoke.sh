#!/bin/bash
#BSUB -J job1a-mini-smoke
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 8192
#BSUB -W 02:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1a_mini_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/job1a_mini_%J.err

# Job 1a Mini Smoke Test (3 epochs on v1.0-mini)
# Gate: no NaN, loss decreasing, LoRA grad norms in [1e-4, 10]

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/checkpoints_job1a_mini

# Verify prototypes exist
for f in panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
         panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt.pt \
         panoptic/clip_prototypes/negative_bank_200.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f — run prereq_protos_job.sh first"
        exit 1
    fi
done

echo "=== Job 1a Mini Smoke Test ==="
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1a_mini.yaml \
    --smoke_test

echo "Smoke test complete."

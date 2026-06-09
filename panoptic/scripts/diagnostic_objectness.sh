#!/bin/bash
#BSUB -J eval-objectness-hyp
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 16384
#BSUB -W 4:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o panoptic/logs/eval_objectness_hypothesis_%J.out
#BSUB -e panoptic/logs/eval_objectness_hypothesis_%J.err

# Diagnostic: Test objectness suppression hypothesis
# Tests Job 1a and Job 1b to see if novel embeddings exist but are suppressed
# Runtime: ~1-2 hours on A100 with mini split, ~4-6 hours with val split

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs

echo "=== Objectness Suppression Hypothesis Test ==="
echo "Testing Job 1a and Job 1b with/without objectness"
echo ""

python -u panoptic/scripts/eval_objectness_hypothesis.py \
    --ckpt_job1a panoptic/checkpoints_job1a/best.pth \
    --cfg_job1a panoptic/configs/job1a.yaml \
    --ckpt_job1b panoptic/checkpoints_job1b/best.pth \
    --cfg_job1b panoptic/configs/job1b.yaml \
    --device cuda \
    --split mini

echo ""
echo "Test complete. Check results above."

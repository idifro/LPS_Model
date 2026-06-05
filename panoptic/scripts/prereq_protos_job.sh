#!/bin/bash
#BSUB -J prereq-protos-224
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 8192
#BSUB -W 01:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/prereq_protos_%J.out
#BSUB -e panoptic/logs/prereq_protos_%J.err

# Generate @224 text prototypes + negative bank (prerequisite for Job 1a)

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/clip_prototypes

echo "=== Generating @224 text prototypes ==="
python -u panoptic/scripts/generate_prototypes_224.py --device cuda

echo "=== Generating negative bank ==="
python -u panoptic/scripts/generate_negative_bank.py --device cuda

echo "=== Verifying outputs ==="
python -c "
import torch
p1 = torch.load('panoptic/clip_prototypes/text_prototypes_vitl14_224.pt')
p2 = torch.load('panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt.pt')
p3 = torch.load('panoptic/clip_prototypes/negative_bank_200.pt')
print(f'Single prompt: {p1[\"embeddings\"].shape}')
print(f'Multi-prompt:  {p2[\"embeddings\"].shape}')
print(f'Negative bank: {p3[\"embeddings\"].shape}')
print('All prototypes generated successfully!')
"

echo "Done."

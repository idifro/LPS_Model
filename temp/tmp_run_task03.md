# Task 0.3 — Run Commands

```bash
# 1. Get GPU node
bsub -Is -q batch_a100 -n 4 -M 8192 -W 12:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash

# 2. Setup environment
module load conda
conda activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

# 3. Run validation
python panoptic/scripts/validate_per_sample_forward.py
```

## Expected output

- Output shape: `[V, 768]` where V ~ 15,000-25,000 voxels
- No NaN/Inf
- Norm range: reasonable positive values
- Forward time: < 1s on A100
- GPU peak memory: ~2,000-3,000 MB
- Final line: `>>> PASS: DisNet produces 768-d features on per-keyframe input.`

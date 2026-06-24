#!/bin/bash
#BSUB -J precompute_v2_pass2
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 16384
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_v2_pass2_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_v2_pass2_%J.err

set -euo pipefail

# Precompute v2 Pass 2: add patch-pooled vox_feats to existing Pass 1 .npz files.
# Required for Phase 10 (L_V ablation B). Can run in parallel with job1b training.
# --vox_feats_only: read existing .npz, compute vox_feats, save back in-place.
# --resume: skip files with n_valid > 0 (already processed). Safe to resubmit.
# Storage estimate: +6 MB/sample × 34k samples ≈ +200 GB on top of Pass 1 (~3 GB).
# Time estimate: ~10-30s/sample (patch extractor × 6 cameras) × 34k ≈ 20-40h.

echo "[$(date)] === Precompute v2 Pass 2 (vox_feats): starting ==="
echo "[$(date)] Host: $(hostname), Job ID: $LSB_JOBID"

echo "[$(date)] Loading modules..."
module load conda
source activate openscene
echo "[$(date)] Conda env: $CONDA_DEFAULT_ENV"

module load cuda/11.0.2
module load cudnn/11.0_v8.0
echo "[$(date)] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4
echo "[$(date)] PYTHONPATH=$PYTHONPATH"

echo "[$(date)] Changing to working directory..."
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
echo "[$(date)] CWD: $(pwd)"
mkdir -p panoptic/logs

DATAROOT=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
OUT_DIR=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2

echo "[$(date)] Verifying Pass 1 output..."
N_PASS1=$(find "$OUT_DIR/v1.0-trainval/" -name "*.npz" 2>/dev/null | wc -l || echo 0)
if [ "$N_PASS1" -lt 28000 ]; then
    echo "[$(date)] ERROR: Only $N_PASS1 Pass 1 .npz files (need >= 28000). Run Pass 1 first."
    exit 1
fi
echo "[$(date)] Found $N_PASS1 Pass 1 .npz files."

N_DONE=$(python -c "
import numpy as np, os, glob
files = glob.glob('$OUT_DIR/v1.0-trainval/*.npz')
done = sum(1 for f in files if int(np.load(f, allow_pickle=False)['n_valid']) > 0)
print(done)
" 2>/dev/null || echo 0)
echo "[$(date)] Already processed (n_valid > 0): $N_DONE / $N_PASS1"

echo "[$(date)] Disk space: $(df -sh /fs/scratch/SX_ETL3_GenAI_Team/mha2cob 2>/dev/null | tail -1)"
echo "[$(date)] Storage note: vox_feats adds ~200 GB. Ensure sufficient space."
echo "[$(date)] Python: $(python --version)"

echo "[$(date)] === Launching Pass 2 ==="
python -u panoptic/scripts/precompute_clip_voxel_features_v2.py \
    --dataroot "$DATAROOT" \
    --version v1.0-trainval \
    --output_dir "$OUT_DIR" \
    --vox_feats_only \
    --split both \
    --resume \
    --device cuda
EXIT_CODE=$?

echo "[$(date)] === Pass 2 exited with code $EXIT_CODE ==="
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Pass 2 failed — check logs"
    exit $EXIT_CODE
fi

N_FINAL=$(python -c "
import numpy as np, glob
files = glob.glob('$OUT_DIR/v1.0-trainval/*.npz')
done = sum(1 for f in files if int(np.load(f, allow_pickle=False)['n_valid']) > 0)
print(done)
" 2>/dev/null || echo "?")
echo "[$(date)] Done. Files with vox_feats (n_valid > 0): $N_FINAL / $N_PASS1"

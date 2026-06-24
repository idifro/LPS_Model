#!/bin/bash
#BSUB -J precompute_v2
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 16384
#BSUB -W 16:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_clip_v2_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_clip_v2_%J.err

set -euo pipefail

# Precompute per-instance crop-CLS CLIP features for job1b-v2 (L_O Pass 1).
# --no_vox_feats: writes zero vox_feats (ablation A); add vox_feats in Pass 2 if needed.
# --resume: safe to resubmit if job is killed (skips already-written .npz files).
# Expected: ~1.2s/sample × 28,130 samples ≈ 9-10h. Wall time 16h gives 6h buffer.

echo "[$(date)] === Precompute v2 (Pass 1): starting ==="
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

echo "[$(date)] Verifying input files..."
PROTO_PATH=panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
if [ ! -f "$PROTO_PATH" ]; then
    echo "[$(date)] ERROR: Missing $PROTO_PATH — run generate_prototypes_v2.py first"
    exit 1
fi
echo "[$(date)] OK: $PROTO_PATH ($(du -sh "$PROTO_PATH" | cut -f1))"

DATAROOT=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
if [ ! -d "$DATAROOT" ]; then
    echo "[$(date)] ERROR: nuScenes dataroot not found: $DATAROOT"
    exit 1
fi
echo "[$(date)] OK: dataroot $DATAROOT"

OUT_DIR=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2
mkdir -p "$OUT_DIR"
echo "[$(date)] Output dir: $OUT_DIR"

N_EXISTING=$(find "$OUT_DIR/v1.0-trainval/" -name "*.npz" 2>/dev/null | wc -l || echo 0)
echo "[$(date)] Existing .npz files (will skip with --resume): $N_EXISTING"

echo "[$(date)] Disk space: $(df -sh /fs/scratch/SX_ETL3_GenAI_Team/mha2cob 2>/dev/null | tail -1)"
echo "[$(date)] Python: $(python --version)"

echo "[$(date)] === Launching precompute ==="
python -u panoptic/scripts/precompute_clip_voxel_features_v2.py \
    --dataroot "$DATAROOT" \
    --version v1.0-trainval \
    --output_dir "$OUT_DIR" \
    --proto_path "$PROTO_PATH" \
    --no_vox_feats \
    --split both \
    --resume \
    --device cuda
EXIT_CODE=$?

echo "[$(date)] === Precompute exited with code $EXIT_CODE ==="
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] ERROR: Precompute failed — check logs"
    exit $EXIT_CODE
fi

N_FINAL=$(find "$OUT_DIR/v1.0-trainval/" -name "*.npz" 2>/dev/null | wc -l || echo 0)
echo "[$(date)] Done. Total .npz files written: $N_FINAL"

#!/bin/bash
# Run all three OpenScene evaluations (fusion, distill, ensemble) sequentially.
# Outputs saved under exp/nusc_mini_openseg_img_viz/

set -e  # stop on first error

EXP=exp/nusc_mini_openseg_img_viz

echo "========================================"
echo "[1/3] Fusion evaluation"
echo "========================================"
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini.yaml \
    TEST.feature_type fusion \
    TEST.save_folder ${EXP}/eval_fusion

echo "========================================"
echo "[2/3] Distill evaluation (pretrained)"
echo "========================================"
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type distill \
    TEST.save_folder ${EXP}/eval_distill

echo "========================================"
echo "[3/3] Ensemble evaluation (pretrained)"
echo "========================================"
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type ensemble \
    TEST.save_folder ${EXP}/eval_ensemble

echo ""
echo "All evaluations complete. Results in ${EXP}/"

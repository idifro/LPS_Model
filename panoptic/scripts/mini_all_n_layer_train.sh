#!/bin/bash
WORKDIR=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
LOG_DIR=$WORKDIR/panoptic/logs/bsub
mkdir -p $LOG_DIR
TRAIN=$WORKDIR/panoptic/train/train_openvocab_e2e.py

echo "Submitting 2 parallel mini training jobs..."

bsub \
  -J "ov_mini_all" \
  -n 8 \
  -gpu "num=1:mode=exclusive_process" \
  -q gpu \
  -R "rusage[mem=32000]" \
  -o $LOG_DIR/mini_all_%J.log \
  -e $LOG_DIR/mini_all_%J.err \
  "cd $WORKDIR && export OMP_NUM_THREADS=8 && python -u $TRAIN --config panoptic/configs/mini_openvocab_e2e_all.yaml"

echo "[SUBMITTED] ov_mini_all"

bsub \
  -J "ov_mini_last_n" \
  -n 8 \
  -gpu "num=1:mode=exclusive_process" \
  -q gpu \
  -R "rusage[mem=32000]" \
  -o $LOG_DIR/mini_last_n_%J.log \
  -e $LOG_DIR/mini_last_n_%J.err \
  "cd $WORKDIR && export OMP_NUM_THREADS=8 && python -u $TRAIN --config panoptic/configs/mini_openvocab_e2e_last_n.yaml"

echo "[SUBMITTED] ov_mini_last_n"
echo "Monitor: bjobs -u \$USER"

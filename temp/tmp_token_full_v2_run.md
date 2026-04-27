# Token Full v2 Run Guide

## Bug Fix Applied
`token_criterion` was not moved to GPU → `log_temperature` (nn.Parameter) stayed on CPU → AMP `unscale_` crashed.
Fix: added `.to(device)` at `train_panoptic_token.py:533`.

## Smoke Test (interactive GPU node)

```bash
# 1. Get interactive GPU
bsub -Is -q batch_a100 -n 4 -M 8192 -W 1:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash

# 2. Setup env
module load conda && source activate openscene
module load cuda/11.0.2 && module load cudnn/11.0_v8.0
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

# 3. Quick smoke test on mini (2-3 min, verifies contrastive loss + aux + differential LR)
python -u panoptic/train/train_panoptic_token.py \
    --config panoptic/configs/mini_token.yaml

# What to verify in output:
#   - "[Token] loss_type=cosine" (mini uses cosine, that's fine)
#   - No CUDA/CPU device errors
#   - Loss decreases over first few steps

# 4. Smoke test with v2 config (contrastive + aux) — runs 1 epoch on trainval then ctrl-C
python -u panoptic/train/train_panoptic_token.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml

# What to verify:
#   - "[Token] loss_type=contrastive, weight=1.0, warmup=500, aux=True"
#   - "[Token] learnable temperature, init=14.2857"  (1/0.07 ≈ 14.29)
#   - "[Optimizer] Differential LR: decoder=1.0e-05, token_head=5.0e-05"
#   - "[WARN] Warm-start checkpoint not found" OR successful load from checkpoints_full/best.pth
#   - token_loss should be ~2-3 (InfoNCE for 16 classes), NOT ~0.02 (that was cosine)
#   - No device mismatch errors
#   - After ~10 steps, Ctrl-C to stop
```

## Full Training Job Submit

```bash
# Make sure checkpoint dir exists
mkdir -p panoptic/checkpoints_token_full_v2

# Submit job
bsub < panoptic/scripts/train_token_job.sh

# Monitor
bjobs -w                          # check job status
bpeek <JOB_ID>                    # peek at live output
tail -f panoptic/logs/token_full_<JOB_ID>.out   # after job starts
```

## Expected Startup Output
```
Device: cuda, world_size: 1
[NuScenesPanoptic] train: 28130 samples from v1.0-trainval
[NuScenesPanoptic] val: 6019 samples from v1.0-trainval
[Backbone] Loaded DisNet from models/nuscenes_openseg.pth.tar, frozen.
[Warm-start] Loaded decoder from panoptic/checkpoints_full/best.pth (PQ=0.473)
[Decoder] 10,921,233 trainable params (token_head: 788,224)
[Token] loss_type=contrastive, weight=1.0, warmup=500, aux=True
[Token] learnable temperature, init=14.2857
[Optimizer] Differential LR: decoder=1.0e-05, token_head=5.0e-05
```

## Key Wandb Metrics to Watch
- `train/loss` — total loss (should start ~35-40, decrease)
- `train/token_loss` — contrastive token loss (~2-3, NOT 0.02)
- `train/temperature` — learnable temp (starts ~14.3, may drift)
- `train/lr` — decoder LR (peaks at 1e-5)
- `train/lr_token` — token head LR (peaks at 5e-5)
- `val/PQ` — linear head PQ (should stay ≥0.46)
- `val/PQ_token` — token head PQ (target: ≥0.45)

## If Full Checkpoint Not Ready Yet
The v2 config warm-starts from `panoptic/checkpoints_full/best.pth`. If the full Phase 4 training hasn't finished yet, the script will warn and start from scratch (much weaker). Wait for Phase 4 to finish first, then submit.

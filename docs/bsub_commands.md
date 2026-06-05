# BSUB Commands Reference

## Pre-requisites

```bash
mkdir -p panoptic/logs panoptic/checkpoints_job1a
```

## Submit Jobs

```bash
# ── Job 1a ──
# Job 1a: LoRA + SingleCLIPHead + C1 + T4 (full trainval, 20 epochs)
bsub < panoptic/scripts/job1a_full.sh

# Job 1a: Mini smoke test
bsub < panoptic/scripts/job1a_mini_smoke.sh

# Prototype generation (prerequisite)
bsub < panoptic/scripts/prereq_protos_job.sh

# ── Job 1b ──
# Precompute CLIP features on mini (~5 min, run interactively)
python panoptic/scripts/precompute_clip_voxel_features.py \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --version v1.0-mini \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features \
    --device cuda --split both

# Job 1b: Mini smoke test (after mini precompute)
python panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_mini.yaml --smoke_test

# Precompute CLIP features on full trainval (~47 hrs, submit as job)
bsub < panoptic/scripts/precompute_clip_features.sh

# Job 1b: Full training (after precompute completes)
bsub < panoptic/scripts/job1b_full.sh
```

## Monitor Jobs

```bash
bjobs                    # list all active jobs
bjobs -l <JOB_ID>       # detailed info for a specific job
bpeek <JOB_ID>          # peek at live stdout
bpeek -f <JOB_ID>       # follow live output (like tail -f)
```

## Job Control

```bash
bkill <JOB_ID>          # kill a job
bstop <JOB_ID>          # suspend a job
bresume <JOB_ID>        # resume a suspended job
```

## Check Output

```bash
# Logs are written to panoptic/logs/
ls panoptic/logs/job1a_full_*.out
cat panoptic/logs/job1a_full_<JOB_ID>.out   # stdout
cat panoptic/logs/job1a_full_<JOB_ID>.err   # stderr
```

## Queue Info

```bash
bqueues                  # list all queues
bqueues batch_a100       # check A100 queue availability
bhosts                   # check host status
```

## Interactive GPU Session (for debugging)

```bash
bsub -Is -q batch_a100 -n 4 -M 8192 -gpu "num=1:mode=exclusive_process" -W 2:00 bash
```

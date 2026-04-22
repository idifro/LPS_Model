# Run Instructions: Panoptic Decoder Training

## Prerequisites

### 1. Install wandb (one-time, on GPU node)
```bash
bsub -Is -q batch_a100 -n 4 -M 8192 -W 1:00 -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash
module load conda && conda activate openscene && module load cuda/11.0.2 && module load cudnn/11.0_v8.0
pip install wandb pyyaml
wandb login
# paste your API key from https://wandb.ai/authorize
exit
```

### 2. Create log directory
```bash
mkdir -p /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs
mkdir -p /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/checkpoints
```

---

## Option A: Interactive (for debugging / smoke test)

```bash
bsub -Is -q batch_a100 -n 4 -M 16384 -W 12:00 -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash
module load conda && conda activate openscene && module load cuda/11.0.2 && module load cudnn/11.0_v8.0
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

# Smoke test: 5 epochs on mini, eval every epoch
python -u panoptic/train/train_panoptic.py \
    --config panoptic/configs/default.yaml \
    --training.epochs=5 \
    --logging.eval_every=1 \
    --logging.log_every=5 \
    --training.mixed_precision=true
```

## Option B: Batch job (for long training)

```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
bsub < panoptic/scripts/train_job.sh
```

Check job status:
```bash
bjobs -l
```

Check logs:
```bash
tail -f panoptic/logs/train_*.out
```

---

## Resume after crash / timeout

Auto-resume is built in. Just resubmit the same job:
```bash
bsub < panoptic/scripts/train_job.sh
```
It auto-detects `interrupted.pth` or `latest.pth` and continues.

Or manually specify:
```bash
python -u panoptic/train/train_panoptic.py \
    --config panoptic/configs/default.yaml \
    --resume panoptic/checkpoints/latest.pth
```

---

## Ablation: override hyperparams from CLI

No need to create a new YAML. Override any value with dot notation:

```bash
# Try higher LR
python -u panoptic/train/train_panoptic.py --training.lr=3e-4

# Try step scheduler instead of cosine warmup
python -u panoptic/train/train_panoptic.py --training.scheduler=step --training.step_size=30

# Try CosineAnnealingLR
python -u panoptic/train/train_panoptic.py --training.scheduler=cosine_annealing

# Disable augmentation
python -u panoptic/train/train_panoptic.py --augmentation.enabled=false

# Disable mixed precision
python -u panoptic/train/train_panoptic.py --training.mixed_precision=false

# Disable wandb
python -u panoptic/train/train_panoptic.py --logging.wandb_enabled=false
```

---

## For full nuScenes (v1.0-trainval)

Create `panoptic/configs/full_nuscenes.yaml` (copy of default.yaml) and change:
```yaml
data:
  dataroot: /path/to/nuscenes/v1.0-trainval
  version: v1.0-trainval
training:
  epochs: 100
  scheduler: cosine_warmup
  warmup_epochs: 2
```

Or override from CLI:
```bash
python -u panoptic/train/train_panoptic.py \
    --data.dataroot=/path/to/nuscenes \
    --data.version=v1.0-trainval \
    --training.epochs=100
```

---

## Monitor training

### WandB dashboard
Go to https://wandb.ai → project `openscene-panoptic`

### GPU usage on node
```bash
nvidia-smi -l 5
```

### Check checkpoint contents
```python
import torch
ckpt = torch.load("panoptic/checkpoints/best.pth", map_location="cpu")
print("Epoch:", ckpt["epoch"])
print("Best PQ:", ckpt["best_pq"])
print("Metrics:", ckpt.get("metrics"))
print("Config:", ckpt["config"]["training"]["scheduler"])
```

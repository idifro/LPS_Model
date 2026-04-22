# OpenScene Full nuScenes Training — 4× A100 Batch Job

## Context

Training the OpenScene 3D distillation model (MinkUNet18A) to match paper metrics (~35.4% distill mIoU) requires the **full nuScenes training split** (~700 scenes) with proper batch size. Interactive jobs (`inter_a100`) have a hard **12-hour ceiling** — not enough for a full run. Solution: use `batch_a100` queue which supports **168 hours (7 days)** walltime and 4× A100 GPUs on a single node, matching the paper's setup.

`distill.py` already has full `mp.spawn`-based single-node multi-GPU support — **no code changes needed**.

---

## Key Facts

### distill.py multi-GPU behaviour
```python
# run/distill.py
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
args.ngpus_per_node = len(args.train_gpu)       # = 4
args.world_size = args.ngpus_per_node * 1       # = 4
mp.spawn(main_worker, nprocs=4, args=(4, args))
# Per-GPU inside main_worker:
args.batch_size = int(args.batch_size / ngpus_per_node)  # config=4 → 1/GPU
args.workers    = int(args.workers    / ngpus_per_node)  # config=4 → 1/GPU
```

**Config `batch_size` = TOTAL across all GPUs.** For 1/GPU × 4 GPUs → set `batch_size: 4`.

### Resume
Checkpoint at `{save_path}/model/model_last.pth.tar` is saved every epoch. Set `resume:` in config to continue.

### DLP batch_a100 queue
- Walltime: up to 168 hours
- Submit: `bsub < script.bsub`
- 4 GPUs: `-gpu "num=4:mode=exclusive_process:mps=no" -R "span[hosts=1]"`

---

## Files

| File | Purpose |
|------|---------|
| `config/nuscenes/ours_openseg_full_4gpu.yaml` | 4-GPU training config |
| `scripts/train_full_nuscenes.bsub` | LSF batch job submission script |

---

## Config: `config/nuscenes/ours_openseg_full_4gpu.yaml`

Key settings:
- `train_gpu: [0, 1, 2, 3]`
- `batch_size: 4` (1 per GPU after internal division)
- `workers: 4` (1 per GPU after internal division)
- `multiprocessing_distributed: True`
- `world_size: 1` (multiplied to 4 inside distill.py)
- `save_path: exp/nusc_full_openseg`
- `vis_pred: False` / `vis_gt: False` (skip viz during training eval passes)

---

## Submit

```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
bsub < scripts/train_full_nuscenes.bsub
```

## Monitor

```bash
bjobs
bpeek <JOBID>
tail -f exp/nusc_full_openseg/train_<JOBID>.stdout
```

## Resume (if job ends before training completes)

1. Edit `config/nuscenes/ours_openseg_full_4gpu.yaml`:
   ```yaml
   resume: exp/nusc_full_openseg/model/model_last.pth.tar
   ```
2. Resubmit: `bsub < scripts/train_full_nuscenes.bsub`

---

## Prerequisites

The full nuScenes dataset must be preprocessed first:
- `data/nuscenes_3d/train/` — ~700 `.pth` scene files
- `data/nuscenes_multiview_openseg/` — fused OpenSeg `.pt` features for all train scenes

The mini dataset has only 10 scenes. See `setup_and_run.md` for preprocessing steps.

---

## Time Estimate

| Config | Iters/epoch | Time/epoch | 5 epochs total |
|--------|-------------|------------|----------------|
| 4 GPU × batch=1, 700 scenes | ~875 | ~2.4 hrs | ~12 hrs |
| 1 GPU × batch=1, 700 scenes | ~3500 | ~9.7 hrs | ~48 hrs |

---

## Post-Training Evaluation

```bash
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_full_4gpu.yaml \
    TEST.feature_type distill \
    TEST.save_folder exp/nusc_full_openseg/eval_distill \
    TEST.model_path exp/nusc_full_openseg/model/model_best.pth.tar
```

Expected: distill mIoU ~35.4% (paper result).

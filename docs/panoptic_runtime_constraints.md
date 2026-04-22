# Panoptic Runtime Constraints

Frozen environment invariants for the panoptic segmentation project.

## Cluster

- **System**: DLP HPC, LSF scheduler (`bsub`)
- **OS**: RHEL 9 (Linux 5.14.0)
- **Scratch**: `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene`

## GPU

- **Type**: NVIDIA A100 96 GB (SXM4)
- **Mode**: `exclusive_process` (no MIG, no MPS)
- **Queue (interactive)**: `bsub -Is -q batch_a100 -n 4 -M 8192 -W 12:00 -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash`
- **Queue (batch)**: `bsub < submit_training.bsub` (up to 48h wall time)

## Environment

- **Conda env**: `openscene`
- **Python**: 3.8
- **PyTorch**: 1.7.1+cu110
- **MinkowskiEngine**: installed in conda env
- **PYTHONPATH**: `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene`

### Module load sequence (required before every GPU session)

```bash
module load conda
conda activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=16
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
```

### Note for Claude Code agent

Claude Code **cannot run GPU scripts**. All GPU-dependent scripts must be given to the user as commands. The user runs them on a GPU node and reports output back.

## Hard Constraints

| Constraint | Value | Reason |
|---|---|---|
| `num_workers` | **0** | Shared memory prefetch overflow causes hangs on this cluster |
| `batch_size` | **1–2** | ME memory fragmentation + frozen backbone memory |
| Checkpoint resume | **Required** | ME fragmentation causes OOM on long runs; segment into 5–10 epoch chunks |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Recommended for reducing ME fragmentation |

## Checkpoint

- **Path**: `models/nuscenes_openseg.pth.tar`
- **Variant**: OpenSeg (ViT-L/14 CLIP)
- **Output dim**: 768
- **Parameters**: 15,575,696
- **State dict**: uses `net3d.*` prefix → load via `model.load_state_dict(state)`
- **Architecture**: MinkUNet18A, `in_channels=3`, `out_channels=768`

## Data

- **nuScenes mini**: `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini`
- **Version**: `v1.0-mini` (10 scenes, ~404 samples)
- **Panoptic labels**: available at `panoptic/v1.0-mini/{token}_panoptic.npz`
- **Panoptic decoding**: `semantic = pan // 1000`, `instance = pan % 1000`

## Baseline Commit

- **Branch**: `feature/panoptic-openscene-phaseA`
- **Frozen from**: `55e0f85b2cb47f3b7266306d1db212f71a7b59cf` on `main`

## Validated Backbone Profile (Task 0.3 — single keyframe)

| Metric | Measured Value |
|---|---|
| Voxels per sweep | 20,688 (at 0.05m voxel size) |
| DisNet output shape | [20688, 768] |
| Forward time | 37 ms |
| GPU peak memory | 207 MB |
| NaN/Inf | None |
| Norm range | [10.05, 924.98], mean 45.0 |

## Memory Envelope (estimated)

| Scenario | Peak GPU (MB) |
|---|---|
| Inference (frozen DisNet only) | **207** (measured) |
| Inference (full pipeline + camera) | ~6,000–8,000 |
| Training (decoder + adapter + token head) | ~20,000–25,000 |
| Training (+ EVA-CLIP camera encoder) | ~30,000–40,000 |

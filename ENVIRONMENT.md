# OpenScene Environment Setup

## Prerequisites
- Linux system (tested on RHEL 9 / x86_64)
- CUDA 11.0 compatible GPU (tested on NVIDIA A100)
- Conda (via `module load conda` on HPC)
- OpenBLAS (for MinkowskiEngine sparse convolution backend)

## Full Setup (from scratch)

```bash
# 1. Load conda
module load conda

# 2. Create environment
conda create -n opensc_env python=3.8 -y
conda activate opensc_env

# 3. Install PyTorch 1.7.1 with CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 4. Install MinkowskiEngine build dependencies
#    On HPC without sudo, use conda for OpenBLAS:
conda install openblas-devel -c anaconda

# 5. Install MinkowskiEngine (requires CUDA toolkit in PATH)
module load cuda/11.0.2
module load cudnn/11.0_v8.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --install-option="--force_cuda" \
    --install-option="--blas=openblas"

# 6. Install remaining dependencies
pip install -r requirements.txt

# 7. Install CLIP (OpenAI)
pip install git+https://github.com/openai/CLIP.git

# 8. (Optional) Install TensorFlow for OpenSeg multi-view feature fusion
pip install tensorflow==2.12.0

# 9. Set environment variables
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=16
```

## Activate Existing Environment

```bash
module load conda
conda activate opensc_env
module load cuda/11.0.2
module load cudnn/11.0_v8.0
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
export OMP_NUM_THREADS=16
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
```

## GPU Job Submission (A100)

```bash
# Interactive session
bsub -Is -q batch_a100 -n 4 -M 8192 -W 12:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash

# Batch job (long training)
bsub -q batch_a100 -n 4 -M 8192 -W 48:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" \
  -o logs/train_%J.out -e logs/train_%J.err \
  < scripts/train_job.sh
```

## Key Version Constraints

| Package | Version | Why |
|---------|---------|-----|
| Python | 3.8 | MinkowskiEngine + torch 1.7 compatibility |
| PyTorch | 1.7.1+cu110 | MinkowskiEngine 0.5.4 built against torch C++ ABI ≤1.7 |
| CUDA | 11.0 | Must match PyTorch cu110 build |
| MinkowskiEngine | 0.5.4 | Sparse 3D convolution backbone (built from source with OpenBLAS) |
| OpenBLAS | (conda) | Required by MinkowskiEngine for sparse matrix ops |
| NumPy | 1.23.5 | Last version supporting py3.8 + torch 1.7 |
| TensorFlow | 2.12.0 | OpenSeg SavedModel inference (feature fusion only) |
| CLIP | 1.0 (git) | Text/image embeddings for open-vocab classification |

## Troubleshooting

### MinkowskiEngine build fails
- Ensure `nvcc --version` shows 11.0 (not system CUDA)
- Ensure `CUDA_HOME` is set: `export CUDA_HOME=/path/to/cuda-11.0`
- If OpenBLAS not found: `conda install openblas-devel -c anaconda`

### "libcudart.so not found" at runtime
- Load CUDA module before running: `module load cuda/11.0.2`

### PyTorch doesn't see GPU
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure you're on a GPU node (not login node)

### num_workers > 0 crashes
- Known issue with MinkowskiEngine memory fragmentation
- Always use `num_workers=0` in DataLoaders

## References
- [OpenScene official installation](https://github.com/pengsongyou/openscene/blob/main/installation.md)
- [MinkowskiEngine docs](https://nvidia.github.io/MinkowskiEngine/)

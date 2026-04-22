# Manual Download Instructions — Pretrained Weights

HPC firewalls block direct URL downloads. Follow these steps to manually obtain the pretrained model.

---

## Pretrained nuScenes OpenSeg Model

**URL**: `https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar`
**Place at**: `models/nuscenes_openseg.pth.tar`

### 1. Download (on a machine with internet access)

```bash
wget https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar
```

### 2. Transfer to HPC

```bash
scp nuscenes_openseg.pth.tar <your_user>@<hpc_login_node>:/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/models/
```

### 3. Verify

```bash
ls -lh models/nuscenes_openseg.pth.tar   # should exist and be ~200MB
python -c "
import torch
ckpt = torch.load('models/nuscenes_openseg.pth.tar', map_location='cpu')
print('Keys:', list(ckpt.keys()))          # → ['state_dict', 'epoch', ...]
print('Epoch:', ckpt.get('epoch', 'N/A'))
"
```

---

## Running Evaluations with Local Weights

All commands use `TEST.model_path models/nuscenes_openseg.pth.tar` to override the URL in the config.

### Distill Evaluation

```bash
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type distill \
    TEST.split val \
    TEST.save_folder exp/nuscenes_mini_openseg/eval_pretrained_distill \
    TEST.test_workers 0 \
    TEST.test_repeats 1 \
    TEST.model_path models/nuscenes_openseg.pth.tar
```

### Ensemble Evaluation

```bash
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type ensemble \
    TEST.split val \
    TEST.save_folder exp/nuscenes_mini_openseg/eval_pretrained_ensemble \
    TEST.test_workers 0 \
    TEST.test_repeats 1 \
    TEST.model_path models/nuscenes_openseg.pth.tar
```

---

## Other Available Pretrained Models

| Dataset | Feature Extractor | URL |
|---------|------------------|-----|
| nuScenes | OpenSeg | `https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar` |
| ScanNet | OpenSeg | `https://cvg-data.inf.ethz.ch/openscene/models/scannet_openseg.pth.tar` |
| Matterport | OpenSeg | `https://cvg-data.inf.ethz.ch/openscene/models/matterport_openseg.pth.tar` |
| nuScenes | LSeg | `https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_lseg.pth.tar` |
| ScanNet | LSeg | `https://cvg-data.inf.ethz.ch/openscene/models/scannet_lseg.pth.tar` |
| Matterport | LSeg | `https://cvg-data.inf.ethz.ch/openscene/models/matterport_lseg.pth.tar` |

Download any of these using the same `wget` + `scp` pattern above, then pass the local path via `TEST.model_path`.

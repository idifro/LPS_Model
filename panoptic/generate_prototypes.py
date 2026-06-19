"""Generate CLIP text prototypes for nuScenes 16 classes.
Run once before training:
    python panoptic/generate_prototypes.py
"""
import sys
import types

# ── Stub torchvision BEFORE any import that pulls it ──
class _Callable:
    def __init__(self, *a, **kw): pass
    def __call__(self, x): return x

for mod in ['torchvision','torchvision.transforms','torchvision.transforms.functional',
            'torchvision.ops','torchvision.ops.misc']:
    sys.modules[mod] = types.ModuleType(mod)
for cls in ['Compose','Resize','CenterCrop','ToTensor','Normalize']:
    setattr(sys.modules['torchvision.transforms'], cls, _Callable)
setattr(sys.modules['torchvision.ops.misc'], 'FrozenBatchNorm2d', _Callable)
sys.modules['torchvision.transforms'].InterpolationMode = types.SimpleNamespace(BICUBIC=2)
sys.modules['torchvision.transforms'].functional = sys.modules['torchvision.transforms.functional']

import os
import torch
import torch.nn.functional as F
import clip

# Patch _transform so clip.load() doesn't crash on Resize
clip.clip._transform = lambda n: _Callable()

NUSCENES_16_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction vehicle",
    "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
    "driveable surface", "other flat", "sidewalk", "terrain", "manmade", "vegetation",
]

TEMPLATES = [
    "a {} in a driving scene",
    "a photo of a {}",
    "a lidar point cloud of a {}",
    "a {} on the road",
    "a {} detected by lidar",
    "a 3D scan of a {}",
    "a {} in autonomous driving",
]

OUT_PATH = "panoptic/clip_prototypes/text_prototypes_diverse_16class.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, _ = clip.load("ViT-L/14@336px", device=device)
    model.eval()
    print("CLIP loaded: ViT-L/14@336px")

    all_prototypes = []
    with torch.no_grad():
        for name in NUSCENES_16_NAMES:
            prompts = [t.format(name) for t in TEMPLATES]
            tokens = clip.tokenize(prompts).to(device)
            feats = model.encode_text(tokens).float()
            feats = F.normalize(feats, dim=-1)
            avg = F.normalize(feats.mean(0, keepdim=True), dim=-1).squeeze(0)
            all_prototypes.append(avg)
            print(f"  {name}: done")

    prototypes = torch.stack(all_prototypes)  # (16, 768)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(prototypes.cpu(), OUT_PATH)
    print(f"\nSaved prototypes {prototypes.shape} → {OUT_PATH}")

if __name__ == "__main__":
    main()
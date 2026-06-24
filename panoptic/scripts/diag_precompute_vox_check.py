import numpy as np, glob, torch
import torch.nn.functional as F

proto_data = torch.load("panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt")
protos = F.normalize(proto_data["embeddings"].float(), dim=-1).numpy()  # (16, 768)

CLASS_NAMES = ["barrier","bicycle","bus","car","construction_vehicle",
               "motorcycle","pedestrian","traffic_cone","trailer","truck",
               "driveable_surface","other_flat","sidewalk","terrain","manmade","vegetation"]

vox_margins = {i: [] for i in range(16)}

for npz_path in glob.glob("/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-mini/*.npz"):
    d = np.load(npz_path, allow_pickle=False)
    vox_feats = d["vox_feats"].astype(np.float32)      # (4096, 768)
    vox_inst  = d["vox_gt_instance_id"]                 # (4096,) int
    obj_cls   = d["obj_classes"]                        # (N_gt,)
    n_valid   = int(d["n_valid"])
    if n_valid == 0:
        continue
    # vox_feats is (4096, 768) — sampled; n_valid counts valid voxels across the
    # full point cloud and can exceed 4096. Iterate all 4096 entries, skip zero vectors.
    norms = np.linalg.norm(vox_feats, axis=1)
    sims = vox_feats @ protos.T                         # (4096, 16)
    for vi in range(len(vox_inst)):
        if norms[vi] < 0.1:                             # zero = no projection
            continue
        if vox_inst[vi] < 0 or vox_inst[vi] >= len(obj_cls):
            continue
        cls_id = int(obj_cls[vox_inst[vi]])
        sim_own  = float(sims[vi, cls_id])
        tmp = sims[vi].copy(); tmp[cls_id] = -1.0
        sim_base = float(tmp.max())
        vox_margins[cls_id].append(sim_own - sim_base)

print("Vox-margin per class (mean ± std):")
for cls_id, margins in vox_margins.items():
    if margins:
        arr = np.array(margins)
        print(f"  [{cls_id:2d}] {CLASS_NAMES[cls_id]:<22}: mean={arr.mean():+.4f}  std={arr.std():.4f}  n={len(arr)}")
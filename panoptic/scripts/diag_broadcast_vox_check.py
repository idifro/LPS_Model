"""Diagnostic: validate broadcast L_V targets from existing obj_feats.

For each sampled voxel in the mini .npz cache, looks up the GT instance's
precomputed obj_feats (crop-CLS) as the broadcast target and evaluates whether
it constitutes a valid L_V teacher.

What this confirms:
  1. vox_gt_instance_id correctly maps sampled voxels to GT instances
  2. THING broadcast targets have positive margin → valid L_V teacher (no Pass 2 needed)
  3. STUFF broadcast targets have negative/near-zero margin → CASD needed for stuff
  4. Coverage: how many of the 4096 voxels per sample have a valid broadcast target

Usage:
    python panoptic/scripts/diag_broadcast_vox_check.py
    python panoptic/scripts/diag_broadcast_vox_check.py \\
        --feat_dir /path/to/clip_features_v2/v1.0-mini \\
        --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
"""
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]
NOVEL_IDS = {2, 5, 6, 15}          # bus, motorcycle, pedestrian, vegetation (0-indexed)
THING_IDS = set(range(10))          # 0-indexed 0–9
STUFF_IDS = set(range(10, 16))      # 0-indexed 10–15


def run(feat_dir, proto_path, max_files=None):
    proto_data = torch.load(proto_path, map_location="cpu")
    protos = F.normalize(proto_data["embeddings"].float(), dim=-1).numpy()  # (16, 768)

    paths = sorted(glob.glob(f"{feat_dir}/*.npz"))
    if max_files:
        paths = paths[:max_files]
    print(f"Scanning {len(paths)} samples from {feat_dir}\n")

    # Per-class accumulators
    margins      = {i: [] for i in range(16)}   # broadcast margin = sim_own - sim_best_other
    coverage_cnt = {i: 0  for i in range(16)}   # voxels with valid broadcast target
    total_cnt    = {i: 0  for i in range(16)}   # all voxels assigned to class (valid or not)
    obj_valid_cnt= {i: 0  for i in range(16)}   # GT instances with obj_valid=True

    n_samples_with_novel = {i: 0 for i in NOVEL_IDS}

    for path in paths:
        try:
            d = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue

        obj_feats   = d["obj_feats"].astype(np.float32)    # (N_inst, 768)
        obj_classes = d["obj_classes"].astype(np.int32)    # (N_inst,)  0-indexed
        obj_valid   = d["obj_valid"].astype(bool)          # (N_inst,)
        vox_inst    = d["vox_gt_instance_id"]              # (4096,)  -1 = unassigned

        n_inst = len(obj_classes)
        for k in range(n_inst):
            if obj_valid[k]:
                obj_valid_cnt[int(obj_classes[k])] += 1

        # Check which novel classes appear in this sample
        present_cls = set(int(c) for c in obj_classes)
        for nid in NOVEL_IDS:
            if nid in present_cls:
                n_samples_with_novel[nid] += 1

        # Per-voxel broadcast evaluation
        for vi in range(len(vox_inst)):
            gi = int(vox_inst[vi])
            if gi < 0 or gi >= n_inst:
                continue                       # unassigned voxel

            cls_id = int(obj_classes[gi])
            total_cnt[cls_id] += 1

            if not obj_valid[gi]:
                continue                       # instance has no valid crop-CLS

            target = obj_feats[gi]             # (768,) — broadcast target
            norm = float(np.linalg.norm(target))
            if norm < 0.1:
                continue

            coverage_cnt[cls_id] += 1

            # Cosine sim to all 16 text prototypes
            t_norm = target / norm
            sims = t_norm @ protos.T           # (16,)
            sim_own = float(sims[cls_id])
            tmp = sims.copy()
            tmp[cls_id] = -1.0
            sim_best_other = float(tmp.max())
            margins[cls_id].append(sim_own - sim_best_other)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"{'Class':<22} {'Type':<6} {'Novel':<6} "
          f"{'Vox/sample':>10} {'Coverage%':>10} {'ObjValid':>9} "
          f"{'Mean margin':>12} {'Std':>7} {'Min':>7} {'Max':>7}")
    print("-" * 100)

    n_samples = len(paths)
    for cls_id in range(16):
        cname  = CLASS_NAMES[cls_id]
        ctype  = "THING" if cls_id in THING_IDS else "STUFF"
        novel  = "NOVEL" if cls_id in NOVEL_IDS else "base"

        vox_per_sample = total_cnt[cls_id] / max(n_samples, 1)
        cov_pct = (coverage_cnt[cls_id] / max(total_cnt[cls_id], 1)) * 100.0

        if margins[cls_id]:
            arr = np.array(margins[cls_id])
            m, s, mn, mx = arr.mean(), arr.std(), arr.min(), arr.max()
            margin_str = f"{m:+.4f}  {s:.4f}  {mn:+.4f}  {mx:+.4f}"
        else:
            margin_str = "     N/A"

        print(f"  [{cls_id:2d}] {cname:<18} {ctype:<6} {novel:<6} "
              f"{vox_per_sample:10.1f} {cov_pct:9.1f}% "
              f"{obj_valid_cnt[cls_id]:9d}  {margin_str}")

    print()
    print("Novel class sample presence:")
    for nid in sorted(NOVEL_IDS):
        print(f"  {CLASS_NAMES[nid]:<20}: present in {n_samples_with_novel[nid]}/{n_samples} samples "
              f"({100*n_samples_with_novel[nid]/max(n_samples,1):.1f}%)")

    print()
    print("═" * 60)
    print("INTERPRETATION:")
    thing_novel_pass = all(
        np.mean(margins[i]) > 0.0 if margins[i] else False
        for i in NOVEL_IDS & THING_IDS
    )
    stuff_novel_negative = all(
        np.mean(margins[i]) < 0.05 if margins[i] else True
        for i in NOVEL_IDS & STUFF_IDS
    )
    print(f"  THING novel broadcast margins positive : {'✅ PASS' if thing_novel_pass else '❌ FAIL'}")
    print(f"  STUFF novel (veg) margin < 0.05        : {'✅ CASD needed' if stuff_novel_negative else '⚠️  revisit — veg may not need CASD'}")

    if thing_novel_pass:
        print("\n  ✅ Broadcast from obj_feats is a valid L_V teacher for THING voxels.")
        print("     No Pass 2 precompute needed for L_V ablation B.")
    else:
        print("\n  ❌ Broadcast margins not positive — check vox_gt_instance_id mapping.")

    if stuff_novel_negative:
        print("  ✅ Vegetation broadcast margin low/negative → online CASD required for stuff.")
        print("     Confirm: skip separate ablation B, combine broadcast+CASD in job1c.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir",
        default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-mini")
    parser.add_argument("--proto_path",
        default="panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt")
    parser.add_argument("--max_files", type=int, default=None,
        help="Limit to first N files for a quick check")
    args = parser.parse_args()
    run(args.feat_dir, args.proto_path, args.max_files)

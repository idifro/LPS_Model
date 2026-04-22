"""Phase 1 verification — validate NuScenesPanoptic dataset and collate.

Checks: dataset length, sample fields, shape consistency, panoptic decoding,
camera paths, and collate function. Does NOT require GPU.
"""
import sys
import os
import random

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

import torch
from torch.utils.data import DataLoader
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, NUSCENES_CLASS_NAMES, IGNORE_LABEL,
)
from panoptic.dataset.collate import panoptic_collate

DATAROOT = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini"
VERSION = "v1.0-mini"


def check_sample(ds, idx):
    """Validate a single sample."""
    s = ds[idx]
    V = s["num_voxels"]
    N = s["num_points"]
    M = s["gt_classes"].shape[0]

    print(f"\n--- Sample {idx} (token={s['token'][:12]}...) ---")
    print(f"  Points: {N}, Voxels: {V}, GT masks: {M}")
    print(f"  vox_coords: {list(s['vox_coords'].shape)}, dtype={s['vox_coords'].dtype}")
    print(f"  vox_feats:  {list(s['vox_feats'].shape)}, dtype={s['vox_feats'].dtype}")
    print(f"  gt_masks:   {list(s['gt_masks'].shape)}, dtype={s['gt_masks'].dtype}")
    print(f"  gt_classes: {list(s['gt_classes'].shape)}, dtype={s['gt_classes'].dtype}")
    print(f"  gt_sem:     {list(s['gt_sem'].shape)}, dtype={s['gt_sem'].dtype}")
    print(f"  inv_idx:    {list(s['inv_idx'].shape)}, dtype={s['inv_idx'].dtype}")

    # Shape checks
    assert s["vox_coords"].shape == (V, 3), f"vox_coords shape mismatch"
    assert s["vox_feats"].shape == (V, 3), f"vox_feats shape mismatch"
    assert s["gt_masks"].shape[1] == V, f"gt_masks V dim mismatch: {s['gt_masks'].shape[1]} != {V}"
    assert s["gt_classes"].shape[0] == M, f"gt_classes count mismatch"
    assert s["gt_sem"].shape[0] == V, f"gt_sem shape mismatch"
    assert s["inv_idx"].shape[0] == N, f"inv_idx shape mismatch"
    assert s["inv_idx"].max() < V, f"inv_idx out of bounds: max={s['inv_idx'].max()} >= V={V}"

    # Semantic range
    valid_sem = s["gt_sem"][s["gt_sem"] != IGNORE_LABEL]
    if len(valid_sem) > 0:
        assert valid_sem.min() >= 0, f"Semantic min < 0: {valid_sem.min()}"
        assert valid_sem.max() < NUM_CLASSES, f"Semantic max >= 16: {valid_sem.max()}"
        unique_sem = valid_sem.unique().tolist()
        class_names = [NUSCENES_CLASS_NAMES[c] for c in unique_sem]
        print(f"  Semantic classes present: {class_names}")
    else:
        print(f"  WARNING: no valid semantic labels in this sample")

    # GT class range
    if M > 0:
        assert s["gt_classes"].min() >= 0, f"gt_classes min < 0"
        assert s["gt_classes"].max() < NUM_CLASSES, f"gt_classes max >= 16"
        # Count things vs stuff
        thing_count = (s["gt_classes"] < 10).sum().item()
        stuff_count = (s["gt_classes"] >= 10).sum().item()
        print(f"  Thing masks: {thing_count}, Stuff masks: {stuff_count}")
        # Check masks are not all-zero
        mask_sums = s["gt_masks"].sum(dim=1)
        assert mask_sums.min() > 0, "Found all-zero GT mask"
        print(f"  Mask sizes: min={mask_sums.min().item()}, max={mask_sums.max().item()}, mean={mask_sums.float().mean().item():.1f} voxels")

    # Camera paths
    assert len(s["cam_paths"]) == 6, f"Expected 6 cameras, got {len(s['cam_paths'])}"
    for cp in s["cam_paths"]:
        assert os.path.exists(cp), f"Camera image missing: {cp}"
    print(f"  All 6 camera images exist")

    return True


def check_collate(ds):
    """Test collate with batch_size=2."""
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=panoptic_collate)
    batch = next(iter(loader))

    print(f"\n--- Collate check (batch_size=2) ---")
    print(f"  coords: {list(batch['coords'].shape)}")
    print(f"  feats:  {list(batch['feats'].shape)}")
    print(f"  num_voxels: {batch['num_voxels']}")
    print(f"  gt_masks: {len(batch['gt_masks'])} samples")
    print(f"  tokens: {batch['tokens']}")

    total_V = sum(batch["num_voxels"])
    assert batch["coords"].shape[0] == total_V, "coords batch size mismatch"
    assert batch["feats"].shape[0] == total_V, "feats batch size mismatch"
    assert batch["coords"].shape[1] == 4, "coords should have batch_idx column"

    # Check batch indices
    batch_ids = batch["coords"][:, 0].unique().tolist()
    assert batch_ids == [0, 1], f"Unexpected batch indices: {batch_ids}"
    print(f"  Batch indices correct: {batch_ids}")

    return True


def main():
    print("=" * 60)
    print("Phase 1 Verification — NuScenesPanoptic Dataset")
    print("=" * 60)

    # Train split
    ds_train = NuScenesPanoptic(dataroot=DATAROOT, version=VERSION, split="train")
    ds_val = NuScenesPanoptic(dataroot=DATAROOT, version=VERSION, split="val")
    print(f"\nTrain samples: {len(ds_train)}")
    print(f"Val samples:   {len(ds_val)}")

    # Check 10 random train samples
    indices = random.sample(range(len(ds_train)), min(10, len(ds_train)))
    for idx in indices:
        ok = check_sample(ds_train, idx)
        assert ok

    # Check first val sample
    check_sample(ds_val, 0)

    # Collate test
    check_collate(ds_train)

    print("\n" + "=" * 60)
    print(">>> ALL CHECKS PASSED — Phase 1 dataset is valid.")
    print("=" * 60)


if __name__ == "__main__":
    main()

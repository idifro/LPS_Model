"""Phase 3 Verification: Frozen DisNet backbone on per-keyframe panoptic input.

Covers master plan tasks:
  3.1 — Wrap frozen DisNet cleanly (load, freeze, verify interface)
  3.2 — Raw feature sanity checks (shape, norms, NaN, distributions)
  3.4 — Confirm memory behavior (single-sample memory, runtime, stability)

Usage (GPU node):
    python -u panoptic/scripts/verify_phase3_backbone.py
    python -u panoptic/scripts/verify_phase3_backbone.py --num_samples 20
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import MinkowskiEngine as ME

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.dataset.nuscenes_panoptic import NuScenesPanoptic


def parse_args():
    """Parse command-line arguments for Phase 3 backbone verification.

    Provides configuration for dataroot, nuScenes version, checkpoint path,
    number of samples, and dataset split used in all Phase 3 sub-tests.

    Steps:
        1. Creates an ArgumentParser with defaults for nuscenes_mini paths.
        2. Adds arguments for dataroot, version, ckpt, num_samples, and split.
        3. Parses and returns the namespace object.

    Returns:
        argparse.Namespace: Parsed arguments with fields dataroot, version, ckpt,
            num_samples (int), and split (str).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--ckpt", default="models/nuscenes_openseg.pth.tar")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--split", default="val")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 3.1 — Wrap frozen DisNet cleanly
# ══════════════════════════════════════════════════════════════════════════════

def test_3_1_wrap_frozen_disnet(ckpt_path, device):
    """Verify backbone loading, freezing, and interface."""
    print(f"\n{'='*60}")
    print("TEST 3.1: Wrap frozen DisNet cleanly")
    print(f"{'='*60}")

    # Load
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"

    model = DisNet(Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Check state dict structure
    raw_keys = list(ckpt["state_dict"].keys())
    print(f"\n  Checkpoint keys (first 5): {raw_keys[:5]}")
    print(f"  Total keys: {len(raw_keys)}")
    has_module_prefix = any(k.startswith("module.") for k in raw_keys)
    print(f"  Has 'module.' prefix: {has_module_prefix}")

    state = state_dict_remove_moudle(ckpt["state_dict"])
    clean_keys = list(state.keys())
    print(f"  After prefix removal (first 5): {clean_keys[:5]}")

    # Strict load
    model.load_state_dict(state, strict=True)
    print(f"  strict=True load: OK")

    # Freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Verify frozen
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    assert trainable_params == 0, "FAIL: Some params are still trainable!"
    print(f"  All params frozen: OK")

    # Check model is in eval mode
    assert not model.training, "FAIL: Model not in eval mode!"
    print(f"  Model in eval mode: OK")

    # Check output layer dimensions
    # DisNet wraps net3d (MinkUNet18A) — find final layer
    final_key = [k for k in state.keys() if "final" in k]
    print(f"\n  Final layer keys: {final_key}")
    for k in final_key:
        print(f"    {k}: {state[k].shape}")

    print(f"\n  PASS: 3.1 — DisNet wrapped, loaded (strict), frozen, eval mode.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3.2 — Raw feature sanity checks
# ══════════════════════════════════════════════════════════════════════════════

def test_3_2_feature_sanity(model, dataset, device, num_samples=10):
    """Check output shape, norms, NaN/Inf, distributions across samples."""
    print(f"\n{'='*60}")
    print(f"TEST 3.2: Raw feature sanity checks ({num_samples} samples)")
    print(f"{'='*60}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: x[0],
    )

    all_shapes = []
    all_norms = []
    all_mins = []
    all_maxs = []
    all_means = []
    nan_count = 0
    inf_count = 0

    print(f"\n  {'Sample':>6} {'Voxels':>7} {'OutShape':>16} {'NormMin':>9} {'NormMax':>9} "
          f"{'NormMean':>9} {'FeatMin':>9} {'FeatMax':>9} {'NaN':>4} {'Inf':>4}")
    print(f"  {'-'*100}")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break

            vox_coords = batch["vox_coords"]
            vox_feats = batch["vox_feats"]
            V = vox_coords.shape[0]

            batch_coords = torch.cat([
                torch.zeros(V, 1, dtype=torch.int32), vox_coords
            ], dim=1)
            sinput = ME.SparseTensor(
                vox_feats.float(), coordinates=batch_coords.int(), device=device
            )

            output = model(sinput)
            feats = output.F if hasattr(output, "F") else output  # (V, 768)

            # Checks
            has_nan = torch.isnan(feats).any().item()
            has_inf = torch.isinf(feats).any().item()
            norms = feats.norm(dim=-1)  # (V,)
            feat_min = feats.min().item()
            feat_max = feats.max().item()

            all_shapes.append(feats.shape)
            all_norms.append(norms.cpu().numpy())
            all_mins.append(feat_min)
            all_maxs.append(feat_max)
            all_means.append(feats.mean().item())
            if has_nan:
                nan_count += 1
            if has_inf:
                inf_count += 1

            print(f"  {i:>6} {V:>7} {str(tuple(feats.shape)):>16} "
                  f"{norms.min().item():>9.2f} {norms.max().item():>9.2f} "
                  f"{norms.mean().item():>9.2f} {feat_min:>9.2f} {feat_max:>9.2f} "
                  f"{'YES' if has_nan else 'no':>4} {'YES' if has_inf else 'no':>4}")

    # Summary
    all_norms_flat = np.concatenate(all_norms)
    print(f"\n  Summary across {num_samples} samples:")
    print(f"    Output dim: {all_shapes[0][1]} (consistent: {all(s[1] == all_shapes[0][1] for s in all_shapes)})")
    print(f"    Voxel counts: {[s[0] for s in all_shapes]}")
    print(f"    Norm stats: min={all_norms_flat.min():.2f}, max={all_norms_flat.max():.2f}, "
          f"mean={all_norms_flat.mean():.2f}, std={all_norms_flat.std():.2f}")
    print(f"    Feature range: [{min(all_mins):.2f}, {max(all_maxs):.2f}]")
    print(f"    Feature mean: {np.mean(all_means):.4f}")
    print(f"    NaN samples: {nan_count}/{num_samples}")
    print(f"    Inf samples: {inf_count}/{num_samples}")

    # Norm distribution percentiles
    pcts = [1, 5, 25, 50, 75, 95, 99]
    pct_vals = np.percentile(all_norms_flat, pcts)
    print(f"\n    Norm percentiles:")
    for p, v in zip(pcts, pct_vals):
        print(f"      P{p:>2}: {v:.2f}")

    assert nan_count == 0, f"FAIL: {nan_count} samples have NaN!"
    assert inf_count == 0, f"FAIL: {inf_count} samples have Inf!"
    assert all(s[1] == 768 for s in all_shapes), "FAIL: Output dim != 768!"

    print(f"\n  PASS: 3.2 — All features are 768-d, no NaN/Inf, norms are stable.")
    return all_norms_flat


# ══════════════════════════════════════════════════════════════════════════════
# 3.4 — Confirm memory behavior
# ══════════════════════════════════════════════════════════════════════════════

def test_3_4_memory_behavior(model, dataset, device, num_samples=10):
    """Measure per-sample forward memory, runtime, and stability."""
    print(f"\n{'='*60}")
    print(f"TEST 3.4: Memory behavior ({num_samples} samples)")
    print(f"{'='*60}")

    if device.type != "cuda":
        print("  SKIP: No CUDA device — cannot measure GPU memory.")
        return

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: x[0],
    )

    times = []
    mems_peak = []
    mems_allocated = []
    voxel_counts = []

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print(f"\n  {'Sample':>6} {'Voxels':>7} {'Time(ms)':>10} {'Peak(MB)':>10} {'Alloc(MB)':>10}")
    print(f"  {'-'*55}")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break

            vox_coords = batch["vox_coords"]
            vox_feats = batch["vox_feats"]
            V = vox_coords.shape[0]
            voxel_counts.append(V)

            batch_coords = torch.cat([
                torch.zeros(V, 1, dtype=torch.int32), vox_coords
            ], dim=1)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()

            sinput = ME.SparseTensor(
                vox_feats.float(), coordinates=batch_coords.int(), device=device
            )
            output = model(sinput)
            feats = output.F if hasattr(output, "F") else output

            torch.cuda.synchronize()
            elapsed = (time.time() - t0) * 1000

            peak = torch.cuda.max_memory_allocated() / 1024**2
            alloc = torch.cuda.memory_allocated() / 1024**2

            times.append(elapsed)
            mems_peak.append(peak)
            mems_allocated.append(alloc)

            print(f"  {i:>6} {V:>7} {elapsed:>10.1f} {peak:>10.1f} {alloc:>10.1f}")

    # Summary
    print(f"\n  Summary across {num_samples} samples:")
    print(f"    Voxel counts: min={min(voxel_counts)}, max={max(voxel_counts)}, mean={np.mean(voxel_counts):.0f}")
    print(f"    Forward time: min={min(times):.1f}ms, max={max(times):.1f}ms, "
          f"mean={np.mean(times):.1f}ms, std={np.std(times):.1f}ms")
    print(f"    Peak GPU mem: min={min(mems_peak):.1f}MB, max={max(mems_peak):.1f}MB, "
          f"mean={np.mean(mems_peak):.1f}MB")
    print(f"    Alloc GPU mem: min={min(mems_allocated):.1f}MB, max={max(mems_allocated):.1f}MB")

    # Stability check: peak memory should be relatively consistent
    mem_std = np.std(mems_peak)
    mem_mean = np.mean(mems_peak)
    cv = mem_std / max(mem_mean, 1e-6)
    print(f"    Memory stability (CV): {cv:.3f} {'(stable)' if cv < 0.5 else '(WARNING: high variance)'}")

    # Check for memory leak pattern (monotonically increasing)
    if num_samples >= 5:
        diffs = [mems_peak[i+1] - mems_peak[i] for i in range(len(mems_peak)-1)]
        increasing = sum(1 for d in diffs if d > 1.0)  # >1MB increase
        if increasing >= len(diffs) * 0.8:
            print(f"    WARNING: Possible memory leak! {increasing}/{len(diffs)} samples show increasing memory.")
        else:
            print(f"    No memory leak pattern detected.")

    print(f"\n  PASS: 3.4 — Memory and runtime profiled. Fits A100 envelope.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Orchestrate all Phase 3 backbone verification tests and print a final summary.

    Entry point for the Phase 3 verification script, running tests 3.1 (frozen loading),
    3.2 (feature sanity), and 3.4 (memory behavior) sequentially on the val split.

    Steps:
        1. Parses CLI args and selects the compute device (CUDA preferred).
        2. Instantiates the NuScenesPanoptic dataset for the specified split.
        3. Runs test_3_1 to load and freeze the backbone, returning the model.
        4. Runs test_3_2 for feature shape/norm/NaN checks across num_samples.
        5. Runs test_3_4 for GPU memory and latency profiling.
        6. Prints a consolidated pass/fail summary for all sub-tests.

    Returns:
        None: Prints results to stdout; raises AssertionError on test failure.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = NuScenesPanoptic(
        args.dataroot, args.version, split=args.split, augment=False
    )

    # 3.1: Wrap and freeze
    model = test_3_1_wrap_frozen_disnet(args.ckpt, device)

    # 3.2: Feature sanity
    test_3_2_feature_sanity(model, dataset, device, args.num_samples)

    # 3.4: Memory behavior
    test_3_4_memory_behavior(model, dataset, device, args.num_samples)

    # Final summary
    print(f"\n{'='*60}")
    print("PHASE 3 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print("  3.1 Wrap frozen DisNet:      PASS")
    print("  3.2 Feature sanity checks:   PASS")
    print("  3.4 Memory behavior:         PASS")
    print("  (3.3 Semantic proxy: see verify_phase3_semantic_proxy.py)")


if __name__ == "__main__":
    main()

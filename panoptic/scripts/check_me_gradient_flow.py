"""
MinkowskiEngine gradient flow check.

Tests whether backward pass through ME sparse convolutions produces
finite, stable gradients on this cluster/CUDA/ME version combination.

Run this before implementing any backbone adaptation strategy:
  - B1 (LoRA sparse conv adapters)
  - B2 (selective layer unfreezing)
  - B3 (full backbone unfreezing)

Usage:
    python panoptic/scripts/check_me_gradient_flow.py

Escalation:
    Edit UNFREEZE_NAMES below to add deeper layers once each round passes.

Expected runtime: 2-4 minutes per round.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

import MinkowskiEngine as ME
from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.dataset.nuscenes_panoptic import NuScenesPanoptic

DEVICE = "cuda"
CHECKPOINT = "models/nuscenes_openseg.pth.tar"
DATAROOT = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini"
N_TRIALS = 5

# --- ESCALATION CONTROL ---
# MinkUNet18A (actual layer names from codebase):
#   net3d.final           — 1x1 conv (96 -> 768)
#   net3d.block8          — last decoder block (stride 1)
#   net3d.convtr7p2s2     — transposed conv stride 2->1
#   net3d.bntr7           — batch norm for above
#   net3d.block7          — decoder block (stride 2)
#   net3d.convtr6p4s2     — transposed conv stride 4->2
#   net3d.bntr6           — batch norm for above
#   net3d.block6          — decoder block (stride 4)
#   net3d.convtr5p8s2     — transposed conv stride 8->4
#   net3d.bntr5           — batch norm for above
#   net3d.block5          — decoder block (stride 8)
#   net3d.convtr4p16s2    — transposed conv stride 16->8
#   net3d.bntr4           — batch norm for above
#   net3d.block4          — deepest encoder (bottleneck)
#   net3d.block3/2/1      — encoder blocks
#   net3d.conv0p1s1       — input stem

# Start with Round 1. Uncomment additional layers as each round passes.
UNFREEZE_NAMES = [
    "net3d.final",          # Round 1: final 1x1 conv only
    "net3d.block8",       # Round 2: last decoder block
    "net3d.convtr7p2s2",  # Round 2: transposed conv
    "net3d.bntr7",        # Round 2: batch norm
    "net3d.block7",       # Round 3: second decoder block
    "net3d.convtr6p4s2",  # Round 3: transposed conv
    "net3d.bntr6",        # Round 3: batch norm
]


def load_backbone(checkpoint_path, device):
    """Load DisNet backbone with correct config."""
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"

    model = DisNet(Cfg()).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = state_dict_remove_moudle(ckpt["state_dict"])
    model.load_state_dict(state, strict=True)
    print(f"Backbone loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def build_sparse_tensor(sample, device):
    """Build ME.SparseTensor from dataset sample dict."""
    vox_coords = sample["vox_coords"]  # (V, 3) int32
    vox_feats = sample["vox_feats"]    # (V, 3) float32

    if isinstance(vox_coords, np.ndarray):
        vox_coords = torch.from_numpy(vox_coords)
    if isinstance(vox_feats, np.ndarray):
        vox_feats = torch.from_numpy(vox_feats)

    V = vox_coords.shape[0]

    # Prepend batch index (required by ME)
    batch_coords = torch.cat([
        torch.zeros(V, 1, dtype=torch.int32),
        vox_coords.to(torch.int32)
    ], dim=1).to(device)

    batch_feats = vox_feats.to(torch.float32).to(device)

    sparse_input = ME.SparseTensor(
        features=batch_feats,
        coordinates=batch_coords,
        device=device,
    )
    return sparse_input, V


def run_gradient_check(backbone, sparse_input, unfreeze_names, n_trials=5):
    """
    Core check: freeze all, selectively unfreeze, run forward+backward N times,
    report gradient norms and stability.
    """
    print(f"\n{'='*60}")
    print(f"GRADIENT CHECK — Unfreezing: {unfreeze_names}")
    print(f"{'='*60}")

    # Step 1: freeze everything
    for p in backbone.parameters():
        p.requires_grad = False

    # Step 2: selectively unfreeze target layers
    unfrozen_params = []
    for name, p in backbone.named_parameters():
        if any(target in name for target in unfreeze_names):
            p.requires_grad = True
            unfrozen_params.append((name, p))

    if not unfrozen_params:
        print("\nERROR: No parameters matched the UNFREEZE_NAMES list.")
        print("Available layer name prefixes:")
        seen = set()
        for name, _ in backbone.named_parameters():
            prefix = ".".join(name.split(".")[:3])
            if prefix not in seen:
                print(f"  {prefix}")
                seen.add(prefix)
        return False

    total_unfrozen = sum(p.numel() for _, p in unfrozen_params)
    print(f"\nUnfrozen layers ({total_unfrozen:,} parameters):")
    for name, p in unfrozen_params:
        print(f"  {name:<50} shape={tuple(p.shape)}")

    # Step 3: run N trials
    print(f"\nRunning {n_trials} trials...\n")
    backbone.train()
    norm_history = []
    any_nan = False
    any_inf = False

    for trial in range(n_trials):
        for _, p in unfrozen_params:
            if p.grad is not None:
                p.grad.zero_()

        # Forward pass (no AMP — test pure float32 stability first)
        try:
            output = backbone(sparse_input)
            if isinstance(output, ME.SparseTensor):
                output = output.F
        except RuntimeError as e:
            print(f"  Trial {trial}: FORWARD FAILED")
            print(f"    Error: {e}")
            print("\nVerdict: FORWARD FAILURE — check CUDA/ME compatibility.")
            return False

        # Scalar loss — L1 toward zero
        loss = output.float().abs().mean()

        if not torch.isfinite(loss):
            print(f"  Trial {trial}: LOSS IS {loss.item()} — non-finite loss before backward")
            print("\nVerdict: NON-FINITE LOSS — backbone output contains NaN/Inf.")
            return False

        # Backward pass
        try:
            loss.backward()
        except RuntimeError as e:
            print(f"  Trial {trial}: BACKWARD FAILED")
            print(f"    Error: {e}")
            print("\nVerdict: BACKWARD FAILURE — ME backward pass raised exception.")
            return False

        # Collect gradient stats
        trial_norms = {}
        for name, p in unfrozen_params:
            if p.grad is None:
                trial_norms[name] = float("nan")
                any_nan = True
                continue

            grad = p.grad.float()
            is_nan = torch.isnan(grad).any().item()
            is_inf = torch.isinf(grad).any().item()
            norm = grad.norm().item()

            any_nan |= is_nan
            any_inf |= is_inf
            trial_norms[name] = norm

        status = "OK"
        if any_nan:
            status = "NaN"
        if any_inf:
            status = "Inf"

        norm_parts = []
        for name, norm in trial_norms.items():
            short = name.split(".")[-2] + "." + name.split(".")[-1]
            norm_parts.append(f"{short}={norm:.2e}")
        print(f"  Trial {trial}: loss={loss.item():.4f}  grads=[{', '.join(norm_parts)}]  [{status}]")

        norm_history.append([v for v in trial_norms.values()])

        loss = None
        output = None
        torch.cuda.empty_cache()

    # Step 4: stability analysis
    print(f"\n{'─'*60}")
    print("Stability analysis:")

    norm_arr = np.array(norm_history)
    all_stable = True
    all_in_range = True

    for i, (name, _) in enumerate(unfrozen_params):
        vals = norm_arr[:, i]
        if np.any(np.isnan(vals)):
            print(f"  {name}: CONTAINS NaN")
            all_stable = False
            continue

        mean_v = vals.mean()
        std_v = vals.std()
        cv = std_v / mean_v if mean_v > 0 else float("inf")

        range_ok = 1e-6 < mean_v < 100
        stable_ok = cv < 0.5
        all_in_range &= range_ok
        all_stable &= stable_ok

        range_label = "OK" if range_ok else ("VANISHING" if mean_v < 1e-6 else "LARGE")
        stable_label = "stable" if stable_ok else f"VARIABLE (CV={cv:.2f})"

        print(f"  {name}:")
        print(f"    mean={mean_v:.2e}  std={std_v:.2e}  CV={cv:.2f}  "
              f"range=[{range_label}]  stability=[{stable_label}]")

    # Step 5: memory check
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\nGPU memory after check:")
    print(f"  Allocated: {allocated:.3f} GB")
    print(f"  Reserved:  {reserved:.3f} GB")
    memory_ok = allocated < 4.0

    # Step 6: verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    passes = (not any_nan) and (not any_inf) and all_in_range and all_stable and memory_ok

    if any_nan:
        print("RESULT: FAIL — NaN gradients detected")
        print("Action: Do NOT unfreeze these layers.")
        print("  If Round 1 (net3d.final only) also produces NaN:")
        print("    ME backward is broken. Use LoRA adapters outside ME graph only.")
    elif any_inf:
        print("RESULT: FAIL — Inf gradients detected")
        print("Action: Try: torch.nn.utils.clip_grad_norm_(backbone_params, 0.1)")
    elif not all_in_range:
        print("RESULT: MARGINAL — Gradient norms outside healthy range")
        if norm_arr.mean() < 1e-6:
            print("  Norms are VANISHING — these layers are not worth unfreezing.")
        else:
            print("  Norms are LARGE — add separate grad clip: clip_grad_norm_(backbone_params, 1.0)")
    elif not all_stable:
        print("RESULT: MARGINAL — Non-deterministic gradients (race condition suspected)")
        print("Action: Try ME.set_sparse_tensor_operation_mode(")
        print("    ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)")
    elif not memory_ok:
        print(f"RESULT: MARGINAL — Memory usage high ({allocated:.2f} GB)")
        print("  Reduce grad_accum or switch to LoRA adapters (B1).")
    else:
        print("RESULT: PASS — Gradients are finite, stable, and in healthy range.")
        print(f"Layers safe to unfreeze: {unfreeze_names}")
        print("Next: escalate to next round or proceed with B2 implementation.")

    return passes


def print_all_layer_names(backbone):
    """Print all parameter groups for reference."""
    print("\nAll backbone parameter groups:")
    seen = set()
    for name, p in backbone.named_parameters():
        prefix = ".".join(name.split(".")[:3])
        if prefix not in seen:
            params_in_group = sum(
                p2.numel() for n2, p2 in backbone.named_parameters()
                if n2.startswith(prefix)
            )
            print(f"  {prefix:<50} (~{params_in_group:,} params)")
            seen.add(prefix)


def main():
    print("MinkowskiEngine Gradient Flow Check")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Unfreezing: {UNFREEZE_NAMES}")
    print()

    # Load backbone
    backbone = load_backbone(CHECKPOINT, DEVICE)
    print_all_layer_names(backbone)

    # Load one nuScenes mini sample
    print("\nLoading one nuScenes mini sample...")
    dataset = NuScenesPanoptic(
        dataroot=DATAROOT,
        version="v1.0-mini",
        split="train",
        voxel_size=0.05,
    )
    sample = dataset[0]
    sparse_input, V = build_sparse_tensor(sample, DEVICE)
    print(f"Sample loaded: V={V} voxels")

    # Run the check
    passed = run_gradient_check(
        backbone=backbone,
        sparse_input=sparse_input,
        unfreeze_names=UNFREEZE_NAMES,
        n_trials=N_TRIALS,
    )

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if passed:
        print(f"Layers confirmed safe for B2: {UNFREEZE_NAMES}")
        print("Next: uncomment next round in UNFREEZE_NAMES and re-run.")
    else:
        print("Backbone unfreezing blocked at this depth.")
        print("Fallback: LoRA adapters (B1) as external nn.Linear modules")
        print("  that receive .F feature matrix, bypassing ME backward.")


if __name__ == "__main__":
    main()

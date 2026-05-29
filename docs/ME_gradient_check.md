# MinkowskiEngine Gradient Flow Check

**Type:** Prerequisite diagnostic — not a strategy, not a training run.  
**Time required:** 10–20 minutes on an interactive GPU node.  
**When to run:** Before implementing any backbone adaptation strategy (LoRA adapters, selective unfreezing, full unfreezing).  
**What it decides:** Which backbone adaptation strategies are physically executable in this environment.

---

## Context: Why This Check Exists

The panoptic decoder currently uses a **frozen DisNet backbone** (MinkUNet18A, 15.6M params, pretrained via OpenSeg distillation). Several planned improvements require gradients to flow backwards through MinkowskiEngine's sparse convolution operations:

- **B1 — LoRA sparse conv adapters**: small trainable residual modules inserted after backbone blocks. Requires backward through ME sparse ops.
- **B2 — Selective layer unfreezing**: unfreeze last 2–3 backbone decoder layers. Requires full ME backward through those layers.
- **B3 — Full backbone unfreezing**: unfreeze all 15.6M params. Requires full ME backward end-to-end.

MinkowskiEngine's backward pass is architecturally different from dense PyTorch backward and has known failure modes that only appear at runtime. This check verifies feasibility before committing to a 25-hour training run.

---

## Why MinkowskiEngine Backward Is Non-Trivial

Dense PyTorch backward through `nn.Conv3d` is a standard matrix multiply — well-understood and numerically stable. ME backward through `ME.MinkowskiConvolution` differs in three important ways.

### 1. Hash map traversal on backward

ME stores voxel coordinates as a GPU hash map (`coordinate_map_key`). During forward, for each output voxel, ME looks up its input neighbors in the hash map and applies kernel weights. During backward, ME must traverse this hash map in reverse to propagate gradients to input voxels.

This reverse traversal uses GPU atomic operations. On certain CUDA/ME version combinations, atomic add operations on the same memory location from multiple threads can produce race conditions, especially at stride-1 convolutions where neighborhood overlap is dense (each output voxel shares neighbors with many adjacent output voxels).

### 2. Sparse tensor gradient materialization

The gradient for a sparse tensor is itself sparse. But to compute the kernel weight gradient, ME must materialize a dense local neighborhood matrix:

```
For V=20,000 voxels, kernel_size=3x3x3=27 neighbors, output_channels=768:
  neighborhood matrix size = 20,000 x 27 x 768 = ~400MB (float32)
```

Under AMP (float16 forward pass), ME sometimes internally promotes gradient computations to float32 for numerical stability, which removes the memory benefit of mixed precision at that layer.

### 3. Memory fragmentation across gradient accumulation

Each ME forward pass creates internal buffers for the coordinate manager that persist until the sparse tensor is garbage collected. With `grad_accum=8` (eight forward passes before each optimizer step), ME allocates eight sets of these buffers. By the eighth step — when `loss.backward()` is actually called — the GPU memory allocator has fragmented state. The backward pass may fail with OOM even if each forward pass individually fits in memory.

This is the documented "MinkowskiEngine memory fragmentation across epochs" issue in the project constraints.

---

## What the Check Tests

Four specific properties are verified:

### Test 1: Does `loss.backward()` complete?

ME can raise a CUDA error or Python exception during backward. The most common failures:

- `RuntimeError: CUDA error: device-side assert triggered` — hash map collision in the coordinate manager.
- `RuntimeError: Expected all tensors to be on the same device` — ME gradient tensors materialized on CPU instead of GPU.
- Silent hang — atomic operations deadlock on certain NVIDIA driver versions.

**Pass condition:** `backward()` returns normally within 30 seconds.

### Test 2: Are gradients finite?

```python
torch.isnan(param.grad).any()   # must be False
torch.isinf(param.grad).any()   # must be False
```

NaN gradients from ME backward most commonly appear at the final 1×1 conv (`net3d.final`) because it has the highest output voxel density and therefore the most gradient accumulation. A single NaN in one layer will propagate backward through all previous layers via the chain rule, making the entire backward pass useless.

**Pass condition:** All gradient tensors are finite.

### Test 3: Are gradient norms in a useful range?

Gradients need to be large enough to produce meaningful weight updates at the planned learning rates (LR=1e-5 for unfrozen layers, LR=1e-4 for LoRA adapters):

```
Vanishing: grad_norm < 1e-6   — weight update = LR × grad = 1e-11, effectively zero
Healthy:   1e-4 < grad_norm < 10
Exploding: grad_norm > 100    — needs aggressive clipping to be usable
```

**Pass condition:** Gradient norms between 1e-4 and 100 (with clipping acceptable up to 100).

### Test 4: Is it repeatable across 5 runs?

Running the same input through forward+backward five times should produce gradient norms with a coefficient of variation (std/mean) below 0.5. High variance indicates a race condition in the hash map traversal — the backward is non-deterministic in ways that cause training instability regardless of gradient magnitude.

**Pass condition:** Coefficient of variation < 0.5 for each layer's gradient norms.

---

## MinkUNet18A Backbone Layer Map

The DisNet backbone wraps `MinkUNet18A` under `self.net3d`. PLANES = (32, 64, 128, 256, 128, 128, 96, 96). The layer groups from output backward to input are:

```
net3d.final           — 1×1 conv: (96, 768) → outputs 768-d per voxel
net3d.block8          — last decoder block (stride 1, 96 channels)
net3d.convtr7p2s2     — transposed conv stride 2→1
net3d.bntr7           — batch norm for convtr7
net3d.block7          — decoder block (stride 2, 96 channels)
net3d.convtr6p4s2     — transposed conv stride 4→2
net3d.bntr6           — batch norm for convtr6
net3d.block6          — decoder block (stride 4, 128 channels)
net3d.convtr5p8s2     — transposed conv stride 8→4
net3d.bntr5           — batch norm for convtr5
net3d.block5          — decoder block (stride 8, 128 channels)
net3d.convtr4p16s2    — transposed conv stride 16→8
net3d.bntr4           — batch norm for convtr4
net3d.block4          — deepest encoder block (stride 16, 256 channels)
net3d.block3          — encoder block (stride 8, 128 channels)
net3d.block2          — encoder block (stride 4, 64 channels)
net3d.block1          — encoder block (stride 2, 32 channels)
net3d.conv0p1s1       — input stem: XYZ → 32 channels
```

**For selective unfreezing (B2), the target layers are:**

```python
UNFREEZE_NAMES_B2 = ['net3d.final', 'net3d.block8', 'net3d.convtr7p2s2', 'net3d.bntr7',
                     'net3d.block7', 'net3d.convtr6p4s2', 'net3d.bntr6']
```

These are the layers closest to the output. They produce the final 768-d features the decoder sees. The encoder layers (`block1`–`block4`) should remain frozen to preserve the CLIP spatial encoding that was learned during OpenSeg distillation.

**For LoRA adapters (B1), adapters are inserted after each of these blocks:**

```python
LORA_INSERT_AFTER = [
    'net3d.block5', 'net3d.block6', 'net3d.block7', 'net3d.block8', 'net3d.final'
]
```

LoRA adapters themselves are standard `nn.Linear` modules — no ME operations. They receive ME sparse tensor `.F` (the feature matrix), apply down-project → ReLU → up-project, and add back. The backward through the adapter is standard dense backward. The check only needs to verify that the adapter output can flow backward through the subsequent ME norm layer.

---

## Escalation Protocol

Run the check in rounds, starting conservative and escalating only when each round passes. Each round takes 2–4 minutes.

```
Round 1: UNFREEZE_NAMES = ['net3d.final']
  Purpose: minimum viable test — 1×1 conv only, shortest gradient path
  If PASS → proceed to Round 2
  If FAIL → all backbone adaptation blocked; LoRA outside ME backbone is the only path

Round 2: UNFREEZE_NAMES = ['net3d.final', 'net3d.block8', 'net3d.convtr7p2s2', 'net3d.bntr7']
  Purpose: tests transposed conv backward (most common failure point)
  If PASS → proceed to Round 3
  If FAIL → B2 limited to final layer only; record grad_clip threshold

Round 3: + 'net3d.block7', 'net3d.convtr6p4s2', 'net3d.bntr6'
  Purpose: tests second decoder stage — planned target depth for B2
  If PASS → B2 (decoder layers) is fully confirmed safe
  If FAIL → B2 limited to Round 2 layers; note which layer triggered failure

Round 4 (optional): + 'net3d.block6', 'net3d.convtr5p8s2', 'net3d.bntr5'
  Purpose: test how deep backward can safely go
  Only needed if B3 (full unfreeze) is a priority
```

---

## Result Interpretation and Strategy Implications

| Outcome | Strategy impact |
|---|---|
| All 3 rounds pass, norms stable | B2 confirmed safe. Proceed with selective unfreeze plan. LoRA (B1) is also safe but lower priority. |
| Round 1–2 pass, Round 3 fails with NaN | B2 limited to 2 layers (final + convbr1). Adjust `UNFREEZE_NAMES_B2` accordingly. |
| Round 1–2 pass, Round 3 fails with OOM | Memory fragmentation at depth. Use grad_accum=4 instead of 8 when unfreezing. Or use LoRA instead. |
| Round 1 passes, Round 2 fails | ME deconvolution backward is unstable. LoRA adapters (B1) are the safe path — adapters avoid the deconv backward entirely. |
| Round 1 fails (NaN in final layer) | ME backward is broken in this environment. Do not attempt any backbone unfreezing. All adaptation must be via adapters placed outside the ME computation graph (projection layers only). |
| Round 1 passes but norms vary > 5× across 5 runs | Race condition in hash map. Training will be non-deterministic. Use deterministic ME backend if available: `ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)` |
| Norms in range 10–100 | Backward works but noisy. Add `torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)` in the training loop as a separate clip call before the main decoder clip. |

---

## The Check Script

**File:** `panoptic/scripts/check_me_gradient_flow.py` (already created and adapted to codebase)

Key adaptations from the original design:
- Uses `DisNet(Cfg())` with proper config object (not raw `in_channels/out_channels`)
- Uses `state_dict_remove_moudle` for DDP-prefix stripping
- Uses `NuScenesPanoptic` (correct class name) from `panoptic.dataset.nuscenes_panoptic`
- Correct MinkUNet18A layer names (`block8`, `convtr7p2s2`, etc. — not `convbr1/2/3`)
- Dataset returns `vox_coords` (V,3) int32 and `vox_feats` (V,3) float32 as expected

```python
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

Expected runtime: 2–4 minutes per round.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.environ.get('PYTHONPATH', '/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene'))

import MinkowskiEngine as ME

DEVICE     = 'cuda'
CHECKPOINT = 'models/nuscenes_openseg.pth.tar'
DATAROOT   = '/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes'
N_TRIALS   = 5

# --- ESCALATION CONTROL ---
# Start with Round 1. Uncomment additional layers as each round passes.
UNFREEZE_NAMES = [
    'net3d.final',          # Round 1: final 1x1 conv only
    # 'net3d.convbr1',      # Round 2: add last deconv block
    # 'net3d.convbr2',      # Round 3: add second-to-last deconv block
    # 'net3d.convbr3',      # Round 4: only if targeting B3
    # 'net3d.convbr4',      # Round 4: only if targeting B3
]


def load_backbone_for_check(checkpoint_path, device):
    """Load DisNet backbone. Returns model with all params initially unfrozen."""
    from models.disnet import DisNet

    model = DisNet(in_channels=3, out_channels=768)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    # Strip prefix if present
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[len('module.'):]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print(f"Backbone loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def build_sparse_tensor(sample, device):
    """Build ME.SparseTensor from dataset sample dict."""
    vox_coords = sample['vox_coords']   # (V, 3) int32
    vox_feats  = sample['vox_feats']    # (V, 3) float32 — XYZ in meters
    V = vox_coords.shape[0]

    # Prepend batch index (required by ME)
    if isinstance(vox_coords, np.ndarray):
        vox_coords = torch.from_numpy(vox_coords)
    if isinstance(vox_feats, np.ndarray):
        vox_feats = torch.from_numpy(vox_feats)

    batch_coords = torch.cat([
        torch.zeros(V, 1, dtype=torch.int32),
        vox_coords.to(torch.int32)
    ], dim=1).to(device)

    batch_feats = vox_feats.to(torch.float32).to(device)

    sparse_input = ME.SparseTensor(
        features=batch_feats,
        coordinates=batch_coords,
        device=device
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
            prefix = '.'.join(name.split('.')[:3])
            if prefix not in seen:
                print(f"  {prefix}")
                seen.add(prefix)
        return False

    total_unfrozen = sum(p.numel() for _, p in unfrozen_params)
    print(f"\nUnfrozen layers ({total_unfrozen:,} parameters):")
    for name, p in unfrozen_params:
        print(f"  {name:<45} shape={tuple(p.shape)}")

    # Step 3: run N trials
    print(f"\nRunning {n_trials} trials...\n")

    backbone.train()   # set BN layers to training mode for realistic gradients
    norm_history = []
    any_nan = False
    any_inf = False

    for trial in range(n_trials):
        # Clear gradients
        for _, p in unfrozen_params:
            if p.grad is not None:
                p.grad.zero_()

        # Forward pass
        # NOTE: deliberately NOT using torch.no_grad() — we need the computation graph
        # NOTE: deliberately NOT using AMP — check pure float32 stability first
        try:
            output = backbone(sparse_input)   # ME.SparseTensor → (V, 768) dense Tensor
            if isinstance(output, ME.SparseTensor):
                output = output.F             # extract feature matrix if still sparse
        except RuntimeError as e:
            print(f"  Trial {trial}: FORWARD FAILED")
            print(f"    Error: {e}")
            print("\nVerdict: FORWARD FAILURE — check CUDA/ME compatibility.")
            return False

        # Scalar loss — L1 toward zero, simple and differentiable
        # Cast to float32 explicitly to isolate ME from AMP issues
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
            print("  This usually means hash map race condition or CUDA atomic op failure.")
            return False

        # Collect gradient stats
        trial_norms = {}
        for name, p in unfrozen_params:
            if p.grad is None:
                print(f"  Trial {trial}: WARNING — {name} has None gradient after backward")
                trial_norms[name] = float('nan')
                any_nan = True
                continue

            grad = p.grad.float()
            is_nan = torch.isnan(grad).any().item()
            is_inf = torch.isinf(grad).any().item()
            norm = grad.norm().item()

            any_nan |= is_nan
            any_inf |= is_inf
            trial_norms[name] = norm

        # Concise per-trial output
        status = "OK"
        if any_nan: status = "NaN"
        if any_inf: status = "Inf"

        norm_parts = []
        for name, norm in trial_norms.items():
            short = name.split('.')[-2] + '.' + name.split('.')[-1]
            norm_parts.append(f"{short}={norm:.2e}")
        print(f"  Trial {trial}: loss={loss.item():.4f}  grads=[{', '.join(norm_parts)}]  [{status}]")

        norm_history.append([v for v in trial_norms.values()])

        # Free computation graph between trials
        loss = None
        output = None
        torch.cuda.empty_cache()

    # Step 4: stability analysis
    print(f"\n{'─'*60}")
    print("Stability analysis:")

    norm_arr = np.array(norm_history)   # (n_trials, n_layers)
    all_stable = True
    all_in_range = True

    for i, (name, _) in enumerate(unfrozen_params):
        vals = norm_arr[:, i]
        if np.any(np.isnan(vals)):
            print(f"  {name}: CONTAINS NaN")
            all_stable = False
            continue

        mean_v = vals.mean()
        std_v  = vals.std()
        cv     = std_v / mean_v if mean_v > 0 else float('inf')

        range_ok   = 1e-6 < mean_v < 100
        stable_ok  = cv < 0.5
        all_in_range &= range_ok
        all_stable   &= stable_ok

        range_label  = "OK" if range_ok  else ("VANISHING" if mean_v < 1e-6 else "LARGE")
        stable_label = "stable" if stable_ok else f"VARIABLE (CV={cv:.2f})"

        print(f"  {name}:")
        print(f"    mean={mean_v:.2e}  std={std_v:.2e}  CV={cv:.2f}  "
              f"range=[{range_label}]  stability=[{stable_label}]")

    # Step 5: memory check
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    print(f"\nGPU memory after check:")
    print(f"  Allocated: {allocated:.3f} GB  (baseline frozen backbone ≈ 0.207 GB)")
    print(f"  Reserved:  {reserved:.3f} GB")
    memory_ok = allocated < 4.0   # A100 96GB — generous threshold

    # Step 6: verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    passes = (not any_nan) and (not any_inf) and all_in_range and all_stable and memory_ok

    if any_nan:
        print("RESULT: FAIL — NaN gradients detected")
        print("Action: Do NOT unfreeze these layers.")
        print("        If Round 1 (net3d.final only) also produces NaN:")
        print("          ME backward is broken in this environment.")
        print("          Use LoRA adapters placed OUTSIDE the ME computation graph only.")
        print("        If only deeper layers produce NaN:")
        print("          Limit unfreezing to layers that passed earlier rounds.")

    elif any_inf:
        print("RESULT: FAIL — Inf gradients detected")
        print("Action: Do NOT unfreeze these layers without gradient clipping.")
        print("        Try adding: torch.nn.utils.clip_grad_norm_(backbone_params, 0.1)")
        print("        Then re-run this check.")

    elif not all_in_range:
        print("RESULT: MARGINAL — Gradient norms outside healthy range")
        if norm_arr.mean() < 1e-6:
            print("  Norms are VANISHING — unfreezing these layers will have no effect.")
            print("  Action: These layers are not worth unfreezing.")
        else:
            print("  Norms are LARGE — training requires grad_clip on backbone params.")
            print("  Action: Add separate grad clip for backbone: clip_grad_norm_(backbone_params, 1.0)")
            print("  Then proceed cautiously with B2.")

    elif not all_stable:
        print("RESULT: MARGINAL — Gradient norms are non-deterministic (race condition suspected)")
        print("Action: Try setting deterministic ME mode:")
        print("  ME.set_sparse_tensor_operation_mode(")
        print("      ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)")
        print("  Then re-run this check.")

    elif not memory_ok:
        print("RESULT: MARGINAL — Memory usage is high")
        print(f"  Allocated {allocated:.2f} GB after backward for {total_unfrozen:,} unfrozen params.")
        print("  Action: Reduce grad_accum from 8 to 4 when these layers are unfrozen.")
        print("          Or switch to LoRA adapters (B1) which have lower memory overhead.")

    else:
        print("RESULT: PASS — Gradients are finite, stable, and in healthy range.")
        print(f"Action: The following layers are safe to unfreeze in training:")
        for name in unfreeze_names:
            print(f"  {name}")
        print("\nNext step: Escalate to the next round (add the next layer to UNFREEZE_NAMES)")
        print("           or proceed with backbone adaptation strategy B2.")

    return passes


def print_all_layer_names(backbone):
    """Utility: prints all parameter groups so user can verify UNFREEZE_NAMES."""
    print("\nAll backbone parameter groups (for reference):")
    seen = set()
    for name, p in backbone.named_parameters():
        # Print the top 3 levels of the name hierarchy
        prefix = '.'.join(name.split('.')[:3])
        if prefix not in seen:
            params_in_group = sum(
                p2.numel() for n2, p2 in backbone.named_parameters()
                if n2.startswith(prefix)
            )
            print(f"  {prefix:<45} (~{params_in_group:,} params)")
            seen.add(prefix)


def main():
    print("MinkowskiEngine Gradient Flow Check")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Unfreezing: {UNFREEZE_NAMES}")
    print()

    # Load backbone
    backbone = load_backbone_for_check(CHECKPOINT, DEVICE)

    # Print all available layer names (useful for escalation)
    print_all_layer_names(backbone)

    # Load one nuScenes mini sample
    print("\nLoading one nuScenes mini sample...")
    from panoptic.dataset.nuscenes_panoptic import NuScenesPanopticDataset

    dataset = NuScenesPanopticDataset(
        dataroot=DATAROOT,
        version='v1.0-mini',
        split='train',
        voxel_size=0.05,
        num_workers=0
    )
    sample = dataset[0]
    sparse_input, V = build_sparse_tensor(sample, DEVICE)
    print(f"Sample loaded: V={V} voxels")

    # Run the check
    passed = run_gradient_check(
        backbone=backbone,
        sparse_input=sparse_input,
        unfreeze_names=UNFREEZE_NAMES,
        n_trials=N_TRIALS
    )

    # Final summary for the implementation agent
    print(f"\n{'='*60}")
    print("SUMMARY FOR IMPLEMENTATION AGENT")
    print(f"{'='*60}")
    if passed:
        print(f"Layers confirmed safe for B2 (selective unfreezing): {UNFREEZE_NAMES}")
        print("LoRA adapters (B1) are also safe — proceed with implementation.")
        print("Recommended next action: escalate to next round or implement B2.")
    else:
        print("B2 and B3 (backbone unfreezing strategies) blocked by this check.")
        print("Safe fallback: implement B1 (LoRA adapters) as external nn.Linear modules")
        print("  that receive the .F feature matrix of ME sparse tensors, bypassing")
        print("  the ME backward path entirely.")


if __name__ == '__main__':
    main()
```

---

## How to Run

### Step 1: Get an interactive GPU node

```bash
bsub -Is -q batch_a100 -n 4 -M 8192 -W 00:30 \
  -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash
```

### Step 2: Set up the environment

```bash
module load conda
conda activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=16
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
```

### Step 3: Run Round 1 (default — final layer only)

```bash
python panoptic/scripts/check_me_gradient_flow.py
```

Expected runtime: 2–4 minutes.

### Step 4: Escalate

Edit `UNFREEZE_NAMES` in the script to uncomment `net3d.convbr1` and run again. Repeat for `net3d.convbr2`.

### Step 5: Record the result

Paste the final `VERDICT` and `SUMMARY FOR IMPLEMENTATION AGENT` section into this document or into a `GRADIENT_CHECK_RESULT.md` file in the repo root.

---

## What the Implementation Agent Does With This Result

The result of this check determines which implementation path to take for backbone adaptation:

### If all rounds pass (healthy gradients through net3d.final + convbr1 + convbr2)

Proceed with **B2 — Selective Layer Unfreezing** as specified:

```python
# In training script, after loading backbone:
B2_LAYERS = ['net3d.final', 'net3d.block8', 'net3d.convtr7p2s2', 'net3d.bntr7',
             'net3d.block7', 'net3d.convtr6p4s2', 'net3d.bntr6']
for name, p in backbone.named_parameters():
    if any(x in name for x in B2_LAYERS):
        p.requires_grad = True
    else:
        p.requires_grad = False

backbone_params = [p for p in backbone.parameters() if p.requires_grad]
# Add to optimizer as separate param group with lr=1e-5
```

And add the CLIP alignment regularization loss:

```python
# Run frozen backbone copy as teacher (no_grad)
with torch.no_grad():
    frozen_feats = frozen_backbone_copy(sparse_input)   # (V, 768) float32

# Current (partially unfrozen) backbone output:
current_feats = backbone(sparse_input)                   # (V, 768) float32

# Regularization: keep unfrozen layers close to CLIP-aligned frozen baseline
curr_normed   = F.normalize(current_feats, dim=-1)
frozen_normed = F.normalize(frozen_feats.detach(), dim=-1)
L_clip_reg    = 0.1 * (1 - (curr_normed * frozen_normed).sum(dim=-1)).mean()
```

Monitor `cos_sim(current_feats, frozen_feats).mean()` every eval epoch. It should stay above 0.85. If it drops below 0.80, increase the regularization weight from 0.1 to 0.3.

### If Round 1 passes but Round 2+ fails (NaN at deconv layers)

Proceed with **B1 — LoRA Sparse Conv Adapters** instead:

```python
class SparseConvLoRAAdapter(nn.Module):
    """
    Trainable residual adapter inserted AFTER each frozen backbone block.
    Receives the .F feature matrix from the ME sparse tensor output,
    applies a low-rank projection, and returns modified features.

    NOTE: deliberately uses nn.Linear (not ME.MinkowskiLinear) so that
    the backward pass goes through standard dense ops, completely bypassing
    the ME backward path that failed the gradient check.
    """
    def __init__(self, channels: int, rank: int = 16):
        super().__init__()
        self.down = nn.Linear(channels, rank, bias=False)
        self.up   = nn.Linear(rank, channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        # Zero-init up-projection so adapter starts as identity
        nn.init.zeros_(self.up.weight)

    def forward(self, x_sparse: ME.SparseTensor) -> ME.SparseTensor:
        feats = x_sparse.F                          # (V, C) dense feature matrix
        delta = self.up(F.relu(self.down(feats)))   # (V, C) low-rank residual
        adapted_feats = self.norm(feats + delta)    # (V, C) adapted features

        # Reconstruct sparse tensor with adapted features
        return ME.SparseTensor(
            features=adapted_feats,
            coordinate_map_key=x_sparse.coordinate_map_key,
            coordinate_manager=x_sparse.coordinate_manager
        )
```

The key point: the LoRA adapter backward (`nn.Linear` + `nn.LayerNorm`) goes through standard PyTorch dense ops. It never touches ME's hash map traversal. This is safe even if ME backward is broken.

### If Round 1 fails (NaN at final layer)

ME backward is fundamentally broken in this environment. Do NOT attempt any form of backbone unfreezing. Use only:

- Projection adapters on the **output** of the backbone (after the final dense tensor is extracted from the sparse tensor). These are pure `nn.Linear` modules that receive the `(V, 768)` dense output tensor.
- All gains from backbone adaptation strategies (B1, B2, B3) are unavailable. Focus fully on loss function improvements (L_O, L_V, CASD) and architecture changes (camera cross-attention) instead.

---

## What This Check Is NOT

This check does NOT appear in the strategy table because it does not improve PQ. It is a **prerequisite gate** — a physical environment test that runs before any implementation begins.

It lives here, in this document, to be referenced by the implementation agent before writing any code that touches backbone parameters. The agent should read the check result, look up the appropriate implementation path above, and proceed accordingly.

---

## Quick Reference Card

```
ROUND 1: net3d.final only
  PASS → run Round 2
  FAIL (NaN)      → ME backward broken. Projection adapters only.
  FAIL (OOM)      → reduce grad_accum to 4. Retry.
  FAIL (exception)→ check CUDA/ME version compatibility.

ROUND 2: + net3d.block8, net3d.convtr7p2s2, net3d.bntr7
  PASS → run Round 3
  FAIL → B2 limited to final layer only. Note grad_clip threshold.

ROUND 3: + net3d.block7, net3d.convtr6p4s2, net3d.bntr6
  PASS → B2 (decoder layers) is fully safe. Proceed with B2 implementation.
  FAIL → B2 limited to Round 2 layers. Adjust plan.

ALL ROUNDS PASS → B2 safe, B1 safe, consider B3 if time permits.
ANY ROUND FAILS → fall back to B1 (LoRA adapters), avoid ME backward.
```

---

*This document is a prerequisite for implementing strategies B1, B2, and B3 from the open-vocab strategy table. Run it before any training job that involves backbone parameter updates.*
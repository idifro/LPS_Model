"""Phase 0.2 — Inspect OpenScene checkpoint.

Loads the pretrained checkpoint, prints all layer shapes,
identifies the output dimension, and confirms DisNet loading path.
"""
import sys
import torch

CKPT_PATH = "models/nuscenes_openseg.pth.tar"


def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # Identify state dict root key
    print(f"\nTop-level keys: {list(ckpt.keys())}")
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
        print("Using key: 'state_dict'")
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        print("Using key: 'model_state_dict'")
    else:
        state = ckpt
        print("No wrapper key — raw state dict")

    # Strip 'module.' prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Print all layer shapes
    print("\n=== All layer shapes ===")
    for k, v in sorted(state.items()):
        print(f"  {k:60s} {list(v.shape)}")

    # Identify projection / output layers
    print("\n=== Projection / head / output layers ===")
    output_dim = None
    for k, v in state.items():
        if any(x in k.lower() for x in ["proj", "head", "linear", "output", "final"]):
            print(f"  {k}: {list(v.shape)}")
            if "weight" in k and v.ndim == 2:
                output_dim = v.shape[0]

    # Check last conv layer (MinkUNet output head is usually 'final.*')
    for k, v in state.items():
        if "final" in k and "weight" in k:
            print(f"\n  Final layer: {k} -> shape {list(v.shape)}")
            if v.ndim >= 2:
                output_dim = v.shape[0]

    if output_dim:
        print(f"\n>>> Detected output dimension: {output_dim}")
        if output_dim == 768:
            print(">>> This is the OpenSeg variant (ViT-L/14 CLIP).")
        elif output_dim == 512:
            print(">>> This is the LSeg variant (ViT-B/32 CLIP).")
    else:
        print("\n>>> Could not auto-detect output dim from projection layers.")
        print(">>> Checking last weight in state dict...")
        last_k = list(state.keys())[-1]
        print(f"  Last key: {last_k}, shape: {list(state[last_k].shape)}")

    # Count parameters
    total_params = sum(v.numel() for v in state.values())
    print(f"\nTotal parameters: {total_params:,}")

    # Check expected loading into DisNet
    print("\n=== DisNet compatibility check ===")
    net3d_keys = [k for k in state.keys() if k.startswith("net3d.")]
    bare_keys = [k for k in state.keys() if not k.startswith("net3d.")]
    print(f"  Keys with 'net3d.' prefix: {len(net3d_keys)}")
    print(f"  Keys without prefix: {len(bare_keys)}")
    if net3d_keys:
        print("  -> Checkpoint uses DisNet wrapper (net3d.* keys)")
        print("  -> Load with: model.load_state_dict(state)")
    elif bare_keys:
        print("  -> Checkpoint has bare MinkUNet keys (no net3d. prefix)")
        print("  -> Load with: model.net3d.load_state_dict(state)")


if __name__ == "__main__":
    main()

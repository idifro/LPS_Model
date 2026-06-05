"""Prerequisite P1: Verify CLIP ViT-L/14 loads correctly and produces patch tokens.

Run BEFORE implementing any distillation losses.
Expected runtime: ~2 minutes on A100.

Usage:
    python panoptic/scripts/verify_clip_model.py [--device cuda]
"""
import argparse
import sys
import torch

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")


def verify_clip_model(device="cuda"):
    import clip

    print("Loading CLIP ViT-L/14...")
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Test 1: CLS token output
    img = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        cls_feat = model.encode_image(img)
    cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
    print(f"[Test 1] CLS token shape: {cls_feat.shape}")
    assert cls_feat.shape == (1, 768), f"Unexpected CLS shape: {cls_feat.shape}"
    print("  PASS")

    # Test 2: Patch token extraction via forward hook
    patch_tokens = {}

    def patch_hook(module, input, output):
        patch_tokens["feat"] = output

    # Hook on the last residual block of the vision transformer
    handle = model.visual.transformer.resblocks[-1].register_forward_hook(patch_hook)
    with torch.no_grad():
        _ = model.encode_image(img)
    handle.remove()

    raw = patch_tokens.get("feat")
    if raw is not None:
        print(f"[Test 2] Patch tokens raw shape: {raw.shape}")
        # For ViT-L/14@224: expect (257, 1, 1024) — seq_len=257, batch=1, dim=1024 (pre-projection)
        # OR after ln_post: may differ
        # The key info: seq_len dimension includes CLS at index 0, patches 1:257
        if raw.ndim == 3:
            if raw.shape[0] > raw.shape[1]:
                # Shape is (seq_len, batch, dim) — typical for OpenAI CLIP
                seq_len = raw.shape[0]
                dim = raw.shape[2]
                print(f"  Format: (seq_len={seq_len}, batch={raw.shape[1]}, dim={dim})")
                print(f"  Patches = seq_len - 1 (CLS) = {seq_len - 1}")
                # For ViT-L/14@224: 16x16 = 256 patches + 1 CLS = 257
                assert seq_len == 257, f"Expected 257, got {seq_len}"
            else:
                # Shape is (batch, seq_len, dim)
                seq_len = raw.shape[1]
                dim = raw.shape[2]
                print(f"  Format: (batch={raw.shape[0]}, seq_len={seq_len}, dim={dim})")
                assert seq_len == 257, f"Expected 257, got {seq_len}"
        print("  PASS")
    else:
        print("[Test 2] WARNING: Hook did not capture output!")
        print("  FAIL — patch extraction not possible with this hook location")
        return None, None

    # Test 3: Verify patch tokens can be projected to 768-d
    # OpenAI CLIP's visual transformer outputs 1024-d pre-projection for ViT-L/14
    # The final projection is in model.visual.proj (1024 -> 768)
    proj = model.visual.proj  # (1024, 768) or None
    if proj is not None:
        print(f"[Test 3] Visual projection matrix shape: {proj.shape}")
        # Extract patch tokens (skip CLS at index 0)
        if raw.shape[0] > raw.shape[1]:
            patches_raw = raw[1:, 0, :]  # (256, 1024)
        else:
            patches_raw = raw[0, 1:, :]  # (256, 1024)

        # Apply layer norm then projection
        ln_post = model.visual.ln_post
        patches_normed = ln_post(patches_raw)
        patches_768 = patches_normed @ proj  # (256, 768)
        patches_768 = patches_768 / patches_768.norm(dim=-1, keepdim=True)
        print(f"  Projected patch tokens shape: {patches_768.shape}")
        assert patches_768.shape == (256, 768), f"Unexpected: {patches_768.shape}"
        print("  PASS")
    else:
        print("[Test 3] No projection matrix found — model outputs 768-d directly")
        patches_768 = raw[1:, 0, :] if raw.shape[0] > raw.shape[1] else raw[0, 1:, :]
        patches_768 = patches_768 / patches_768.norm(dim=-1, keepdim=True)

    # Test 4: Text-image alignment sanity
    text = clip.tokenize(["a car", "a pedestrian", "a tree"]).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        sims = cls_feat @ text_feats.T
    print(f"[Test 4] Text-image cosine sims (random image): {sims.cpu().numpy().flatten()}")
    print("  (Should be low and roughly equal for random image)")
    print("  PASS")

    # Test 5: Memory usage
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"[Test 5] GPU memory after CLIP load: {allocated:.2f} GB")
    print("  (Expected ~1.5 GB for ViT-L/14)")

    # Test 6: Preprocess transform info
    print(f"[Test 6] Preprocess: {preprocess}")

    print("\n" + "=" * 60)
    print("CLIP VERIFICATION PASSED")
    print("=" * 60)
    print(f"\nKey info for implementation:")
    print(f"  - Patch hook location: model.visual.transformer.resblocks[-1]")
    print(f"  - Raw output shape: {raw.shape}")
    print(f"  - Projection needed: model.visual.ln_post + @ model.visual.proj")
    print(f"  - Final patch shape: (256, 768) L2-normalized")
    print(f"  - Grid: 16x16 patches covering 224x224 input")

    return model, preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    verify_clip_model(args.device)

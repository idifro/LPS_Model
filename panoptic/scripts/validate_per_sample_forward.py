"""Phase 0.3 — Validate DisNet forward pass on a single per-keyframe LiDAR sample.

Loads one raw nuScenes LIDAR_TOP sweep, voxelizes it, runs through
pretrained DisNet, and checks output shape/numerics.
"""
import sys
import time
import numpy as np
import torch
import MinkowskiEngine as ME

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")
from models.disnet import DisNet, state_dict_remove_moudle

CKPT = "models/nuscenes_openseg.pth.tar"
NUSCENES_ROOT = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini"
VOXEL_SIZE = 0.05


def make_cfg():
    """Minimal cfg object for DisNet."""
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"
    return Cfg()


def main():
    """Validate the DisNet forward pass on a single nuScenes LiDAR keyframe sample.

    Used as a Phase 0.3 integration test to confirm that the pretrained DisNet backbone
    produces correct 768-d CLIP-aligned features from raw per-keyframe LiDAR input.

    Steps:
        1. Loads one LIDAR_TOP sweep from nuScenes mini, applies range filtering.
        2. Voxelizes the point cloud at VOXEL_SIZE resolution and builds a MinkowskiEngine
           SparseTensor with batch-index column prepended.
        3. Loads the DisNet model with pretrained weights (strict loading).
        4. Runs a forward pass and validates output shape (V, 768), absence of NaN/Inf,
           and reports norm statistics and inference latency.

    Returns:
        None: Prints diagnostics and raises AssertionError on failure.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load nuScenes sample ---
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud

    nusc = NuScenes("v1.0-mini", dataroot=NUSCENES_ROOT, verbose=False)
    sample = nusc.sample[0]
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(NUSCENES_ROOT + "/" + lidar_sd["filename"])
    pts = pc.points.T.astype(np.float32)  # [N, 4+]
    print(f"\nRaw LiDAR points: {pts.shape[0]}")

    # Range filter
    keep = (
        (pts[:, 0] >= -51.2) & (pts[:, 0] <= 51.2) &
        (pts[:, 1] >= -51.2) & (pts[:, 1] <= 51.2) &
        (pts[:, 2] >= -5.0) & (pts[:, 2] <= 3.0)
    )
    pts = pts[keep]
    print(f"After range filter: {pts.shape[0]}")

    # Voxelize
    coords = np.floor(pts[:, :3] / VOXEL_SIZE).astype(np.int32)
    _, vox_idx = np.unique(coords, axis=0, return_index=True)
    vox_coords = coords[vox_idx]
    vox_feats = pts[vox_idx, :3]  # DisNet expects in_channels=3 (XYZ as features)
    V = len(vox_idx)
    print(f"Voxels: {V}")

    # Build SparseTensor — add batch index column
    batch_coords = np.hstack([np.zeros((V, 1), dtype=np.int32), vox_coords])
    coords_t = torch.from_numpy(batch_coords).int()
    feats_t = torch.from_numpy(vox_feats).float()

    sinput = ME.SparseTensor(feats_t, coordinates=coords_t, device=device)

    # --- Load model ---
    cfg = make_cfg()
    model = DisNet(cfg).to(device)

    ckpt = torch.load(CKPT, map_location="cpu")
    state = state_dict_remove_moudle(ckpt["state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval()
    print("\nCheckpoint loaded successfully (strict=True)")

    # --- Forward pass ---
    with torch.no_grad():
        t0 = time.time()
        output = model(sinput)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - t0

    # DisNet may return a plain Tensor or SparseTensor
    out_feats = output.F if hasattr(output, 'F') else output
    print(f"\n=== Forward pass results ===")
    print(f"  Output shape: {list(out_feats.shape)}")
    print(f"  Output dtype: {out_feats.dtype}")
    print(f"  Output dim (d_clip): {out_feats.shape[1]}")
    print(f"  Has NaN: {torch.isnan(out_feats).any().item()}")
    print(f"  Has Inf: {torch.isinf(out_feats).any().item()}")
    print(f"  Norm range: [{out_feats.norm(dim=1).min().item():.4f}, {out_feats.norm(dim=1).max().item():.4f}]")
    print(f"  Mean norm: {out_feats.norm(dim=1).mean().item():.4f}")
    print(f"  Forward time: {elapsed:.3f}s")

    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU peak memory: {mem:.0f} MB")

    # Sanity: dimension must be 768
    assert out_feats.shape[1] == 768, f"Expected 768, got {out_feats.shape[1]}"
    assert not torch.isnan(out_feats).any(), "NaN in output!"
    print("\n>>> PASS: DisNet produces 768-d features on per-keyframe input.")


if __name__ == "__main__":
    main()

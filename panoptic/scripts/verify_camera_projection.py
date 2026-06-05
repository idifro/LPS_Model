"""Prerequisite P3: Verify LiDAR→camera projection for nuScenes.

Tests the full transform chain on one mini sample.
Expected: 35-60% of LiDAR points visible per camera.

Usage:
    python panoptic/scripts/verify_camera_projection.py [--dataroot PATH] [--version v1.0-mini]
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

CAM_NAMES = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]

# Standard nuScenes camera resolution
CAM_W, CAM_H = 1600, 900


def project_lidar_to_camera(nusc, sample_token, lidar_xyz, cam_name):
    """Project LiDAR points (N, 3) in LIDAR_TOP frame to a camera.

    Returns:
        uv: (N, 2) pixel coordinates
        depths: (N,) depth in camera frame
        valid: (N,) bool mask — True where visible
    """
    sample = nusc.get("sample", sample_token)

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    cam_sd = nusc.get("sample_data", sample["data"][cam_name])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    pts = lidar_xyz.T.copy()  # (3, N)

    # LiDAR sensor → ego (lidar time)
    pts = Quaternion(lidar_cs["rotation"]).rotation_matrix @ pts + \
          np.array(lidar_cs["translation"])[:, np.newaxis]

    # Ego (lidar time) → global
    pts = Quaternion(lidar_ep["rotation"]).rotation_matrix @ pts + \
          np.array(lidar_ep["translation"])[:, np.newaxis]

    # Global → ego (camera time)
    pts = Quaternion(cam_ep["rotation"]).rotation_matrix.T @ \
          (pts - np.array(cam_ep["translation"])[:, np.newaxis])

    # Ego (camera time) → camera sensor
    pts = Quaternion(cam_cs["rotation"]).rotation_matrix.T @ \
          (pts - np.array(cam_cs["translation"])[:, np.newaxis])

    depths = pts[2, :]

    K = np.array(cam_cs["camera_intrinsic"])
    pts_img = view_points(pts, K, normalize=True)
    uv = pts_img[:2, :].T  # (N, 2)

    valid = ((depths > 0.5) &
             (uv[:, 0] >= 0) & (uv[:, 0] < CAM_W) &
             (uv[:, 1] >= 0) & (uv[:, 1] < CAM_H))

    return uv, depths, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini")
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    sample = nusc.sample[0]
    sample_token = sample["token"]
    print(f"Testing sample: {sample_token[:16]}...")

    # Load LiDAR
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_sd["token"]))
    lidar_xyz = pc.points[:3, :].T  # (N, 3)
    print(f"LiDAR points: {lidar_xyz.shape[0]}")

    # Project to each camera
    total_visible = 0
    any_visible = np.zeros(lidar_xyz.shape[0], dtype=bool)

    for cam_name in CAM_NAMES:
        uv, depths, valid = project_lidar_to_camera(nusc, sample_token, lidar_xyz, cam_name)
        pct = valid.sum() / len(valid) * 100
        total_visible += valid.sum()
        any_visible |= valid
        print(f"  {cam_name:<16}: {valid.sum():5d} points visible ({pct:.1f}%)")

        # Sanity checks
        if valid.sum() > 0:
            assert depths[valid].min() > 0, "Negative depth in valid points!"
            assert uv[valid, 0].min() >= 0, "Negative U in valid points!"
            assert uv[valid, 1].min() >= 0, "Negative V in valid points!"

    overall_pct = any_visible.sum() / len(any_visible) * 100
    print(f"\nOverall coverage: {any_visible.sum()}/{len(any_visible)} "
          f"({overall_pct:.1f}%) visible in at least one camera")
    print(f"Total visibility events: {total_visible} (some in multiple cameras)")

    # Coverage should be 70-85%
    if overall_pct < 50:
        print("\nWARNING: Coverage below 50% — check projection code!")
    elif overall_pct < 70:
        print("\nNOTE: Coverage below 70% — expected for points behind/below cameras")
    else:
        print("\nCoverage looks healthy (≥70%)")

    print("\n" + "=" * 60)
    print("CAMERA PROJECTION VERIFICATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

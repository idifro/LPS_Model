"""
Convert nuScenes mini dataset to OpenScene format.

Produces:
  - data/nuscenes_3d/{split}/{scene_name}.pth   (3D point clouds + labels)
  - data/nuscenes_2d/{split}/{scene_name}/color/{cam}.jpg
  - data/nuscenes_2d/{split}/{scene_name}/pose/{cam}.npy   (4x4 cam2global)
  - data/nuscenes_2d/{split}/{scene_name}/K/{cam}.npy      (3x3 intrinsics)

Labels: Uses lidarseg .bin files if available, falls back to panoptic .npz
    (extracts semantic class via panoptic_value % 1000).

Usage:
  module load conda && conda activate openscene
  cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
  python scripts/preprocess/convert_nuscenes_mini.py
"""

import os
import math
import numpy as np
import cv2
import torch
from nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val
from pyquaternion import Quaternion

# ======================== CONFIG ========================
NUSCENES_ROOT = '/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini'
OUT_3D_ROOT = 'data/nuscenes_3d'
OUT_2D_ROOT = 'data/nuscenes_2d'
IMG_SIZE = (800, 450)  # (width, height)
ORIGINAL_IMG_SIZE = (1600, 900)

# Camera name mapping: nuScenes channel → OpenScene name
CAM_CHANNELS = [
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
]
CAM_NAMES = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']

# ======================== CLASS REMAP ========================
# From preprocess_3d_nuscenes.py — maps 32 nuScenes classes to 16 OpenScene classes
NUSCENES_CLASS_REMAP = 256 * np.ones(32, dtype=np.float64)
NUSCENES_CLASS_REMAP[2] = 7   # human.pedestrian.adult → person
NUSCENES_CLASS_REMAP[3] = 7   # human.pedestrian.child → person
NUSCENES_CLASS_REMAP[4] = 7   # human.pedestrian.construction_worker → person
NUSCENES_CLASS_REMAP[6] = 7   # human.pedestrian.police_officer → person
NUSCENES_CLASS_REMAP[9] = 1   # movable_object.barrier → barrier
NUSCENES_CLASS_REMAP[12] = 8  # movable_object.trafficcone → traffic cone
NUSCENES_CLASS_REMAP[14] = 2  # vehicle.bicycle → bicycle
NUSCENES_CLASS_REMAP[15] = 3  # vehicle.bus.bendy → bus
NUSCENES_CLASS_REMAP[16] = 3  # vehicle.bus.rigid → bus
NUSCENES_CLASS_REMAP[17] = 4  # vehicle.car → car
NUSCENES_CLASS_REMAP[18] = 5  # vehicle.construction → construction vehicle
NUSCENES_CLASS_REMAP[21] = 6  # vehicle.motorcycle → motorcycle
NUSCENES_CLASS_REMAP[22] = 9  # vehicle.trailer → trailer
NUSCENES_CLASS_REMAP[23] = 10 # vehicle.truck → truck
NUSCENES_CLASS_REMAP[24] = 11 # flat.driveable_surface → drivable surface
NUSCENES_CLASS_REMAP[25] = 12 # flat.other → other flat
NUSCENES_CLASS_REMAP[26] = 13 # flat.sidewalk → sidewalk
NUSCENES_CLASS_REMAP[27] = 14 # flat.terrain → terrain
NUSCENES_CLASS_REMAP[28] = 15 # static.manmade → manmade
NUSCENES_CLASS_REMAP[30] = 16 # static.vegetation → vegetation


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    """Adjust camera intrinsics for image resize.
    From preprocess_2d_nuscenes.py."""
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(
        image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def get_transform_matrix(record):
    """Build a 4x4 transform matrix from a nuScenes record with rotation + translation."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    T[:3, 3] = np.array(record['translation'])
    return T


def get_last_sample_token(nusc, scene):
    """Get the last sample token for a scene."""
    sample_token = scene['last_sample_token']
    return sample_token


def get_all_sample_tokens(nusc, scene):
    """Get all sample tokens for a scene in order."""
    tokens = []
    sample_token = scene['first_sample_token']
    while sample_token:
        tokens.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next'] if sample['next'] else None
    return tokens


def process_3d(nusc, scene, split, scene_name):
    """Process 3D data: aggregate all keyframes into one point cloud with labels."""
    out_dir = os.path.join(OUT_3D_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, scene_name + '.pth')
    if os.path.exists(out_path):
        print(f'  [3D] {out_path} already exists, skipping')
        return

    all_coords = []
    all_labels = []

    sample_tokens = get_all_sample_tokens(nusc, scene)
    print(f'  [3D] Processing {len(sample_tokens)} keyframes...')

    for sample_token in sample_tokens:
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', lidar_token)

        # Read LiDAR points
        lidar_path = os.path.join(NUSCENES_ROOT, sd_record['filename'])
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        xyz = points[:, :3]  # x, y, z

        # Read labels: try lidarseg first, fall back to panoptic
        lidarseg_path = os.path.join(
            NUSCENES_ROOT, 'lidarseg', 'v1.0-mini',
            lidar_token + '_lidarseg.bin')
        panoptic_path = os.path.join(
            NUSCENES_ROOT, 'panoptic', 'v1.0-mini',
            lidar_token + '_panoptic.npz')

        if os.path.exists(lidarseg_path):
            labels = np.fromfile(lidarseg_path, dtype=np.uint8)
        elif os.path.exists(panoptic_path):
            panoptic_data = np.load(panoptic_path)['data']
            labels = (panoptic_data % 1000).astype(np.uint8)
        else:
            raise FileNotFoundError(
                f'No lidarseg or panoptic labels found for {lidar_token}.\n'
                f'  Tried: {lidarseg_path}\n'
                f'  Tried: {panoptic_path}')

        assert len(xyz) == len(labels), \
            f'Point/label mismatch: {len(xyz)} vs {len(labels)} for {lidar_token}'

        # Transform to global frame
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        lidar2ego = get_transform_matrix(cs_record)

        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego2global = get_transform_matrix(pose_record)

        lidar2global = ego2global @ lidar2ego

        # Apply transform
        xyz_hom = np.hstack([xyz, np.ones((len(xyz), 1))])
        xyz_global = (lidar2global @ xyz_hom.T).T[:, :3]

        all_coords.append(xyz_global.astype(np.float32))
        all_labels.append(labels)

    coords = np.concatenate(all_coords, axis=0)
    labels_raw = np.concatenate(all_labels, axis=0)

    # Apply class remap: 32 → 16 classes, then subtract 1
    remapped_labels = NUSCENES_CLASS_REMAP[labels_raw]
    remapped_labels -= 1  # now 0-15 for valid, 255 for unmapped

    torch.save((coords, 0, remapped_labels), out_path)
    n_valid = (remapped_labels != 255).sum()
    print(f'  [3D] Saved {out_path}: {coords.shape[0]} points, '
          f'{n_valid} with valid labels ({n_valid/len(remapped_labels)*100:.1f}%)')


def process_2d(nusc, scene, split, scene_name):
    """Process 2D data: extract last keyframe's 6 camera images, poses, intrinsics."""
    scene_out_dir = os.path.join(OUT_2D_ROOT, split, scene_name)
    color_dir = os.path.join(scene_out_dir, 'color')
    pose_dir = os.path.join(scene_out_dir, 'pose')
    K_dir = os.path.join(scene_out_dir, 'K')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(K_dir, exist_ok=True)

    # Check if already done
    if len(os.listdir(color_dir)) == 6:
        print(f'  [2D] {scene_out_dir} already exists, skipping')
        return

    # Use the last sample
    last_sample_token = get_last_sample_token(nusc, scene)
    sample = nusc.get('sample', last_sample_token)

    for cam_channel, cam_name in zip(CAM_CHANNELS, CAM_NAMES):
        cam_token = sample['data'][cam_channel]
        cam_sd = nusc.get('sample_data', cam_token)

        # Read and resize image
        img_path = os.path.join(NUSCENES_ROOT, cam_sd['filename'])
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(os.path.join(color_dir, cam_name + '.jpg'), img_resized)

        # Compute cam2global transform
        cs_record = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam2ego = get_transform_matrix(cs_record)

        pose_record = nusc.get('ego_pose', cam_sd['ego_pose_token'])
        ego2global = get_transform_matrix(pose_record)

        cam2global = ego2global @ cam2ego
        np.save(os.path.join(pose_dir, cam_name + '.npy'), cam2global)

        # Extract and adjust intrinsics
        K = np.array(cs_record['camera_intrinsic'])  # 3x3
        K = adjust_intrinsic(K, intrinsic_image_dim=ORIGINAL_IMG_SIZE, image_dim=IMG_SIZE)
        np.save(os.path.join(K_dir, cam_name + '.npy'), K)

    print(f'  [2D] Saved 6 cameras to {scene_out_dir}')


def main():
    print(f'Loading nuScenes mini from {NUSCENES_ROOT}...')
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_ROOT, verbose=False)

    # Build scene name to scene record mapping
    scene_map = {s['name']: s for s in nusc.scene}

    splits = {
        'train': mini_train,
        'val': mini_val,
    }

    for split, scene_names in splits.items():
        print(f'\n=== Processing {split} split ({len(scene_names)} scenes) ===')
        for scene_name in sorted(scene_names):
            if scene_name not in scene_map:
                print(f'  WARNING: {scene_name} not found in dataset, skipping')
                continue

            scene = scene_map[scene_name]
            print(f'\nProcessing {scene_name} ({scene["description"][:60]}...)')

            process_3d(nusc, scene, split, scene_name)
            process_2d(nusc, scene, split, scene_name)

    print('\n=== Done! ===')
    print(f'3D data: {OUT_3D_ROOT}/')
    print(f'2D data: {OUT_2D_ROOT}/')


if __name__ == '__main__':
    main()

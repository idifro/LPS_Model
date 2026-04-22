"""Custom collate for NuScenesPanoptic dataset.

Handles variable-length GT masks and prepares batched sparse coordinates
with batch index in column 0 (MinkowskiEngine convention).
"""
import torch


def panoptic_collate(batch):
    """Collate a list of dataset dicts into a batched dict.

    Returns:
        coords:     (sum(Vi), 4) int32   — col0=batch_idx, col1-3=voxel coords
        feats:      (sum(Vi), 3) float32 — XYZ features
        gt_masks:   list of (Mi, Vi) bool tensors (one per sample, variable M and V)
        gt_classes: list of (Mi,) long tensors
        gt_sem:     list of (Vi,) long tensors
        inv_idx:    list of (Ni,) long tensors
        cam_paths:  list of list[6] str
        tokens:     list of str
        num_voxels: list of int  — for reconstructing per-sample outputs
    """
    coords_list = []
    feats_list = []
    gt_masks_list = []
    gt_classes_list = []
    gt_sem_list = []
    inv_idx_list = []
    cam_paths_list = []
    tokens_list = []
    num_voxels_list = []

    for b_idx, sample in enumerate(batch):
        V = sample["num_voxels"]
        # Prepend batch index to coordinates
        batch_col = torch.full((V, 1), b_idx, dtype=torch.int32)
        coords_list.append(torch.cat([batch_col, sample["vox_coords"]], dim=1))
        feats_list.append(sample["vox_feats"])
        gt_masks_list.append(sample["gt_masks"])
        gt_classes_list.append(sample["gt_classes"])
        gt_sem_list.append(sample["gt_sem"])
        inv_idx_list.append(sample["inv_idx"])
        cam_paths_list.append(sample["cam_paths"])
        tokens_list.append(sample["token"])
        num_voxels_list.append(V)

    return {
        "coords": torch.cat(coords_list, dim=0),         # (sum(V), 4)
        "feats": torch.cat(feats_list, dim=0),            # (sum(V), 3)
        "gt_masks": gt_masks_list,                         # list of tensors
        "gt_classes": gt_classes_list,                     # list of tensors
        "gt_sem": gt_sem_list,                             # list of tensors
        "inv_idx": inv_idx_list,                           # list of tensors
        "cam_paths": cam_paths_list,                       # list of list[6]
        "tokens": tokens_list,                             # list of str
        "num_voxels": num_voxels_list,                     # list of int
    }

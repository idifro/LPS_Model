"""Voxelizer module for converting continuous 3D point clouds into discrete voxel grids.

This module provides the Voxelizer class and a rotation matrix helper function,
originally taken from BPNet (CVPR'21). It is used in the data loading pipeline
to transform raw point cloud coordinates into quantized voxel coordinates suitable
for sparse convolutional networks. The module supports optional data augmentation
via random rotation, scaling, and translation, as well as spatial clipping to
restrict the point cloud to a bounded region.

Reference:
    https://github.com/wbhu/BPNet/blob/main/dataset/voxelizer.py
"""

import collections
import numpy as np
from dataset.voxelization_utils import sparse_quantize
from scipy.linalg import expm, norm


def M(axis, theta):
    """Compute a 3x3 rotation matrix for a given axis and angle using the matrix exponential.

    This utility is called by ``Voxelizer.get_transformation_matrix`` whenever
    rotation augmentation is enabled. It constructs the skew-symmetric matrix
    from the normalized axis scaled by theta, then exponentiates it to obtain
    a proper rotation matrix (Rodrigues' formula in exponential-map form).

    Steps:
        1. Normalize the axis vector to unit length.
        2. Scale the unit axis by the rotation angle ``theta``.
        3. Build a 3x3 skew-symmetric matrix via ``np.cross(np.eye(3), ...)``.
        4. Compute the matrix exponential with ``scipy.linalg.expm`` to yield
           the rotation matrix.

    Args:
        axis (numpy.ndarray): A 3-element array specifying the rotation axis
            (does not need to be unit length; it is normalized internally).
        theta (float): The rotation angle in radians.

    Returns:
        numpy.ndarray: A 3x3 orthogonal rotation matrix of dtype float64.
    """
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:
    """Converts continuous 3D point clouds into discrete voxel representations.

    The Voxelizer is instantiated once per dataset configuration and reused for
    every sample during data loading. It encapsulates the full voxelization
    pipeline: optional spatial clipping, random augmentation (rotation, scale,
    translation), coordinate quantization via floor division, and duplicate
    removal through sparse hashing.

    Typical usage in the data-loading pipeline:
        1. A dataset ``__getitem__`` method loads raw (x, y, z) coordinates,
           per-point features, and labels.
        2. ``voxelize(coords, feats, labels)`` is called, which internally
           calls ``clip`` (if bounds are set) and ``get_transformation_matrix``
           to obtain the rigid transformation, then quantizes and deduplicates
           the resulting voxel coordinates.

    Attributes:
        voxel_size (float): Side length of a single voxel cell.
        clip_bound (list or None): Per-axis (min, max) clipping boundaries.
        ignore_label (int): Label value used for ignored / unlabeled points.
        use_augmentation (bool): Whether to apply random augmentation.
        scale_augmentation_bound (tuple or None): (low, high) multiplier range.
        rotation_augmentation_bound (tuple or None): Per-axis rotation ranges.
        translation_augmentation_ratio_bound (tuple or None): Per-axis translation ratio ranges.
    """

    def __init__(self,
                 voxel_size=1,
                 clip_bound=None,
                 use_augmentation=False,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):
        '''
        Args:
          voxel_size: side length of a voxel
          clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
          scale_augmentation_bound: None or (0.9, 1.1)
          rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
            Use random order of x, y, z to prevent bias.
          translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
          ignore_label: label assigned for ignore (not a training label).
        '''
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    def get_transformation_matrix(self):
        """Compute the 4x4 voxelization and rotation transformation matrices.

        Called once per sample inside ``voxelize`` to build the rigid
        transformation that maps raw point-cloud coordinates into voxel space.
        When augmentation is disabled, the rotation matrix is the identity and
        the voxelization matrix is a uniform scale of ``1 / voxel_size``.

        Steps:
            1. If rotation augmentation is enabled, sample a random angle for
               each axis within its configured bounds, compute a rotation
               matrix for each axis via ``M(axis, theta)``, shuffle the order
               to avoid axis bias, and compose them by matrix multiplication.
            2. Build a diagonal voxelization (scale) matrix equal to
               ``1 / voxel_size``, optionally multiplied by a random scale
               factor drawn from ``scale_augmentation_bound``.
            3. Return both 4x4 matrices so the caller can compose them.

        Args:
            None.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A pair of 4x4 float64
                matrices ``(voxelization_matrix, rotation_matrix)`` where
                the voxelization matrix encodes scaling and the rotation
                matrix encodes the (possibly augmented) rotation.
        """
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1 / self.voxel_size
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None, trans_aug_ratio=None):
        """Compute a boolean mask that selects points inside the configured clip bounds.

        Called by ``voxelize`` when ``self.clip_bound`` is not None. The clip
        region is an axis-aligned box defined by ``self.clip_bound`` and
        centered around the point-cloud centroid (or a supplied center),
        optionally shifted by a translation augmentation ratio.

        Steps:
            1. Compute the bounding box (min/max) of the input coordinates.
            2. Determine the center as the midpoint of the bounding box, or
               use the provided ``center``.
            3. If ``trans_aug_ratio`` is given, shift the center by the
               element-wise product of the ratio and the bounding-box size.
            4. Build a per-point boolean mask where each axis coordinate lies
               within ``[clip_min + center, clip_max + center)``.

        Args:
            coords (numpy.ndarray): Point coordinates of shape (N, 3).
            center (numpy.ndarray or None): Optional 3-element array for the
                clip-box center. If None, the centroid of ``coords`` is used.
            trans_aug_ratio (numpy.ndarray or None): Optional 3-element
                translation augmentation ratio applied to the bounding-box size.

        Returns:
            numpy.ndarray: A boolean array of shape (N,) that is True for
                points inside the clip bounds and False otherwise.
        """
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) &
                     (coords[:, 0] < (lim[0][1] + center[0])) &
                     (coords[:, 1] >= (lim[1][0] + center[1])) &
                     (coords[:, 1] < (lim[1][1] + center[1])) &
                     (coords[:, 2] >= (lim[2][0] + center[2])) &
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None, link=None, return_ind=False):
        """Execute the full voxelization pipeline on a single point-cloud sample.

        This is the main entry point called by dataset ``__getitem__`` methods.
        It chains clipping, augmentation, quantization, and deduplication to
        produce sparse voxel coordinates ready for a sparse convolutional
        network.

        Steps:
            1. If ``clip_bound`` is set, optionally sample a translation
               augmentation ratio, then call ``clip`` to remove out-of-bounds
               points and filter coords/feats/labels accordingly.
            2. Call ``get_transformation_matrix`` to obtain voxelization (scale)
               and rotation matrices. Compose them if augmentation is active.
            3. Convert coordinates to homogeneous form, apply the rigid
               transformation, and floor the result to obtain integer voxel
               coordinates.
            4. Translate all coordinates so the minimum is at the origin.
            5. Call ``sparse_quantize`` to find unique voxel indices and a
               reconstruction mapping from original to unique indices.
            6. If features contain normals (columns 3-6), rotate the normals
               by the rotation matrix.
            7. Return the deduplicated voxel coordinates, features, labels,
               reconstruction indices, and optionally the raw unique indices
               or a link array subset.

        Args:
            coords (numpy.ndarray): Raw point coordinates of shape (N, 3).
            feats (numpy.ndarray): Per-point feature matrix of shape (N, D).
            labels (numpy.ndarray or None): Per-point labels of shape (N,).
            center (numpy.ndarray or None): Optional clip-box center override.
            link (numpy.ndarray or None): Optional per-point link indices; if
                provided, the deduplicated subset is returned as the last
                element.
            return_ind (bool): If True, return the raw unique indices as the
                last element instead of link-based output.

        Returns:
            tuple: A tuple whose contents vary by flags:
                - ``(coords_aug, feats, labels, inds_reconstruct)`` by default.
                - ``(coords_aug, feats, labels, inds_reconstruct, inds)`` when
                  ``return_ind`` is True.
                - ``(coords_aug, feats, labels, inds_reconstruct, link[inds])``
                  when ``link`` is provided.
        """
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        if self.clip_bound is not None:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            clip_inds = self.clip(coords, center, trans_aug_ratio)
            if clip_inds.sum():
                coords, feats = coords[clip_inds], feats[clip_inds]
                if labels is not None:
                    labels = labels[clip_inds]

        # Get rotation and scale
        M_v, M_r = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_v
        if self.use_augmentation:
            rigid_transformation = M_r @ rigid_transformation

        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

        # Align all coordinates to the origin.
        min_coords = coords_aug.min(0)
        M_t = np.eye(4)
        M_t[:3, -1] = -min_coords
        rigid_transformation = M_t @ rigid_transformation
        coords_aug = np.floor(coords_aug - min_coords)

        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)
        coords_aug, feats, labels = coords_aug[inds], feats[inds], labels[inds]

        # Normal rotation
        if feats.shape[1] > 6:
            feats[:, 3:6] = feats[:, 3:6] @ (M_r[:3, :3].T)

        if return_ind:
            return coords_aug, feats, labels, np.array(inds_reconstruct), inds
        if link is not None:
            return coords_aug, feats, labels, np.array(inds_reconstruct), link[inds]

        return coords_aug, feats, labels, np.array(inds_reconstruct)

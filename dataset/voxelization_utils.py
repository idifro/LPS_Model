"""Voxelization utility functions for hashing and quantizing 3D coordinates.

This module provides hash functions (FNV-1a and ravel-based) and a
``sparse_quantize`` routine used by the Voxelizer class to convert
continuous point-cloud coordinates into unique discrete voxel indices.
Originally from the Minkowski Engine codebase accompanying the paper
"4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks"
(CVPR 2019, https://arxiv.org/abs/1904.08755).
"""

import torch
import numpy as np
from collections import Sequence


def fnv_hash_vec(arr):
    """Compute FNV64-1A hashes for each row of a 2D integer coordinate array.

    Used by ``sparse_quantize`` when ``hash_type='fnv'`` to produce a unique
    hash per voxel coordinate, enabling fast duplicate detection via
    ``np.unique``. The FNV-1a variant multiplies by the FNV prime then
    XORs each coordinate column into the running hash.

    Steps:
        1. Cast the input array to uint64.
        2. Initialize each row's hash to the FNV offset basis (14695981039346656037).
        3. For each column, multiply the hash by the FNV prime (1099511628211)
           and XOR with the column value.
        4. Return the resulting per-row hash array.

    Args:
        arr (numpy.ndarray): A 2D integer array of shape (N, D) representing
            N discretized coordinates in D dimensions.

    Returns:
        numpy.ndarray: A 1D uint64 array of shape (N,) containing the
            FNV64-1A hash for each row.
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
                 np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """Compute a ravel-style hash for each row of a 2D integer coordinate array.

    Used by ``sparse_quantize`` when ``hash_type='ravel'`` (the default) to
    map multi-dimensional voxel coordinates to unique 1D keys. This is
    analogous to ``np.ravel_multi_index`` but operates in Fortran order and
    handles negative coordinates by shifting all values to non-negative.

    Steps:
        1. Subtract the per-column minimum so all values are >= 0.
        2. Cast to uint64 and compute the per-column max + 1 as the radix.
        3. Accumulate a mixed-radix representation by iterating over columns
           in Fortran (column-major) order: ``key = key * radix + column``.
        4. Return the resulting per-row hash array.

    Args:
        arr (numpy.ndarray): A 2D integer array of shape (N, D) representing
            N discretized coordinates in D dimensions.

    Returns:
        numpy.ndarray: A 1D uint64 array of shape (N,) containing the
            ravel hash for each row.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=255,
                    set_ignore_label_when_collision=False,
                    return_index=False,
                    hash_type='fnv',
                    quantization_size=1):
    r'''Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates.

    Args:
        coords (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a matrix of size
        :math:`N \times D` where :math:`N` is the number of points in the
        :math:`D` dimensional space.

        feats (:attr:`numpy.ndarray` or :attr:`torch.Tensor`, optional): a matrix of size
        :math:`N \times D_F` where :math:`N` is the number of points and
        :math:`D_F` is the dimension of the features.

        labels (:attr:`numpy.ndarray`, optional): labels associated to eah coordinates.

        ignore_label (:attr:`int`, optional): the int value of the IGNORE LABEL.

        set_ignore_label_when_collision (:attr:`bool`, optional): use the `ignore_label`
        when at least two points fall into the same cell.

        return_index (:attr:`bool`, optional): True if you want the indices of the
        quantized coordinates. False by default.

        hash_type (:attr:`str`, optional): Hash function used for quantization. Either
        `ravel` or `fnv`. `ravel` by default.

        quantization_size (:attr:`float`, :attr:`list`, or
        :attr:`numpy.ndarray`, optional): the length of the each side of the
        hyperrectangle of of the grid cell.

    .. note::
        Please check `examples/indoor.py` for the usage.

    '''
    use_label = labels is not None
    use_feat = feats is not None
    if not use_label and not use_feat:
        return_index = True

    assert hash_type in [
        'ravel', 'fnv'
    ], "Invalid hash_type. Either ravel, or fnv allowed. You put hash_type=" + hash_type
    assert coords.ndim == 2, \
        "The coordinates must be a 2D matrix. The shape of the input is " + str(coords.shape)
    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]
    if use_label:
        assert coords.shape[0] == len(labels)

    # Quantize the coordinates
    dimension = coords.shape[1]
    if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
        assert len(
            quantization_size
        ) == dimension, "Quantization size and coordinates size mismatch."
        quantization_size = [i for i in quantization_size]
    elif np.isscalar(quantization_size):  # Assume that it is a scalar
        quantization_size = [quantization_size for i in range(dimension)]
    else:
        raise ValueError('Not supported type for quantization_size.')
    discrete_coords = np.floor(coords / np.array(quantization_size))

    # Hash function type
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coords)
    else:
        key = fnv_hash_vec(discrete_coords)

    if use_label:
        _, inds, counts = np.unique(key, return_index=True, return_counts=True)
        filtered_labels = labels[inds]
        if set_ignore_label_when_collision:
            filtered_labels[counts > 1] = ignore_label
        if return_index:
            return inds, filtered_labels
        else:
            return discrete_coords[inds], feats[inds], filtered_labels
    else:
        _, inds, inds_reverse = np.unique(key, return_index=True, return_inverse=True)
        if return_index:
            return inds, inds_reverse
        else:
            if use_feat:
                return discrete_coords[inds], feats[inds]
            else:
                return discrete_coords[inds]

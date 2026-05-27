"""3D point cloud augmentation transforms for sparse tensor data.

This module provides feature-space and coordinate-space augmentation
transforms designed for 3D point cloud processing pipelines. Feature
transforms (chromatic translation, auto-contrast, jitter, hue-saturation)
operate on per-point RGB colour channels, while coordinate transforms
(random horizontal flip, elastic distortion) operate on XYZ positions.

All transforms follow the ``__call__(coords, feats, labels)`` convention
so they can be composed via the :class:`Compose` helper. They are used
in the dataset loading stage before voxelisation / sparse-tensor creation.
"""

import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class ChromaticTranslation(object):
    """Apply a uniform random colour translation to point cloud RGB features.

    A single random offset vector is sampled and added to every point's RGB
    channels, which shifts the overall colour tone of the scene. This is a
    feature-space augmentation that helps the model become invariant to
    global illumination changes.

    The transform is applied with 95 % probability; otherwise the input is
    returned unchanged. Results are clipped to [0, 255].

    Args:
        trans_range_ratio (float): Controls the magnitude of the translation.
            The offset per channel is ``255 * 2 * trans_range_ratio * U``
            where ``U ~ Uniform(-0.5, 0.5)``.
    """

    def __init__(self, trans_range_ratio=1e-1):
        """Initialise ChromaticTranslation with the given range ratio.

        Stores the translation range ratio that governs the maximum
        colour shift applied to point features during augmentation.

        Args:
            trans_range_ratio (float): Ratio controlling the magnitude of
                the random colour offset. Default is ``0.1``.
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels):
        """Apply the chromatic translation to point cloud features.

        With 95 % probability, a single random 3-channel offset is sampled
        and added to the RGB columns of ``feats``. Values are clipped to
        the valid [0, 255] range.

        Args:
            coords (np.ndarray): Point coordinates of shape ``(N, 3)``.
            feats (np.ndarray): Point features of shape ``(N, C)`` where
                the first 3 columns are RGB in [0, 255].
            labels (np.ndarray): Per-point labels of shape ``(N,)``.

        Returns:
            tuple: ``(coords, feats, labels)`` with ``feats`` potentially
            modified in-place.
        """
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels


class ChromaticAutoContrast(object):
    """Stretch per-channel intensity range to [0, 255] and blend with the original.

    This augmentation computes a contrast-enhanced version of the point
    features by linearly rescaling each RGB channel so that the minimum
    maps to 0 and the maximum maps to 255. The enhanced result is blended
    with the original features using a configurable (or random) blend
    factor. Applied with 20 % probability.

    Used as a feature-space augmentation in 3D scene understanding
    pipelines to improve robustness to varying contrast conditions.
    """

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        """Initialise ChromaticAutoContrast.

        Stores blending parameters that control how strongly the
        contrast-stretched features replace the originals.

        Args:
            randomize_blend_factor (bool): If ``True``, sample a uniform
                random blend factor in [0, 1] at each call. Default ``True``.
            blend_factor (float): Fixed blend factor used when
                ``randomize_blend_factor`` is ``False``. Default ``0.5``.
        """
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels):
        """Apply auto-contrast augmentation to the point cloud features.

        With 20 % probability, rescales each RGB channel of ``feats``
        so the per-channel min is 0 and max is 255, then blends the
        rescaled result with the original using the configured blend
        factor.

        Args:
            coords (np.ndarray): Point coordinates, shape ``(N, 3)``.
            feats (np.ndarray): Point features, shape ``(N, C)``; first
                3 columns are RGB in [0, 255].
            labels (np.ndarray): Per-point labels, shape ``(N,)``.

        Returns:
            tuple: ``(coords, feats, labels)`` with ``feats`` possibly
            modified.
        """
        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = np.min(feats, 0, keepdims=True)
            hi = np.max(feats, 0, keepdims=True)

            scale = 255 / (hi - lo)

            contrast_feats = (feats - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats, labels


class ChromaticJitter(object):
    """Add per-point Gaussian noise to RGB features.

    Samples independent Gaussian noise for every point and adds it to the
    first three feature channels (RGB). This simulates sensor noise and
    minor colour variations, encouraging the model to be robust to small
    colour perturbations. Applied with 95 % probability.
    """

    def __init__(self, std=0.01):
        """Initialise ChromaticJitter with the given noise standard deviation.

        The actual noise magnitude in colour-space is ``std * 255``.

        Args:
            std (float): Standard deviation of the noise relative to the
                [0, 1] colour range. Default ``0.01``.
        """
        self.std = std

    def __call__(self, coords, feats, labels):
        """Apply per-point Gaussian colour jitter.

        With 95 % probability, independent Gaussian noise scaled by
        ``self.std * 255`` is added to every point's RGB channels.
        Results are clipped to [0, 255].

        Args:
            coords (np.ndarray): Point coordinates, shape ``(N, 3)``.
            feats (np.ndarray): Point features, shape ``(N, C)``; first
                3 columns are RGB in [0, 255].
            labels (np.ndarray): Per-point labels, shape ``(N,)``.

        Returns:
            tuple: ``(coords, feats, labels)`` with ``feats`` possibly
            modified in-place.
        """
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return coords, feats, labels


class HueSaturationTranslation(object):
    """Randomly shift hue and scale saturation of point cloud RGB features.

    Converts the first three feature columns from RGB to HSV, applies a
    random additive hue shift and multiplicative saturation scaling, then
    converts back to RGB. This augmentation increases colour diversity and
    helps the model generalise across different lighting conditions.

    The conversion follows the algorithm from Python's ``colorsys`` module
    but is vectorised with NumPy for efficiency on large point clouds.
    """

    @staticmethod
    def rgb_to_hsv(rgb):
        """Convert an array of RGB values to HSV colour space.

        Implements the same algorithm as ``colorsys.rgb_to_hsv`` but in a
        fully vectorised NumPy form. R, G, B values are expected in
        [0, 255]; the returned H and S are in [0, 1] and V is in
        [0, 255]. An optional alpha channel (column index 3+) is
        preserved unchanged.

        Args:
            rgb (np.ndarray): Array of shape ``(N, 3)`` or ``(N, 4+)``
                with RGB (and optionally alpha) values in [0, 255].

        Returns:
            np.ndarray: HSV array of the same shape as ``rgb`` with H in
            [0, 1], S in [0, 1], and V in [0, 255].
        """
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        """Convert an array of HSV values back to RGB colour space.

        Implements the inverse of :meth:`rgb_to_hsv` using the same
        vectorised approach. H and S are expected in [0, 1], V in
        [0, 255]. Any alpha channel beyond column index 2 is copied
        through unchanged.

        Args:
            hsv (np.ndarray): Array of shape ``(N, 3)`` or ``(N, 4+)``
                with H in [0, 1], S in [0, 1], V in [0, 255].

        Returns:
            np.ndarray: RGB uint8 array of the same shape with values in
            [0, 255].
        """
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        """Initialise HueSaturationTranslation with range limits.

        Stores the maximum hue shift and saturation scaling factor used
        when augmenting point cloud colour features.

        Args:
            hue_max (float): Maximum absolute hue shift in [0, 1].
                The actual shift is sampled uniformly from
                ``[-hue_max, hue_max]``.
            saturation_max (float): Maximum relative saturation change.
                The saturation is multiplied by a factor sampled
                uniformly from ``[1 - saturation_max, 1 + saturation_max]``.
        """
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels):
        """Apply random hue shift and saturation scaling to RGB features.

        Converts ``feats[:, :3]`` from RGB to HSV, adds a random hue
        offset, multiplies saturation by a random factor, converts back
        to RGB, and clips results to [0, 255].

        Args:
            coords (np.ndarray): Point coordinates, shape ``(N, 3)``.
            feats (np.ndarray): Point features, shape ``(N, C)``; first
                3 columns are RGB in [0, 255].
            labels (np.ndarray): Per-point labels, shape ``(N,)``.

        Returns:
            tuple: ``(coords, feats, labels)`` with augmented RGB in
            ``feats[:, :3]``.
        """
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return coords, feats, labels


##############################
# Coordinate transformations
##############################
class RandomHorizontalFlip(object):
    """Randomly flip point cloud coordinates along horizontal axes.

    For each horizontal axis (all spatial axes except the designated
    upright axis), the coordinates are independently reflected with 50 %
    probability. This is a coordinate-space augmentation that helps the
    model become invariant to mirror symmetries of indoor/outdoor scenes.

    Applied with an outer 95 % probability gate; if that gate fails the
    input is returned unchanged.
    """

    def __init__(self, upright_axis, is_temporal):
        """Initialise RandomHorizontalFlip.

        Determines which spatial axes are horizontal (i.e. not the
        upright axis) and therefore eligible for flipping. Supports an
        optional temporal dimension.

        Args:
            upright_axis (str): One of ``'x'``, ``'y'``, ``'z'``
                identifying the gravity-aligned axis that should *not*
                be flipped.
            is_temporal (bool): If ``True``, the coordinate tensor has 4
                columns (x, y, z, t) instead of 3.
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels):
        """Flip point coordinates along randomly chosen horizontal axes.

        With 95 % outer probability, iterates over each horizontal axis
        and independently flips it with 50 % probability by reflecting
        coordinates around the axis maximum: ``x' = max(x) - x``.

        Args:
            coords (np.ndarray): Point coordinates, shape ``(N, D)``
                where ``D`` is 3 or 4 (if temporal).
            feats (np.ndarray): Point features, shape ``(N, C)``.
            labels (np.ndarray): Per-point labels, shape ``(N,)``.

        Returns:
            tuple: ``(coords, feats, labels)`` with ``coords`` possibly
            modified in-place.
        """
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels


class ElasticDistortion:
    """Apply elastic distortion to 3D point cloud coordinates.

    Creates smooth, spatially-varying random displacements by generating
    coarse-grained Gaussian noise, smoothing it with repeated box-filter
    convolutions, and interpolating the result back to the original point
    positions. This coordinate-space augmentation simulates non-rigid
    deformations and is applied with 95 % probability.
    """

    def __init__(self, distortion_params):
        """Initialise ElasticDistortion with granularity/magnitude pairs.

        Each pair defines one pass of elastic distortion with a
        particular noise grid resolution and displacement scale.
        Multiple pairs are applied sequentially.

        Args:
            distortion_params (list[tuple[float, float]] | None): List of
                ``(granularity, magnitude)`` pairs. ``granularity``
                controls the spatial frequency of the noise (in the same
                units as the coordinates), and ``magnitude`` scales the
                displacement. If ``None``, the transform is a no-op.
        """
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, granularity, magnitude):
        '''Apply elastic distortion on sparse coordinate space.

          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        '''
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min +
                                       granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords = coords + interp(coords) * magnitude
        return coords

    def __call__(self, pointcloud):
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    pointcloud = self.elastic_distortion(pointcloud, granularity, magnitude)
        return pointcloud


class Compose(object):
    '''Composes several transforms together.'''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class cfl_collate_fn_factory:
    '''Generates collate function for coords, feats, labels.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    '''

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch = [], [], []

        batch_id = 0
        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                    f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
                )
                break
            coords_batch.append(
                torch.cat((torch.from_numpy(coords[batch_id]).int(),
                           torch.ones(num_points, 1).int() * batch_id), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())

            batch_id += 1

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).int()
        return coords_batch, feats_batch, labels_batch


class cflt_collate_fn_factory:
    '''Generates collate function for coords, feats, labels, point_clouds, transformations.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    '''

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels, pointclouds, transformations = list(zip(*list_data))
        cfl_collate_fn = cfl_collate_fn_factory(limit_numpoints=self.limit_numpoints)
        coords_batch, feats_batch, labels_batch = cfl_collate_fn(list(zip(coords, feats, labels)))
        num_truncated_batch = coords_batch[:, -1].max().item() + 1

        batch_id = 0
        pointclouds_batch, transformations_batch = [], []
        for pointcloud, transformation in zip(pointclouds, transformations):
            if batch_id >= num_truncated_batch:
                break
            pointclouds_batch.append(
                torch.cat((torch.from_numpy(pointcloud), torch.ones(pointcloud.shape[0], 1) * batch_id),
                          1))
            transformations_batch.append(
                torch.cat(
                    (torch.from_numpy(transformation), torch.ones(transformation.shape[0], 1) * batch_id),
                    1))
            batch_id += 1

        pointclouds_batch = torch.cat(pointclouds_batch, 0).float()
        transformations_batch = torch.cat(transformations_batch, 0).float()
        return coords_batch, feats_batch, labels_batch, pointclouds_batch, transformations_batch

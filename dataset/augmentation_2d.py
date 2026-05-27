"""2D image and label augmentation transforms for semantic segmentation.

This module provides a collection of image-level augmentation transforms
designed for 2D semantic segmentation pipelines. Each transform operates
on an ``(image, label)`` pair, where ``image`` is an H x W x C numpy
array (or torch tensor after :class:`ToTensor`) and ``label`` is an
H x W integer array of per-pixel class indices.

Transforms follow the ``__call__(image, label)`` convention and can be
composed using :class:`Compose`. They are typically used in the dataset
``__getitem__`` method before batching.
"""

import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch


class Compose(object):
    """Chain multiple image/label transforms into a single callable.

    Sequentially applies a list of transforms to an ``(image, label)``
    pair, passing the output of one transform as the input to the next.
    This is the primary mechanism for building augmentation pipelines in
    the 2D segmentation data loader.

    Example usage::

        pipeline = Compose([RandScale([0.5, 2.0]), ToTensor()])
        image, label = pipeline(image, label)
    """

    def __init__(self, segtransform):
        """Initialise Compose with an ordered list of transforms.

        Stores the list of transforms that will be applied sequentially
        when the instance is called.

        Args:
            segtransform (list): Ordered list of callable transform
                objects, each accepting and returning ``(image, label)``.
        """
        self.segtransform = segtransform

    def __call__(self, image, label):
        """Apply all transforms sequentially to the image/label pair.

        Iterates through the stored transforms in order, feeding each
        output into the next transform's input.

        Args:
            image (np.ndarray): Input image of shape ``(H, W, C)`` or
                ``(H, W)``.
            label (np.ndarray): Segmentation label map of shape
                ``(H, W)`` with integer class indices.

        Returns:
            tuple: ``(image, label)`` after all transforms have been
            applied.
        """
        for t in self.segtransform:
            image, label = t(image, label)
        return image, label


class ToTensor(object):
    """Convert numpy image and label arrays to PyTorch tensors.

    Converts an ``(H, W, C)`` numpy image to a ``(C, H, W)``
    ``torch.FloatTensor`` and an ``(H, W)`` numpy label to a
    ``torch.LongTensor``. Grayscale images of shape ``(H, W)`` are
    automatically expanded to ``(H, W, 1)`` before transposing.

    This transform is typically placed near the end of the augmentation
    pipeline, after all spatial and colour transforms that operate on
    numpy arrays.
    """

    def __call__(self, image, label):
        """Convert the image and label from numpy arrays to tensors.

        Transposes the image from ``(H, W, C)`` to ``(C, H, W)``,
        converts to ``torch.FloatTensor``, and converts the label to
        ``torch.LongTensor``. Validates input types and dimensions.

        Args:
            image (np.ndarray): Image array of shape ``(H, W, C)`` or
                ``(H, W)``.
            label (np.ndarray): Label array of shape ``(H, W)`` with
                integer class indices.

        Returns:
            tuple: ``(image, label)`` as ``(torch.FloatTensor,
            torch.LongTensor)``.

        Raises:
            RuntimeError: If inputs are not numpy arrays or have
                unexpected dimensions.
        """
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label


class Normalize(object):
    """Normalize a tensor image channel-wise with mean and standard deviation.

    Subtracts the per-channel mean and optionally divides by the
    per-channel standard deviation: ``channel = (channel - mean) / std``.
    Operates on a ``torch.FloatTensor`` of shape ``(C, H, W)`` (i.e.
    after :class:`ToTensor`). The label tensor is passed through
    unchanged.

    This is typically the last step in the augmentation pipeline before
    the image is fed to the neural network.
    """

    def __init__(self, mean, std=None):
        """Initialise Normalize with per-channel mean and optional std.

        If ``std`` is ``None``, only mean subtraction is performed.
        Otherwise both mean and std must have the same length (equal to
        the number of image channels).

        Args:
            mean (list[float]): Per-channel mean values to subtract.
            std (list[float] | None): Per-channel standard deviation
                values to divide by. Default ``None`` (no division).
        """
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        """Apply channel-wise normalisation to the image tensor.

        Subtracts ``self.mean`` and, if provided, divides by
        ``self.std`` for each channel of the image in-place.

        Args:
            image (torch.FloatTensor): Image tensor of shape
                ``(C, H, W)``.
            label (torch.LongTensor): Label tensor of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` with ``image`` normalised
            in-place.
        """
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label


class Resize(object):
    """Resize image and label to a fixed target size.

    Uses bilinear interpolation for the image and nearest-neighbour
    interpolation for the label (to preserve discrete class indices).
    This transform is used when the model requires a fixed input
    resolution regardless of the original image size.
    """

    def __init__(self, size):
        """Initialise Resize with the target spatial dimensions.

        Validates that ``size`` is a two-element iterable representing
        the desired ``(height, width)``.

        Args:
            size (tuple[int, int]): Target size as ``(h, w)``.
        """
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label):
        """Resize the image and label to the stored target size.

        The image is resized with bilinear interpolation and the label
        with nearest-neighbour interpolation to avoid creating
        non-existent class indices.

        Args:
            image (np.ndarray): Image of shape ``(H, W, C)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` resized to ``self.size``.
        """
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label


class RandScale(object):
    """Randomly resize image and label by a uniformly sampled scale factor.

    Samples a scale factor from the configured range and optionally
    perturbs the aspect ratio. Both the image and label are resized
    accordingly (bilinear for the image, nearest-neighbour for the
    label). This augmentation encourages the model to handle objects at
    varying scales.
    """

    def __init__(self, scale, aspect_ratio=None):
        """Initialise RandScale with scale range and optional aspect ratio range.

        Validates that ``scale`` is a two-element iterable ``[min, max]``
        with ``0 < min < max``. If ``aspect_ratio`` is provided it must
        follow the same constraints.

        Args:
            scale (list[float]): ``[scale_min, scale_max]`` defining the
                uniform sampling range for the scale factor.
            aspect_ratio (list[float] | None): ``[ar_min, ar_max]``
                defining the sampling range for the aspect-ratio
                perturbation. Default ``None`` (no aspect-ratio change).
        """
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        """Apply random scaling (and optional aspect ratio change) to image and label.

        Samples a scale factor uniformly from ``self.scale`` and, if
        configured, an aspect-ratio factor. The x and y scale factors
        are derived as ``scale * sqrt(ar)`` and ``scale / sqrt(ar)``
        respectively.

        Args:
            image (np.ndarray): Image of shape ``(H, W, C)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` resized by the sampled factors.
        """
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    '''Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    '''

    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        return image, label


class RandRotate(object):
    """Randomly rotate image and label by a uniformly sampled angle.

    Samples a rotation angle from the configured range and applies an
    affine rotation around the image centre. Bilinear interpolation is
    used for the image and nearest-neighbour for the label. Empty border
    regions are filled with the specified padding/ignore values.

    This augmentation helps the model generalise to rotated views.
    """

    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        """Initialise RandRotate with angle range, padding, and probability.

        Validates that the rotation range, padding values, and ignore
        label are well-formed.

        Args:
            rotate (list[float]): ``[angle_min, angle_max]`` in degrees.
                The rotation angle is sampled uniformly from this range.
            padding (list[float]): Three-element list of border fill
                values for the image (one per channel).
            ignore_label (int): Fill value for the label border region.
                Default ``255``.
            p (float): Probability of applying the rotation. Default
                ``0.5``.
        """
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        """Apply random rotation to the image and label.

        With probability ``self.p``, computes a 2D rotation matrix for
        a randomly sampled angle, then warps both image and label using
        ``cv2.warpAffine``. The image uses bilinear interpolation while
        the label uses nearest-neighbour.

        Args:
            image (np.ndarray): Image of shape ``(H, W, C)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` with rotation applied (or
            unchanged if the random gate is not triggered).
        """
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    """Randomly flip the image and label horizontally (left-right).

    Applies ``cv2.flip`` along the vertical axis (flip code 1) with the
    configured probability. This is one of the most common spatial
    augmentations for scene understanding, exploiting the left-right
    symmetry present in most real-world scenes.
    """

    def __init__(self, p=0.5):
        """Initialise RandomHorizontalFlip with the flip probability.

        Args:
            p (float): Probability of flipping. Default ``0.5``.
        """
        self.p = p

    def __call__(self, image, label):
        """Flip the image and label horizontally with probability ``self.p``.

        Uses ``cv2.flip(img, 1)`` to mirror around the vertical axis.
        Both image and label are flipped identically to maintain spatial
        correspondence.

        Args:
            image (np.ndarray): Image of shape ``(H, W, C)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` possibly flipped.
        """
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    """Randomly flip the image and label vertically (top-bottom).

    Applies ``cv2.flip`` along the horizontal axis (flip code 0) with
    the configured probability. Useful for aerial/satellite imagery or
    other domains where vertical orientation is arbitrary.
    """

    def __init__(self, p=0.5):
        """Initialise RandomVerticalFlip with the flip probability.

        Args:
            p (float): Probability of flipping. Default ``0.5``.
        """
        self.p = p

    def __call__(self, image, label):
        """Flip the image and label vertically with probability ``self.p``.

        Uses ``cv2.flip(img, 0)`` to mirror around the horizontal axis.
        Both image and label are flipped identically.

        Args:
            image (np.ndarray): Image of shape ``(H, W, C)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` possibly flipped.
        """
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    """Apply Gaussian blur to the image with 50 % probability.

    Convolves the image with a square Gaussian kernel of the specified
    radius, leaving the label unchanged. This augmentation simulates
    camera defocus and helps the model tolerate blurry inputs.
    """

    def __init__(self, radius=5):
        """Initialise RandomGaussianBlur with the kernel radius.

        The kernel size is ``(radius, radius)`` and standard deviation
        is computed automatically by OpenCV (sigma = 0).

        Args:
            radius (int): Side length of the square Gaussian kernel.
                Must be odd. Default ``5``.
        """
        self.radius = radius

    def __call__(self, image, label):
        """Optionally blur the image with a Gaussian kernel.

        With 50 % probability, applies ``cv2.GaussianBlur`` to the
        image. The label is always returned unchanged.

        Args:
            image (np.ndarray): Image of shape ``(H, W, C)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` with ``image`` possibly blurred.
        """
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR(object):
    """Convert an image from RGB channel order to BGR channel order.

    Some pretrained models (e.g. those converted from Caffe) expect BGR
    input. This transform swaps the first and third channels using
    ``cv2.cvtColor``. The label is passed through unchanged.
    """

    def __call__(self, image, label):
        """Convert the image from RGB to BGR.

        Uses ``cv2.cvtColor`` with ``cv2.COLOR_RGB2BGR`` to swap the
        red and blue channels.

        Args:
            image (np.ndarray): RGB image of shape ``(H, W, 3)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` with ``image`` in BGR order.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    """Convert an image from BGR channel order to RGB channel order.

    OpenCV loads images in BGR by default. This transform converts to
    the RGB order expected by most PyTorch models using
    ``cv2.cvtColor``. The label is passed through unchanged.
    """

    def __call__(self, image, label):
        """Convert the image from BGR to RGB.

        Uses ``cv2.cvtColor`` with ``cv2.COLOR_BGR2RGB`` to swap the
        blue and red channels.

        Args:
            image (np.ndarray): BGR image of shape ``(H, W, 3)``.
            label (np.ndarray): Label map of shape ``(H, W)``.

        Returns:
            tuple: ``(image, label)`` with ``image`` in RGB order.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label

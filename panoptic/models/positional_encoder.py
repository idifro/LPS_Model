"""3D sinusoidal positional encoding for point clouds.

Adapted from MaskPLS. Normalizes x,y by /50 and z by /4 (nuScenes ranges).
"""
import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Sinusoidal 3D positional encoding for point cloud coordinates.

    Generates fixed fourier-feature embeddings for each point so the decoder
    can reason about spatial locality.  Used in the panoptic decoder pipeline
    to inject positional information into mask features before cross-attention.

    Working:
        1. Input xyz coordinates are normalized to nuScenes ranges
           (x,y divided by 50, z divided by 4).
        2. Log-spaced frequency bands are generated from 1 to max_freq/2.
        3. Each coordinate is multiplied by each frequency, then sin and cos
           are concatenated to form the embedding.
        4. The result is zero-padded on the left to reach exactly feat_size.

    Args:
        feat_size (int): Output embedding dimensionality (default 256).
        max_freq (float): Maximum frequency for the log-spaced bands.
        base (int): Base of the logarithmic spacing.
        dimensionality (int): Number of spatial dimensions (default 3 for xyz).

    Returns:
        Tensor of shape (B, N, feat_size) with positional encodings.
    """

    def __init__(self, feat_size=256, max_freq=5000, base=2, dimensionality=3):
        """Initialize frequency bands and zero-padding for positional encoding.

        Sets up the number of frequency bands per spatial dimension and computes
        the amount of left zero-padding needed so the output width equals feat_size.
        Called once at model construction; the actual frequency tensors are built
        on-the-fly in forward() to match the input device.

        Args:
            feat_size (int): Desired output feature dimension. The product
                2 * num_bands * dimensionality is padded up to this value.
            max_freq (float): Upper bound of the log-spaced frequency range.
            base (int): Base for torch.logspace frequency generation.
            dimensionality (int): Number of coordinate axes (3 for xyz).
        """
        super().__init__()
        self.max_freq = max_freq
        self.dimensionality = dimensionality
        self.num_bands = math.floor(feat_size / dimensionality / 2)
        self.base = base
        pad = feat_size - self.num_bands * 2 * dimensionality
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, x):
        """
        x: (B, N, 3) point coordinates
        returns: (B, N, feat_size) positional encoding
        """
        x = x.clone()
        # Normalize for nuScenes ranges: x,y ∈ [-50,50], z ∈ [-5,3]
        x[:, :, 0] = x[:, :, 0] / 50.0
        x[:, :, 1] = x[:, :, 1] / 50.0
        x[:, :, 2] = x[:, :, 2] / 4.0
        x = x.unsqueeze(-1)
        scales = torch.logspace(
            0.0,
            math.log(self.max_freq / 2) / math.log(self.base),
            self.num_bands,
            base=self.base,
            device=x.device,
            dtype=x.dtype,
        )
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(2)
        return self.zero_pad(x)

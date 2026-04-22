"""3D sinusoidal positional encoding for point clouds.

Adapted from MaskPLS. Normalizes x,y by /50 and z by /4 (nuScenes ranges).
"""
import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, feat_size=256, max_freq=5000, base=2, dimensionality=3):
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

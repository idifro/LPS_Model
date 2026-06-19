'''3D model for distillation.'''

from collections import OrderedDict
from models.mink_unet import mink_unet as model3D
from torch import nn


def state_dict_remove_moudle(state_dict):
    """Strip the 'module.' prefix from every key in a PyTorch state dict.

    When a model is saved after DistributedDataParallel (DDP) training, each
    parameter key is automatically prefixed with 'module.'. This helper
    removes that prefix so the state dict can be loaded into the unwrapped
    (non-DDP) model without key-mismatch errors.

    Called during checkpoint loading before passing the state dict to
    ``model.load_state_dict()``, typically in evaluation or fine-tuning
    scripts that do not use DDP.

    Steps:
        1. Create a new empty OrderedDict.
        2. Iterate over every (key, value) pair in the incoming state dict.
        3. Replace the first occurrence of 'module.' in each key with ''.
        4. Insert the cleaned key and its tensor into the new dict.
        5. Return the cleaned OrderedDict.

    Args:
        state_dict (OrderedDict[str, torch.Tensor]): A PyTorch state dict
            whose keys may contain the 'module.' prefix added by DDP.

    Returns:
        OrderedDict[str, torch.Tensor]: A new state dict with identical
            values but with the 'module.' prefix removed from every key.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    """Factory function that instantiates a Minkowski-engine sparse 3D UNet.

    Provides a single entry point for building the 3D backbone used in the
    distillation pipeline. Wrapping the call in a factory keeps the rest of
    the codebase decoupled from the concrete ``mink_unet`` constructor and
    makes it straightforward to swap architectures in the future.

    Called by ``DisNet.__init__`` to build the 3D encoder with the desired
    input channels, output feature dimension, spatial dimension, and
    architecture variant.

    Steps:
        1. Forward all keyword arguments to ``mink_unet`` (imported as
           ``model3D``).
        2. ``mink_unet`` selects the appropriate architecture variant
           (e.g., MinkUNet34C) and returns the constructed model.
        3. Return the fully initialised model.

    Args:
        **kwargs: Arbitrary keyword arguments forwarded to ``mink_unet``.
            Expected keys include:
            - in_channels (int): Number of input feature channels (e.g., 3
              for RGB).
            - out_channels (int): Dimension of the output feature vector
              (512 for LSeg, 768 for OpenSeg).
            - D (int): Spatial dimensionality of the sparse tensor (3 for
              3-D point clouds).
            - arch (str): Architecture identifier string (e.g.,
              ``'MinkUNet34C'``).

    Returns:
        nn.Module: A Minkowski-engine sparse UNet model ready for training
            or inference.
    """
    model = model3D(**kwargs)
    return model


class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, cfg=None):
        """Initialise the distillation network with a Minkowski sparse UNet.

        Determines the output feature dimension based on the 2-D feature
        extractor specified in the config: 512 for LSeg or 768 for OpenSeg.
        This ensures the 3-D backbone produces embeddings in the same
        vector space as the 2-D teacher, which is essential for the
        feature-distillation loss.

        Called once when the training or evaluation script instantiates the
        model. The resulting ``DisNet`` wraps a single ``net3d`` module that
        can be used with MinkowskiEngine sparse tensors.

        Steps:
            1. If ``cfg.feature_2d_extractor`` is not set, default to
               ``'openseg'``.
            2. Set ``last_dim`` to 512 when the extractor is LSeg, or 768
               when it is OpenSeg; raise ``NotImplementedError`` otherwise.
            3. Build the 3-D sparse UNet via ``constructor3d`` with 3 input
               channels (RGB/coordinates), ``last_dim`` output channels,
               spatial dimension 3, and the architecture variant from
               ``cfg.arch_3d``.
            4. Store the network as ``self.net3d``.

        Args:
            cfg (object, optional): Configuration object with at least the
                following attributes:
                - feature_2d_extractor (str): Name of the 2-D feature
                  extractor (``'lseg'`` or ``'openseg'``).
                - arch_3d (str): Architecture identifier for the 3-D UNet
                  (e.g., ``'MinkUNet34C'``).
        """
        super(DisNet, self).__init__()
        if not hasattr(cfg, 'feature_2d_extractor'):
            cfg.feature_2d_extractor = 'openseg'
        if 'lseg' in cfg.feature_2d_extractor:
            last_dim = 512
        elif 'openseg' in cfg.feature_2d_extractor:
            last_dim = 768
        else:
            raise NotImplementedError

        # MinkowskiNet for 3D point clouds
        net3d = constructor3d(in_channels=3, out_channels=last_dim, D=3, arch=cfg.arch_3d)
        self.net3d = net3d

    def forward(self, sparse_3d):
        """Run a forward pass through the 3-D sparse UNet backbone.

        Takes a MinkowskiEngine SparseTensor that encodes a 3-D point cloud
        (coordinates + RGB features) and produces a dense feature matrix
        whose rows are per-point embeddings aligned with the 2-D teacher's
        feature space (512-d for LSeg, 768-d for OpenSeg).

        Called during both training (to compute the distillation loss
        against 2-D features) and inference (to obtain 3-D per-point
        embeddings for downstream tasks such as open-vocabulary
        segmentation).

        Steps:
            1. Pass ``sparse_3d`` into ``self.net3d`` (the Minkowski UNet).
            2. The UNet applies sparse 3-D convolutions, batch
               normalisation, and ReLU activations through its
               encoder-decoder structure.
            3. The final linear layer projects each point's features to the
               target embedding dimension (``out_channels``).
            4. Return the resulting SparseTensor (or dense feature tensor,
               depending on the UNet variant).

        Args:
            sparse_3d (MinkowskiEngine.SparseTensor): Sparse tensor with
                coordinates of shape ``(N, 4)`` (batch index + x, y, z) and
                features of shape ``(N, 3)`` (e.g., RGB or normalised
                coordinates), where *N* is the total number of active
                voxels across the batch.

        Returns:
            MinkowskiEngine.SparseTensor: Output sparse tensor with
                features of shape ``(N, C)`` where *C* is ``out_channels``
                (512 for LSeg, 768 for OpenSeg).
        """
        return self.net3d(sparse_3d)

    def forward_with_context(self, x, context_encoder, pos_enc=None):
        """Delegate to inner MinkUNet forward_with_context."""
        return self.net3d.forward_with_context(x, context_encoder, pos_enc)

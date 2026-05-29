# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
"""Minkowski U-Net architectures for sparse 3D point cloud feature extraction.

This module implements a family of U-Net encoder-decoder networks built on top of
MinkowskiEngine sparse convolutions. The networks process 3D point clouds represented
as sparse tensors and produce per-point feature vectors suitable for downstream tasks
such as semantic segmentation, open-vocabulary labelling, or panoptic segmentation.

The module provides a base class ``MinkUNetBase`` with a symmetric 4-stage encoder and
4-stage decoder connected by skip concatenations, plus numerous concrete variants that
differ in block type (BasicBlock vs. Bottleneck), layer counts, and channel widths.
A convenience factory function ``mink_unet`` instantiates any supported variant by name.

Typical usage in the OpenScene pipeline:
    model = mink_unet(in_channels=3, out_channels=256, D=3, arch='MinkUNet34C')
    features = model(sparse_tensor)  # returns (N, out_channels) dense feature matrix
"""

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.resnet_base import ResNetBase


class MinkUNetBase(ResNetBase):
    """Sparse U-Net encoder-decoder backbone using Minkowski convolutions.

    This class implements a symmetric U-Net architecture that operates on
    MinkowskiEngine SparseTensors. The encoder consists of 4 down-sampling
    stages (each with a stride-2 convolution followed by residual blocks),
    and the decoder mirrors them with 4 transposed-convolution up-sampling
    stages. Skip connections concatenate encoder features to decoder features
    at matching spatial resolutions, preserving fine-grained geometric detail.

    Subclasses must set the class attributes ``BLOCK`` and ``PLANES`` to
    select the residual block type (BasicBlock or Bottleneck) and the channel
    widths for each of the 8 stages (4 encoder + 4 decoder).

    Class Attributes:
        BLOCK: Residual block class (``BasicBlock`` or ``Bottleneck``).
        PLANES (Tuple[int, ...]): Channel widths for 8 stages
            (enc1-enc4, dec4-dec1).
        DILATIONS (Tuple[int, ...]): Dilation rates per stage.
        LAYERS (Tuple[int, ...]): Number of residual blocks per stage.
        INIT_DIM (int): Number of channels after the initial 5x5 convolution.
        OUT_TENSOR_STRIDE (int): Target output tensor stride (always 1).
    """

    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        """Initialise the Minkowski U-Net by delegating to ``ResNetBase``.

        ``ResNetBase.__init__`` calls ``self.network_initialization`` which
        builds all convolution, batch-norm, residual-block, and transposed-
        convolution layers, followed by weight initialisation. After this
        constructor returns the model is ready for a forward pass.

        Args:
            in_channels (int): Number of input feature channels per point
                (e.g. 3 for RGB, 6 for XYZRGB).
            out_channels (int): Number of output feature channels per point
                (e.g. 256 for CLIP-aligned features, 20 for class logits).
            D (int): Spatial dimensionality of the sparse tensor (default 3).
        """
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        """Build all encoder, decoder, and final projection layers.

        Called automatically by ``ResNetBase.__init__``. This method constructs
        the full U-Net graph:

        Encoder path (spatial resolution halves at each stage):
            1. Initial 5x5 conv (in_channels -> INIT_DIM) at stride 1.
            2. Stride-2 conv + block1 (PLANES[0]) at tensor stride 2.
            3. Stride-2 conv + block2 (PLANES[1]) at tensor stride 4.
            4. Stride-2 conv + block3 (PLANES[2]) at tensor stride 8.
            5. Stride-2 conv + block4 (PLANES[3]) at tensor stride 16 (bottleneck).

        Decoder path (spatial resolution doubles at each stage via transposed conv):
            6. Transposed conv -> cat(skip from block3) -> block5 (PLANES[4]), stride 8.
            7. Transposed conv -> cat(skip from block2) -> block6 (PLANES[5]), stride 4.
            8. Transposed conv -> cat(skip from block1) -> block7 (PLANES[6]), stride 2.
            9. Transposed conv -> cat(skip from conv0)  -> block8 (PLANES[7]), stride 1.

        Final: 1x1 convolution projecting PLANES[7] -> out_channels.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            D (int): Spatial dimensionality for all Minkowski convolutions.
        """
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7],
            out_channels,
            kernel_size=1,
            # has_bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        """Run the full encoder-decoder forward pass on a sparse tensor.

        The input SparseTensor is passed through the 4-stage encoder, producing
        intermediate feature maps at tensor strides 1, 2, 4, and 8 which are
        cached for skip connections. After the bottleneck at stride 16, the
        decoder progressively up-samples with transposed convolutions, concatenates
        the matching-resolution encoder features (skip connection), and refines
        with residual blocks. The final 1x1 convolution projects to
        ``out_channels`` and the dense feature matrix is returned.

        Args:
            x (ME.SparseTensor): Input sparse tensor with feature shape
                (N, in_channels), where N is the number of active voxels.

        Returns:
            torch.Tensor: Dense feature matrix of shape (N, out_channels)
                extracted from the final SparseTensor's ``.F`` attribute.
        """
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out) # bottleneck

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out).F

    def forward_with_context(self, x, context_encoder, pos_enc=None):
        """Forward pass that injects a context encoder at the stride-16 bottleneck.

        Runs the encoder exactly as forward(), then applies context_encoder
        on the bottleneck features (N_bn, 256) BEFORE the decoder starts.
        This is the correct insertion point: small N (~300-1500) and the 256-d
        bottleneck matches SceneContextEncoder(dim=256).

        Args:
            x: ME.SparseTensor input
            context_encoder: SceneContextEncoder(dim=256, ...) callable
            pos_enc: VoxelPositionalEncoding (optional), applied before attention
        Returns:
            (N, out_channels) dense feature tensor, same as forward()
        """
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16 (bottleneck)
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # ── Context encoder inserted HERE at bottleneck ──────────────────
        # out.F: (N_bn, 256) where N_bn ≈ 300-1500 at stride-16
        batch_idx = out.C[:, 0].long()
        bn_feats = out.F
        if pos_enc is not None:
            coords_xyz = out.C[:, 1:].float() * out.tensor_stride[0]
            bn_feats = pos_enc(bn_feats, coords_xyz)
        bn_feats = context_encoder(bn_feats, batch_idx)
        # Write updated features back into sparse tensor
        out = ME.SparseTensor(
            features=bn_feats,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
        )
        # ─────────────────────────────────────────────────────────────────

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out).F

class MinkUNet14(MinkUNetBase):
    """MinkUNet with 14-layer depth using BasicBlock (1 block per stage).

    Total depth is 14 layers because each of the 8 stages has exactly 1
    residual BasicBlock (2 conv layers each) minus shared stem/head layers.
    Subclasses must set PLANES to specify channel widths.
    """

    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    """MinkUNet with 18-layer depth using BasicBlock (2 blocks per stage).

    Each of the 8 stages contains 2 BasicBlocks, giving a moderate network
    depth suitable for real-time 3D segmentation tasks.
    Subclasses must set PLANES to specify channel widths.
    """

    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    """MinkUNet with 34-layer depth using BasicBlock with heavier encoder.

    The encoder stages use (2, 3, 4, 6) blocks for progressively deeper
    feature extraction, while decoder stages use 2 blocks each.
    Subclasses must set PLANES to specify channel widths.
    """

    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    """MinkUNet with 50-layer depth using Bottleneck blocks.

    Same layer distribution as MinkUNet34 but uses Bottleneck blocks
    (3 convolutions each with expansion factor 4) for higher capacity.
    Subclasses must set PLANES to specify channel widths.
    """

    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    """MinkUNet with 101-layer depth using Bottleneck blocks.

    The deepest encoder stage uses 23 Bottleneck blocks, providing very
    high capacity for complex 3D scene understanding at the cost of
    increased compute. Subclasses must set PLANES to specify channel widths.
    """

    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    """MinkUNet14 variant A with PLANES = (32, 64, 128, 256, 128, 128, 96, 96).

    A lightweight configuration with narrower decoder channels (96) suitable
    for memory-constrained settings or faster inference.
    """

    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    """MinkUNet14 variant B with PLANES = (32, 64, 128, 256, 128, 128, 128, 128).

    Slightly wider final decoder stages than variant A, offering a balance
    between model size and representational capacity.
    """

    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    """MinkUNet14 variant C with PLANES = (32, 64, 128, 256, 192, 192, 128, 128).

    Wider early decoder stages (192) for richer high-level feature fusion
    with skip connections from the encoder.
    """

    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    """MinkUNet14 variant D with PLANES = (32, 64, 128, 256, 384, 384, 384, 384).

    The widest MinkUNet14 variant with 384 channels throughout the decoder,
    maximising feature capacity at the expense of higher memory usage.
    """

    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    """MinkUNet18 variant A with PLANES = (32, 64, 128, 256, 128, 128, 96, 96).

    A compact 18-layer configuration with narrow final decoder channels (96),
    well-suited for lightweight feature distillation pipelines.
    """

    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    """MinkUNet18 variant B with PLANES = (32, 64, 128, 256, 128, 128, 128, 128).

    Uniform 128-channel decoder providing consistent feature dimensionality
    across all decoder stages of this 18-layer network.
    """

    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    """MinkUNet18 variant D with PLANES = (32, 64, 128, 256, 384, 384, 384, 384).

    The widest 18-layer variant with 384-channel decoder stages, offering
    high capacity for learning rich per-point feature representations.
    """

    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    """MinkUNet34 variant A with PLANES = (32, 64, 128, 256, 256, 128, 64, 64).

    A 34-layer configuration with a tapering decoder (256 -> 64) that
    progressively reduces channel width towards the output.
    """

    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    """MinkUNet34 variant B with PLANES = (32, 64, 128, 256, 256, 128, 64, 32).

    Similar to variant A but with an even narrower final stage (32 channels),
    producing the most compressed decoder features among MinkUNet34 variants.
    """

    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    """MinkUNet34 variant C with PLANES = (32, 64, 128, 256, 256, 128, 96, 96).

    The default architecture used in OpenScene for distilling 2D vision-language
    features into 3D point clouds, balancing depth and width effectively.
    """

    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


def mink_unet(in_channels=3, out_channels=20, D=3, arch='MinkUNet18A'):
    """Factory function to instantiate a Minkowski U-Net variant by name.

    This is the primary entry point used by training and inference scripts in
    the OpenScene pipeline. It maps an architecture name string to the
    corresponding concrete ``MinkUNetBase`` subclass, constructs it with the
    given channel and dimensionality parameters, and returns the ready-to-use
    ``nn.Module``.

    Args:
        in_channels (int): Number of input feature channels per point
            (default 3, e.g. RGB).
        out_channels (int): Number of output feature channels per point
            (default 20, e.g. number of semantic classes or embedding dim).
        D (int): Spatial dimensionality of the sparse tensor (default 3).
        arch (str): Architecture name string. Supported values are
            ``'MinkUNet14A'``, ``'MinkUNet14B'``, ``'MinkUNet14C'``,
            ``'MinkUNet14D'``, ``'MinkUNet18A'``, ``'MinkUNet18B'``,
            ``'MinkUNet18D'``, ``'MinkUNet34A'``, ``'MinkUNet34B'``,
            ``'MinkUNet34C'``.

    Returns:
        MinkUNetBase: An instance of the requested architecture subclass.

    Raises:
        Exception: If ``arch`` does not match any supported architecture name.
    """
    if arch == 'MinkUNet18A':
        return MinkUNet18A(in_channels, out_channels, D)
    elif arch == 'MinkUNet18B':
        return MinkUNet18B(in_channels, out_channels, D)
    elif arch == 'MinkUNet18D':
        return MinkUNet18D(in_channels, out_channels, D)
    elif arch == 'MinkUNet34A':
        return MinkUNet34A(in_channels, out_channels, D)
    elif arch == 'MinkUNet34B':
        return MinkUNet34B(in_channels, out_channels, D)
    elif arch == 'MinkUNet34C':
        return MinkUNet34C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14A':
        return MinkUNet14A(in_channels, out_channels, D)
    elif arch == 'MinkUNet14B':
        return MinkUNet14B(in_channels, out_channels, D)
    elif arch == 'MinkUNet14C':
        return MinkUNet14C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14D':
        return MinkUNet14D(in_channels, out_channels, D)
    else:
        raise Exception('architecture not supported yet'.format(arch))
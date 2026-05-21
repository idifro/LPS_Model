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

"""Minkowski Engine sparse ResNet base module for 3D point cloud classification.

This module provides the foundational ResNet architecture built on top of
MinkowskiEngine sparse convolutions, enabling efficient processing of 3D
point cloud data. It defines ``ResNetBase``, an abstract base class that
constructs the standard ResNet encoder pipeline (conv-bn-relu-pool followed
by four residual stages, a final convolution, global pooling, and a linear
head). Concrete depth variants (ResNet14 through ResNet101) are derived by
simply specifying BLOCK type and LAYERS tuple.

Usage context:
    This module is imported by higher-level architectures such as
    ``MinkUNetBase`` (in ``resnet.py``), which overrides
    ``network_initialization`` and ``forward`` to add a U-Net decoder while
    reusing ``_make_layer`` and ``weight_initialization`` from this base.

Typical pipeline position:
    Input sparse tensor -> ResNetBase encoder -> global feature vector ->
    classification head (or overridden by MinkUNet for dense prediction).
"""

import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class ResNetBase(nn.Module):
    """Abstract base class for Minkowski-Engine sparse ResNet encoders.

    This class provides the shared scaffolding for all sparse ResNet depth
    variants used in 3D point cloud understanding tasks. It is not
    instantiated directly; subclasses must set ``BLOCK`` and ``LAYERS``.

    Usage context:
        Serves as the parent for concrete variants (ResNet14 … ResNet101)
        and, more importantly, is the grandparent of ``MinkUNetBase`` which
        overrides ``network_initialization`` and ``forward`` to attach a
        U-Net style decoder while inheriting ``_make_layer`` and
        ``weight_initialization``.

    Design:
        1. ``__init__`` delegates to ``network_initialization`` to build
           layers and ``weight_initialization`` to set parameter values.
        2. ``_make_layer`` constructs a sequence of residual blocks,
           optionally adding a 1x1 downsample shortcut.
        3. ``forward`` runs the standard conv-bn-relu-pool -> 4 stages ->
           conv5 -> global pool -> linear pipeline.

    Attributes:
        BLOCK (type | None): Residual block class (``BasicBlock`` or
            ``Bottleneck``). Must be set by subclasses.
        LAYERS (tuple[int, ...]): Number of blocks per stage.
        INIT_DIM (int): Channel width after the first convolution.
        PLANES (tuple[int, ...]): Output channel widths for each of the
            four residual stages.
    """

    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        """Initialise the sparse ResNet encoder by building and initialising all layers.

        This constructor first calls the parent ``nn.Module.__init__``, then
        delegates to two helper methods: ``network_initialization`` (which
        creates every convolution, batch-norm, pooling, and linear layer) and
        ``weight_initialization`` (which applies Kaiming-normal and constant
        initialisers).

        Usage context:
            Called by every concrete ResNet variant (ResNet14 … ResNet101) and
            indirectly by ``MinkUNetBase`` (which overrides
            ``network_initialization`` but still relies on this constructor
            chain via ``super().__init__``).

        Steps:
            1. Invoke ``nn.Module.__init__`` for standard PyTorch bookkeeping.
            2. Store the spatial dimensionality ``D``.
            3. Assert that the subclass has set ``BLOCK``.
            4. Build the network graph via ``network_initialization``.
            5. Set parameter values via ``weight_initialization``.

        Args:
            in_channels (int): Number of input feature channels per point
                (e.g. 3 for RGB or 6 for XYZRGB).
            out_channels (int): Number of output classes / embedding dims
                produced by the final linear layer.
            D (int): Spatial dimensionality of the sparse tensor (default 3).
        """
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        """Build every layer of the sparse ResNet encoder graph.

        Constructs the full encoder pipeline: an initial 5x5 convolution with
        stride-2 down-sampling, batch normalisation, ReLU activation, and
        average pooling, followed by four residual stages (each created via
        ``_make_layer``), a 3x3 stride-3 convolution, another batch-norm +
        ReLU, a global max-pooling that collapses the sparse tensor to a
        single feature vector per batch element, and a final linear classifier.

        Usage context:
            Called once from ``__init__``. Subclass ``MinkUNetBase`` overrides
            this method entirely to add transposed-convolution decoder stages
            while reusing ``_make_layer`` for both encoder and decoder blocks.

        Steps:
            1. Set ``self.inplanes`` to ``INIT_DIM`` (64 by default).
            2. Create ``conv1`` (5x5, stride 2), ``bn1``, ``relu``, ``pool``
               (2x2 avg).
            3. Build ``layer1`` … ``layer4`` with channel widths from
               ``PLANES`` and depths from ``LAYERS``, each with stride 2.
            4. Add ``conv5`` (3x3, stride 3), ``bn5`` for further reduction.
            5. Add ``glob_avg`` (MinkowskiGlobalMaxPooling).
            6. Add ``final`` linear projection to ``out_channels``.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels (classes).
            D (int): Spatial dimensionality for all Minkowski operations.
        """

        self.inplanes = self.INIT_DIM
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, stride=2, dimension=D)

        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=D)

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)

        self.conv5 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)

        self.glob_avg = ME.MinkowskiGlobalMaxPooling(dimension=D)

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        """Apply Kaiming-normal and constant initialisers to all layers.

        Iterates over every sub-module and applies the appropriate
        initialisation scheme so that signal magnitudes are preserved
        through the deep residual network at the start of training.

        Usage context:
            Called once at the end of ``__init__``, after
            ``network_initialization`` has created all layers.  Inherited
            unchanged by ``MinkUNetBase`` and all depth variants.

        Steps:
            1. For each ``MinkowskiConvolution``: apply Kaiming-normal
               (fan-out, ReLU) to the sparse kernel tensor.
            2. For each ``MinkowskiBatchNorm``: set weight (gamma) to 1 and
               bias (beta) to 0 so the layer initially acts as identity.
        """
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    bn_momentum=0.1):
        """Create a sequential stage of residual blocks with optional downsampling.

        Builds one residual stage consisting of ``blocks`` consecutive residual
        blocks. The first block may include a stride > 1 and a learned 1x1
        convolution shortcut (downsample) to match spatial resolution and
        channel dimensions; subsequent blocks use identity shortcuts.

        Usage context:
            Called by ``network_initialization`` (and its overrides in
            ``MinkUNetBase``) to construct each of the four encoder stages
            and, in the U-Net variant, corresponding decoder stages.

        Steps:
            1. If ``stride != 1`` or the current ``inplanes`` differs from
               ``planes * block.expansion``, create a downsample shortcut
               consisting of a 1x1 ``MinkowskiConvolution`` followed by
               ``MinkowskiBatchNorm``.
            2. Append the first block with the given stride, dilation, and
               optional downsample.
            3. Update ``self.inplanes`` to ``planes * block.expansion``.
            4. Append the remaining ``blocks - 1`` blocks with stride 1.
            5. Wrap all blocks in ``nn.Sequential`` and return.

        Args:
            block (type): Residual block class (``BasicBlock`` or
                ``Bottleneck`` from ``MinkowskiEngine``).
            planes (int): Base channel width for this stage.
            blocks (int): Number of residual blocks in this stage.
            stride (int): Spatial stride for the first block (default 1).
            dilation (int): Dilation rate for all blocks (default 1).
            bn_momentum (float): Batch-norm momentum (default 0.1, currently
                unused but kept for API compatibility).

        Returns:
            nn.Sequential: A sequential container of ``blocks`` residual
                blocks, each producing sparse tensors with
                ``planes * block.expansion`` feature channels.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    dimension=self.D))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Run the full sparse ResNet encoder forward pass.

        Passes the input sparse tensor through the initial convolution block,
        four residual stages, a final convolution block, global max-pooling,
        and a linear head to produce per-batch-element predictions.

        Usage context:
            Called during training and inference for pure classification
            tasks. ``MinkUNetBase`` overrides this method to add decoder
            stages and skip connections for dense per-point prediction.

        Steps:
            1. ``conv1`` -> ``bn1`` -> ``relu`` -> ``pool``: initial
               feature extraction and spatial down-sampling.
            2. ``layer1`` … ``layer4``: four residual stages that
               progressively increase channels and decrease resolution.
            3. ``conv5`` -> ``bn5`` -> ``relu``: additional spatial
               compression before pooling.
            4. ``glob_avg``: collapse sparse tensor to one feature vector
               per batch element (shape ``[B, C]``).
            5. ``final``: linear projection to ``out_channels``.

        Args:
            x (ME.SparseTensor): Input sparse tensor with feature shape
                ``[N, in_channels]`` where ``N`` is the total number of
                active sites across the batch.

        Returns:
            ME.SparseTensor: Output sparse tensor with feature shape
                ``[B, out_channels]`` after global pooling, where ``B``
                is the batch size.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.glob_avg(x)
        return self.final(x)


class ResNet14(ResNetBase):
    """Sparse ResNet-14 encoder using BasicBlock with 1 block per stage.

    This is the shallowest variant, suitable for quick experiments or
    small point-cloud datasets. Total depth: 1+1+1+1 = 4 residual
    blocks (14 weighted layers counting conv1, conv5, and the linear head).
    """

    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    """Sparse ResNet-18 encoder using BasicBlock with 2 blocks per stage.

    A lightweight baseline commonly used for 3D scene understanding
    benchmarks. Total depth: 2+2+2+2 = 8 residual blocks (18 weighted
    layers including conv1, conv5, and the linear head).
    """

    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    """Sparse ResNet-34 encoder using BasicBlock with (3,4,6,3) blocks.

    A medium-capacity model offering a good trade-off between accuracy
    and compute for 3D sparse tasks. Total depth: 3+4+6+3 = 16 residual
    blocks (34 weighted layers).
    """

    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    """Sparse ResNet-50 encoder using Bottleneck with (3,4,6,3) blocks.

    Switches from BasicBlock to Bottleneck (expansion=4), substantially
    increasing capacity while keeping computational cost manageable via
    1x1 projection layers. Total depth: 3+4+6+3 = 16 bottleneck blocks
    (50 weighted layers).
    """

    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    """Sparse ResNet-101 encoder using Bottleneck with (3,4,23,3) blocks.

    The deepest variant provided, with a significantly enlarged third
    stage (23 bottleneck blocks). Intended for large-scale 3D datasets
    where higher capacity improves feature quality. Total depth:
    3+4+23+3 = 33 bottleneck blocks (101 weighted layers).
    """

    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
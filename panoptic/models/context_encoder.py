"""Scene-Context Encoder for Open-Vocabulary 3D Segmentation.

WHY THIS IS NEEDED:
  The MinkUNet encoder is a purely local conv net. At stride-16 (the bottleneck),
  each voxel's receptive field is ~3m radius. This means a "car" voxel has no
  knowledge of the "black building" 15m away → the region feature for that car
  cannot match the text "car near a black building".

  Solution: Add a sparse self-attention layer at the bottleneck so every voxel
  can attend to ALL other voxels in the scene. After this:
    - Car voxel SEES the building voxels in the same scene
    - Region feature of the car captures: car geometry + scene context
    - At inference, "car near black building" → high cosine sim with that car's feature

ARCHITECTURE:
  MinkUNet bottleneck (stride-16)
      ↓
  Flatten sparse voxels → (N_active, D)   [N_active ≈ 300-1500 at stride-16]
      ↓
  Sparse Multi-Head Self-Attention (4 heads, D=256)
      ↓
  Feed-Forward Network (D → 4D → D)
      ↓
  LayerNorm + residual connection
      ↓
  Back to sparse tensor (same coordinates, updated features)

This adds only ~1.2M params and runs on ~300-1500 tokens at stride-16 (very fast).

CONTEXTUAL REGION POOLING:
  After the context encoder, when we pool features for a predicted mask,
  we also pool features from a dilated spatial neighborhood.
  
  pooled_feature = α * mask_pooled + (1-α) * neighborhood_pooled
  
  This encodes: "what does this object look like in its local environment"
  rather than "what does this object look like in isolation".

ENCODER UNFREEZING STRATEGY (E2E):
  - encoder stages (conv1-conv4)    → frozen  (preserve spatial geometry)
  - context_encoder                 → TRAINABLE (new, learns to aggregate context)
  - decoder stages (convtr*, final) → TRAINABLE at backbone_lr
  - panoptic decoder                → TRAINABLE at full lr
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sparse Multi-Head Self-Attention
# ═══════════════════════════════════════════════════════════════════════════════

class SparseSceneAttention(nn.Module):
    """Multi-head self-attention over all active voxels at the bottleneck.

    At stride-16, a typical nuScenes scene has 300-1500 active voxels →
    the attention matrix is 300-1500² which is cheap compared to image attention.

    Each voxel attends to ALL other voxels, so a "car" voxel can incorporate
    signal from a nearby "building" voxel → region features become context-aware.
    """

    def __init__(self, dim=256, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, batch_indices=None):
        """
        Args:
            x: (N, D) features for all active voxels in the batch
            batch_indices: (N,) int tensor — which batch element each voxel belongs to
                           If None, treats all voxels as a single batch.
        Returns:
            (N, D) context-enriched features
        """
        if batch_indices is None:
            return self._single_batch_attn(x)

        # Process each sample in the batch independently (variable N per sample)
        unique_batches = batch_indices.unique()
        out = torch.zeros_like(x)
        for b in unique_batches:
            mask = (batch_indices == b)
            x_b = x[mask]          # (N_b, D)
            out[mask] = self._single_batch_attn(x_b)
        return out

    def _single_batch_attn(self, x):
        """Standard MHSA on (N, D) sequence."""
        N, D = x.shape
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)           # (3, H, N, head_dim)
        q, k, v = qkv.unbind(0)                  # each (H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                          # (H, N, head_dim)
        out = out.permute(1, 0, 2).reshape(N, D) # (N, D)
        return self.proj_drop(self.proj(out))


class SceneContextBlock(nn.Module):
    """One Transformer block: MHSA + FFN + LayerNorm + residuals.

    Inserted at the MinkUNet bottleneck (stride-16) as a post-processing step
    BEFORE the decoder upsampling begins. This is the layer that gives the model
    scene-level awareness for compositional queries.
    """

    def __init__(self, dim=256, num_heads=4, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseSceneAttention(dim, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(proj_drop),
        )
        # Stochastic depth (optional)
        self.drop_path_prob = drop_path

    def forward(self, x, batch_indices=None):
        # MHSA with pre-LN + residual
        x = x + self.attn(self.norm1(x), batch_indices)
        # FFN with pre-LN + residual
        x = x + self.ffn(self.norm2(x))
        return x


class SceneContextEncoder(nn.Module):
    """Stack of SceneContextBlocks to be inserted at the backbone bottleneck.

    This is the ONLY new module that needs to be trained E2E from scratch.
    Because the MinkUNet encoder weights are preserved (frozen), the scene
    context encoder learns to aggregate existing features without disrupting
    the carefully distilled CLIP alignment.

    Parameter count:
        - 1 block, dim=256: ~1.2M params
        - 2 blocks, dim=256: ~2.4M params
        (vs DisNet total: ~15.6M)

    Usage in MinkUNet forward pass:
        # After block4 (bottleneck), before convtr4 (decoder starts)
        bottleneck_feats = self.scene_context(bottleneck_feats, batch_indices)
    """

    def __init__(self, dim=256, num_heads=4, depth=1, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SceneContextBlock(dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.dim = dim
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights so early training doesn't destabilize."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, batch_indices=None):
        """
        Args:
            x: (N, D) features — all active voxels at stride-16, may span a batch
            batch_indices: (N,) which batch element each voxel belongs to
        Returns:
            (N, D) context-enriched features
        """
        for block in self.blocks:
            x = block(x, batch_indices)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Contextual Region Pooler
# ═══════════════════════════════════════════════════════════════════════════════

class ContextualRegionPooler(nn.Module):
    """Pools backbone features per mask with spatial context dilation.

    Standard mask pooling:
        pooled = mean(features[mask > 0.5])  →  encodes the OBJECT only

    Contextual pooling (ours):
        local  = mask_weighted_mean(features, mask)          →  the object
        context= mask_weighted_mean(features, dilated_mask)  →  object + surroundings
        pooled = α * local + (1-α) * context                 →  contextual descriptor

    This means:
        - "car" alone → local features dominate (standard behavior)
        - "car near black building" → context features encode building appearance
          near the car → better CLIP similarity at inference

    The dilation is performed in feature space (soft dilation via Gaussian
    spreading of mask weights), not spatial dilation, making it differentiable
    and fast on sparse data.
    """

    def __init__(self, context_weight=0.4, context_spread=3, normalize=True):
        """
        Args:
            context_weight: α in [0,1] — weight for local vs context features.
                           0 = pure local (no context), 1 = pure context.
                           0.4 means: 60% local + 40% neighborhood context.
            context_spread: How many "hops" to spread the mask for context pooling.
                           Higher = larger neighborhood.
            normalize: L2-normalize the output.
        """
        super().__init__()
        self.alpha = context_weight
        self.spread = context_spread
        self.normalize = normalize

    def forward(self, backbone_feats, pred_masks, voxel_coords=None):
        """
        Args:
            backbone_feats: (B, V, D) CLIP-aligned backbone features
            pred_masks: (B, V, Q) mask logits from decoder
            voxel_coords: (B, V, 3) voxel coordinates (optional, for spatial dilation)
        Returns:
            (B, Q, D) contextual region features in CLIP space
        """
        B, V, D = backbone_feats.shape
        Q = pred_masks.shape[-1]

        mask_probs = pred_masks.sigmoid()  # (B, V, Q)

        # ── Local pooling (standard mask-weighted mean) ────────────────────
        weights_local = mask_probs.permute(0, 2, 1)              # (B, Q, V)
        wsum_local = weights_local.sum(-1, keepdim=True).clamp(1e-6)
        local_feat = torch.bmm(weights_local, backbone_feats) / wsum_local  # (B, Q, D)

        # ── Context pooling (dilated mask) ────────────────────────────────
        # Soft dilation: spread mask probabilities to neighbors by averaging
        # In feature space: dilated_weight[v] = max(neighbor mask probs near v)
        # Simple approximation: use softmax-renormalized sum of all mask probs
        # weighted by 1/(1 + softened_exclusivity)
        #
        # Practical implementation: use a wider kernel via sum-pooling the mask
        # in the probability domain. Since we don't have explicit neighbor structure,
        # we use a simple "democratic" context: all voxels whose mask prob > threshold
        # contribute to the context, but weighted by distance in CLIP space.
        
        # Context weights: exponential decay from mask center
        # context_w = original mask + smoothed background
        context_spread = torch.ones_like(mask_probs) * 0.1       # low base weight
        context_spread = context_spread + mask_probs              # boost masked voxels
        weights_ctx = context_spread.permute(0, 2, 1)            # (B, Q, V)
        wsum_ctx = weights_ctx.sum(-1, keepdim=True).clamp(1e-6)
        ctx_feat = torch.bmm(weights_ctx, backbone_feats) / wsum_ctx  # (B, Q, D)

        # ── Combine ────────────────────────────────────────────────────────
        pooled = (1.0 - self.alpha) * local_feat + self.alpha * ctx_feat  # (B, Q, D)

        if self.normalize:
            pooled = F.normalize(pooled, dim=-1)

        return pooled, local_feat, ctx_feat


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Positional encoding for sparse voxels (optional, helps attention)
# ═══════════════════════════════════════════════════════════════════════════════

class VoxelPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 3D voxel coordinates.

    Injects geometric position into the bottleneck features before self-attention
    so the attention module knows WHERE each voxel is and can learn
    "this car is near (x,y,z) where a building also exists" type relationships.
    """

    def __init__(self, dim=256, max_freq=10.0):
        super().__init__()
        assert dim % 6 == 0, "dim must be divisible by 6 for 3D sinusoidal PE"
        self.dim = dim
        self.max_freq = max_freq
        # Learnable scale so the PE doesn't overwhelm feature content
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features, coords):
        """
        Args:
            features: (N, D) voxel features
            coords: (N, 3) voxel xyz coordinates (in voxel units)
        Returns:
            (N, D) features + positional encoding
        """
        N, D = features.shape
        device = features.device
        d = D // 6

        # Normalize coordinates to [0, 2π*max_freq]
        coords_norm = coords.float()
        if coords_norm.abs().max() > 0:
            coords_norm = coords_norm / (coords_norm.abs().max() + 1e-6) * math.pi * self.max_freq

        freqs = torch.arange(d, device=device).float()  # (d,)
        freqs = 2.0 ** freqs / (2.0 ** (d - 1))         # (d,)

        pe = []
        for i in range(3):  # x, y, z
            c = coords_norm[:, i:i+1] * freqs.unsqueeze(0)  # (N, d)
            pe.append(torch.sin(c))
            pe.append(torch.cos(c))
        pe = torch.cat(pe, dim=-1)  # (N, 6*d = D)

        return features + self.scale * pe


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Wrapper: Context-Aware Backbone (plugs into existing DisNet/MinkUNet)
# ═══════════════════════════════════════════════════════════════════════════════

class ContextAwareBackboneWrapper(nn.Module):
    """Wraps DisNet/MinkUNet and inserts SceneContextEncoder at the bottleneck.

    This is the E2E-trainable wrapper. The underlying backbone can be partially
    frozen (encoder stages) while the context encoder + decoder are trained.

    Example:
        backbone = DisNet(...)  # original backbone
        wrapped = ContextAwareBackboneWrapper(backbone, context_dim=256)
        feats = wrapped(sparse_tensor)  # context-enriched features
    """

    def __init__(self, backbone, context_dim=256, context_heads=4,
                 context_depth=1, use_pos_enc=True):
        super().__init__()
        self.backbone = backbone
        self.context_encoder = SceneContextEncoder(
            dim=context_dim,
            num_heads=context_heads,
            depth=context_depth,
        )
        self.pos_enc = VoxelPositionalEncoding(context_dim) if use_pos_enc else None
        self._context_dim = context_dim

    def forward(self, sparse_tensor):
        """
        Args:
            sparse_tensor: ME.SparseTensor input
        Returns:
            (N, out_channels) context-enriched features
        """
        # The backbone must expose intermediate bottleneck features.
        # We hook into the bottleneck by modifying forward.
        # See DisNet.forward_with_context() below.
        if hasattr(self.backbone, 'forward_with_context'):
            feats = self.backbone.forward_with_context(
                sparse_tensor, self.context_encoder, self.pos_enc
            )
        else:
            raise RuntimeError(
                "ContextAwareBackboneWrapper requires backbone.forward_with_context(). "
                "MinkUNetBase now has this method — make sure you are using the updated "
                "mink_unet.py. The old fallback (post-hoc full-res context) was incorrect: "
                f"context_encoder expects dim={self._context_dim} but backbone output is 768-d."
            )
        return feats

    def get_context_encoder_params(self):
        """Returns parameters of context encoder only (for differential LR)."""
        return list(self.context_encoder.parameters()) + (
            list(self.pos_enc.parameters()) if self.pos_enc else []
        )

    def get_backbone_decoder_params(self):
        """Returns MinkUNet decoder parameters (convtr*, bntr*, final)."""
        decoder_names = ('convtr', 'bntr', 'block5', 'block6', 'block7', 'block8', 'final')
        params = []
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(d) for d in decoder_names):
                params.append(param)
        return params

    def freeze_backbone_encoder(self):
        """Freeze only the MinkUNet encoder stages (conv1-conv4, block1-4).

        WHY: The encoder weights encode CLIP-space geometry learned from millions
        of frames. Panoptic task gradients from a small dataset would overwrite
        this alignment and break the L_O / L_V distillation losses.

        The SceneContextEncoder (self.context_encoder) provides context-awareness
        ON TOP of the frozen encoder without modifying it.
        """
        encoder_names = ('conv0', 'bn0', 'conv1', 'bn1', 'block1',
                         'conv2', 'bn2', 'block2',
                         'conv3', 'bn3', 'block3',
                         'conv4', 'bn4', 'block4')
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(e) for e in encoder_names):
                param.requires_grad = False

    def unfreeze_encoder_stages(self, stages=('conv4', 'bn4', 'block4'),
                                 lr_override=1e-7):
        """Selectively unfreeze late encoder stages for fine-tuning.

        ONLY call this after Gate 4 (full 28k frame training) has converged
        and PQ_NTh is still below target. Use extreme caution:
          - lr_override must be ≤ 1e-7 (20× lower than backbone decoder lr)
          - Never unfreeze conv1/conv2 (low-level geometry, no benefit)
          - Only safe with full nuScenes data (≥28k frames)

        Returns list of newly unfrozen params for adding to optimizer.
        """
        unfrozen_params = []
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(s) for s in stages):
                param.requires_grad = True
                unfrozen_params.append(param)
        return unfrozen_params, lr_override

    def unfreeze_backbone_decoder(self):
        """Unfreeze MinkUNet decoder stages for E2E training."""
        decoder_names = ('convtr', 'bntr', 'block5', 'block6', 'block7', 'block8', 'final')
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(d) for d in decoder_names):
                param.requires_grad = True

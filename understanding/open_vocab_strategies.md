# Making the Panoptic Decoder Truly Open-Vocabulary

## Root Cause: Why Current Token Head Failed

The current contrastive (InfoNCE) loss with **only 16 fixed text prototypes** is mathematically equivalent to a 16-class softmax classifier:

```
L = -log( exp(sim(pred, proto_gt) / τ) / Σ_{c=1}^{16} exp(sim(pred, proto_c) / τ) )
```

With only 16 fixed denominators, the optimal solution is to map every query to the nearest of 16 centroids. The network learns a **16-to-16 bijection**, not generalizable CLIP alignment. Novel class prototypes were never seen as negatives during training, so the token head has no reason to produce embeddings near them.

### The Fundamental Problem Visualized

Imagine CLIP embedding space as a 768-dimensional sphere. The 16 class prototypes are 16 fixed points on this sphere. Training says: "map query for car → car prototype, query for truck → truck prototype." The token head learns the **shortest path** from decoder space to those 16 points — it creates 16 tunnels, not a general map of the sphere.

At inference, "pole" or "bench" prototypes are in regions the token head has **never been trained to reach**. Its outputs live on a 16-point manifold, not the full CLIP sphere.

---

## Strategy A: Mask-Pooled Frozen Features (No Learned Classifier)

### What It Is

Completely remove any learned classification pathway. Instead, use the frozen DisNet backbone features directly — pool them through predicted masks to get per-object features, then classify by cosine similarity to any text embedding.

### Why It Is Useful

The frozen DisNet was distilled from CLIP ViT-L/14@336px via OpenSeg. This means every voxel already carries a 768-dim feature that should be near the CLIP text embedding of whatever object that voxel belongs to. If you pool (average) these features within a predicted mask region, the result should be a "what is this object?" descriptor in CLIP space.

This is the **theoretically purest** approach because:
1. No learned classifier means no closed-set collapse is possible
2. The backbone features are the best CLIP-aligned signal you have in 3D
3. Classification becomes a pure retrieval problem: nearest text prototype wins
4. Works for ANY text prompt — truly unlimited vocabulary

### How It Works Step by Step

```
TRAINING PHASE (class-agnostic mask learning):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input: LiDAR point cloud → voxelize → (V, 3) coords + (V, 3) feats
    │
    ▼
  Frozen DisNet backbone → (V, 768) frozen features
    │
    ├─── backbone_proj (768→256) ──→ decoder source features (cross-attn K,V)
    │
    └─── mask_feat_proj (768→256) ──→ mask prediction features

  Learnable queries (Q=106) ──→ Transformer Decoder (9 layers)
    │
    ├─── mask_embed MLP ──→ pred_masks (V, Q)     ← trained with BCE + Dice
    └─── objectness_head ──→ obj_scores (Q, 1)    ← trained with BCE (is this a real object?)

  NO class_embed head. NO CE loss. NO token head.

  Matching: Hungarian matching using ONLY mask cost (BCE + Dice).
  Loss = w_bce * L_bce + w_dice * L_dice + w_obj * L_objectness

INFERENCE PHASE (open-vocabulary classification):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Same forward pass → pred_masks (V, Q)
    │
    ▼
  Pool frozen backbone features through masks:
    backbone_feats = F.normalize(frozen_disnet(points), dim=-1)   # (V, 768)
    mask_weights = pred_masks.sigmoid()                           # (V, Q)
    mask_weights = mask_weights / mask_weights.sum(0).clamp(1)   # normalize per query
    pooled = mask_weights.T @ backbone_feats                      # (Q, 768)
    pooled = F.normalize(pooled, dim=-1)                          # unit sphere
    │
    ▼
  Classify with ANY text:
    text_feats = CLIP_text_encoder(["car", "pedestrian", ..., "bench", "pole"])
    logits = pooled @ text_feats.T                                # (Q, C)
    scores = logits.max(dim=-1)                                   # confidence
    labels = logits.argmax(dim=-1)                                # class
    │
    ▼
  Panoptic assignment:
    Sort queries by objectness * max_score (descending)
    Greedy non-overlapping assignment (same as current)
```

### Concrete Code Changes

**In `panoptic_loss.py`:**
```python
# Remove classification loss entirely
# Keep only mask losses
loss_dict = {}
loss_dict["loss_mask"] = sigmoid_ce_loss(pred_masks, gt_masks)
loss_dict["loss_dice"] = dice_loss(pred_masks, gt_masks)
loss_dict["loss_obj"] = F.binary_cross_entropy_with_logits(obj_scores, gt_obj)
# NO loss_dict["loss_ce"]
```

**In `matcher.py`:**
```python
# Remove class cost from matching
# cost = cost_mask * w_mask + cost_dice * w_dice
# NO cost_class term — masks matched purely by spatial overlap
```

**In `mask_decoder.py`:**
```python
# Replace class_embed with objectness head
self.objectness_head = nn.Linear(hidden_dim, 1)  # binary: real object or not
# Remove: self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
```

**New inference function:**
```python
def panoptic_inference_openvocab(pred_masks, backbone_feats, text_protos, obj_scores, overlap_thr=0.8):
    # 1. Pool features
    backbone_feats = F.normalize(backbone_feats, dim=-1)
    mask_probs = pred_masks.sigmoid()
    weights = mask_probs / mask_probs.sum(0).clamp(min=1)
    pooled = F.normalize(weights.T @ backbone_feats, dim=-1)

    # 2. Classify
    sim = pooled @ text_protos.T  # (Q, C)
    scores, labels = sim.max(dim=-1)

    # 3. Combine with objectness
    obj = obj_scores.sigmoid().squeeze(-1)
    combined_scores = scores * obj

    # 4. Greedy assignment (same logic)
    ...
```

### Why This Could Fail

1. **Backbone-pool already failed at PQ=0.209** — but this was due to bugs (no pre-normalization, hard thresholds, no objectness filtering). The fixed recipe above should be dramatically better.

2. **Point-level distillation ≠ object-level features** — DisNet was trained to match per-pixel OpenSeg features. **CONFIRMED (2026-05-27)**: Averaging many voxels DOES "blur" the semantic signal. Prototype bias calibration showed that random pools give ±0.08 cosine similarity (near zero) because different-class vectors cancel. **Mitigation: use per-voxel logit aggregation (Strategy K/L) instead of feature pooling.**

3. **Without CE loss, masks may degrade** — the CE loss provides a class-discriminative signal that helps queries specialize. Without it, multiple queries might compete for the same object, or queries might produce fragmented masks.

### How to Fix Those Failures

- **For blur**: use ATTENTION-weighted pooling (use mask logits before sigmoid as attention), not uniform average. High-logit voxels contribute more.
- **For mask degradation**: keep stuff queries with FIXED class assignment (not open-vocab). Only thing queries are class-agnostic. This preserves structure.
- **For query competition**: use auxiliary losses on intermediate decoder layers (already implemented).

---

## Strategy B: Class-Agnostic Training + Diverse Text Supervision

### What It Is

A hybrid of Strategy A and Strategy C. Train the mask decoder class-agnostically (no CE loss), but add a lightweight **projection adapter** on top of mask-pooled features that is trained with a LARGE and DIVERSE set of text embeddings as negatives. This adapter learns to "clean up" the pooled features without collapsing to a small set.

### Why It Is Useful

Strategy A's weakness is that raw backbone features may be noisy when pooled. Strategy C's weakness is that 16 fixed prototypes cause collapse. Strategy B takes the best of both:

- **Class-agnostic masks** → no closed-set bias in mask learning
- **Diverse text negatives** → the adapter preserves full CLIP geometry because it must distinguish among HUNDREDS of concepts, not just 16
- **Lightweight adapter** → fixes scale/normalization issues in pooled features

The key insight from OpenSeg's success: when you train with diverse vocabulary (thousands of words from captions), the resulting embedding space generalizes. When you train with 16 words, it doesn't.

### How It Works Step by Step

**Precomputation (one-time):**
```python
# Generate a text embedding bank of 1000+ concepts
concepts = [
    # 16 known nuScenes classes
    "barrier", "bicycle", "bus", "car", ...
    # 200+ driving-related concepts
    "pole", "lamp post", "fire hydrant", "bench", "tree trunk", "curb",
    "pothole", "speed bump", "stop sign", "crosswalk", "parking meter",
    "mailbox", "garbage bin", "fence post", "guard rail", "bollard", ...
    # 200+ general concepts (for regularization)
    "dog", "cat", "bird", "cloud", "puddle", "shadow", "gravel",
    "asphalt", "concrete", "metal", "glass", "plastic", "wood", ...
    # 200+ adversarial concepts (things that LOOK like nuScenes objects)
    "van", "SUV", "minivan", "pickup truck",  # similar to car/truck
    "scooter", "moped", "segway",             # similar to motorcycle
    "child", "jogger", "skateboarder",         # similar to pedestrian
    ...
]
text_bank = CLIP_text_encoder(["a " + c + " in a scene" for c in concepts])
text_bank = F.normalize(text_bank, dim=-1)  # (1000+, 768)
torch.save(text_bank, "text_bank_1000.pt")
```

**Training architecture:**
```
Frozen DisNet → (V, 768)
    │
    ├── backbone_proj → Decoder → pred_masks (class-agnostic, BCE+Dice only)
    │
    └── raw features (V, 768)
            │
      ┌─────┴─────┐
      │  Mask Pool │  (soft weighted average through pred_masks)
      └─────┬─────┘
            │
      (Q, 768) pooled features
            │
      ┌─────┴─────┐
      │  Adapter   │  Linear(768,768) + LayerNorm + L2Norm
      └─────┬─────┘
            │
      (Q, 768) adapted features
            │
      ┌─────┴───────────────────────────────┐
      │  Contrastive loss vs 1000 concepts: │
      │  L = -log(exp(sim_gt/τ) / Σexp(sim_all/τ))
      │  with dynamic negative sampling     │
      └─────────────────────────────────────┘
```

**Key difference from current token head:**
- Current: MLP takes decoder output (256-d) → maps to CLIP space → 16 negatives
- Strategy B: Simple projection takes POOLED BACKBONE (768-d) → cleans up → 1000 negatives

The adapter starts from features that are ALREADY in CLIP space — it just fixes noise. It doesn't need to learn the CLIP space from scratch (which is what the token head had to do from 256-d decoder features).

### Concrete Loss Function
```python
def diverse_contrastive_loss(adapted_feats, gt_classes, text_bank, temperature=0.07):
    """
    adapted_feats: (Q_matched, 768) — adapted pooled features for matched queries
    gt_classes: (Q_matched,) — GT class indices (0-15)
    text_bank: (1000, 768) — full text embedding bank (includes 16 GT + 984 others)
    """
    # Cosine similarities to all concepts
    sim = adapted_feats @ text_bank.T / temperature  # (Q, 1000)

    # For each query, the positive is text_bank[gt_class_idx]
    # The negatives are ALL 1000 concepts
    labels = gt_classes  # index into text_bank (first 16 entries are GT classes)
    loss = F.cross_entropy(sim, labels)

    return loss
```

### Why This Could Fail

1. **Engineering overhead** — curating 1000 concepts requires thought
2. **Adapter might overfit** — if training set is small, adapter sees same pooled features repeatedly
3. **Masks still need to be good** — class-agnostic training may produce worse masks, making the pooled features noisier

### Mitigation

- Start with 200 negatives (not 1000) and scale up
- Use dropout on the adapter (0.1)
- Freeze adapter periodically and evaluate open-vocab
- Keep mask quality: use aux losses, gradient accumulation, cosine warmup

---

## Strategy C: Preserve Token Head + Fix the Loss

### What It Is

Keep the existing architecture (decoder → token head MLP → 768-d CLIP space) but fundamentally change the loss function to prevent closed-set collapse. Three sub-options.

### Why It Is Useful

The token head architecture is fine — a 3-layer MLP from 256-d to 768-d with L2 normalization is a reasonable design. The PROBLEM is purely in the loss. Fixing the loss preserves all existing code and checkpoints while potentially achieving open-vocab.

This is the **lowest disruption** option — you keep the exact same forward pass, evaluation code, and model checkpoint structure. Only the training loss changes.

---

### Option C1: Large Negative Bank

#### How It Works

Currently InfoNCE has 16 denominators. Make it 200+.

```python
# Before training: precompute a static bank
negative_bank = CLIP_text("a pole in a scene", "a bench in a scene", ...)  # (200, 768)

# During training:
def token_loss_with_negatives(pred_tokens, gt_classes, text_protos_16, negative_bank):
    """
    pred_tokens: (Q_matched, 768) — token head output
    gt_classes: (Q_matched,) — GT class indices
    text_protos_16: (16, 768) — known class prototypes
    negative_bank: (200, 768) — additional negative concepts
    """
    # Combine known + novel into one big prototype matrix
    all_protos = torch.cat([text_protos_16, negative_bank])  # (216, 768)

    # Cosine similarities
    sim = pred_tokens @ all_protos.T / self.temperature  # (Q, 216)

    # GT labels are indices 0-15 (the known classes come first)
    loss = F.cross_entropy(sim, gt_classes)

    return loss
```

#### Why It Works

With 216 denominators, the token head cannot simply learn 16 tunnels. To correctly assign "car" a higher score than "van", "SUV", "pickup truck" (all present as negatives), it must learn the **fine-grained structure** of CLIP space around "car". This forces generalization.

At inference, when you present "pole" prototype (never seen as a positive), the token head has been trained to push features away from many similar concepts — so it has learned to navigate the full embedding space, not just 16 points.

#### Dynamic Negative Sampling (Advanced)

Instead of a fixed bank, sample different negatives each batch:
```python
# Each batch: sample 64 random negatives from the 1000-concept bank
batch_neg_idx = torch.randint(0, 1000, (64,))
batch_negatives = full_bank[batch_neg_idx]  # (64, 768)
all_protos = torch.cat([text_protos_16, batch_negatives])  # (80, 768)
```

This provides maximum diversity across training and prevents the model from memorizing the negative bank.

---

### Option C2: Distillation from Frozen Features (Student-Teacher)

#### How It Works

Add a regularization term that keeps the token head output close to what the frozen backbone would produce for the same mask region. The backbone features ARE in CLIP space — the token head should produce similar outputs.

```python
def token_loss_with_distillation(pred_tokens, gt_classes, text_protos, backbone_feats, pred_masks):
    # 1. Standard contrastive loss (existing)
    sim = pred_tokens @ text_protos.T / temperature
    L_contrastive = F.cross_entropy(sim, gt_classes)

    # 2. Distillation: token head output should match mask-pooled backbone
    with torch.no_grad():
        backbone_normed = F.normalize(backbone_feats, dim=-1)
        mask_weights = pred_masks.sigmoid().detach()  # detach to avoid mask gradients
        mask_weights = mask_weights / mask_weights.sum(0).clamp(1)
        teacher_feats = F.normalize(mask_weights.T @ backbone_normed, dim=-1)  # (Q, 768)

    L_distill = (1 - F.cosine_similarity(pred_tokens, teacher_feats, dim=-1)).mean()

    # 3. Combined
    return L_contrastive + λ_distill * L_distill  # λ_distill = 0.5
```

#### Why It Works

The distillation term acts as an **anchor** preventing drift. Even as the contrastive loss tries to push tokens toward GT prototypes, the distillation loss says "but don't move too far from what the backbone naturally produces for this region." This maintains the general CLIP space structure.

Think of it as: contrastive loss says "move toward car", distillation loss says "but stay in the neighborhood of where backbone features already point." The result is a token head that refines backbone features rather than replacing them.

---

### Option C3: Embedding Space Regularization

#### How It Works

Penalize the token head if its output lands in a "dead zone" of CLIP space far from any known concept.

```python
def token_loss_with_manifold_reg(pred_tokens, gt_classes, text_protos, concept_bank):
    """
    concept_bank: (1000, 768) — dense sampling of CLIP text space
    """
    # Standard contrastive
    L_contrastive = ...  # same as before

    # Manifold regularization: nearest concept distance
    distances = 1 - pred_tokens @ concept_bank.T  # (Q, 1000)
    min_dist = distances.min(dim=-1).values  # (Q,) — distance to nearest concept
    L_manifold = min_dist.mean()  # penalize being far from ANY concept

    # Diversity regularization: prevent all tokens from collapsing to same point
    token_sim = pred_tokens @ pred_tokens.T  # (Q, Q)
    L_diversity = token_sim.triu(diagonal=1).mean()  # penalize high pairwise similarity

    return L_contrastive + 0.1 * L_manifold + 0.01 * L_diversity
```

#### Why It Works

- **Manifold loss**: ensures token head outputs stay on the "surface" of meaningful CLIP concepts, not in empty space between them
- **Diversity loss**: prevents mode collapse where all queries produce the same embedding (e.g., everything becomes "car")

---

## Strategy D: Camera Features as Open-Vocab Source (OpenMask3D Style)

### What It Is

The most architecturally significant change. Instead of using 3D backbone features for classification, use **2D CLIP image features** directly extracted from the 6 nuScenes cameras. For each predicted 3D mask, project it onto relevant cameras, pool CLIP ViT patch features within the projected region, and classify by cosine similarity to text.

### Why It Is Useful

The fundamental weakness of 3D backbone features for open-vocab is:

1. DisNet was distilled from OpenSeg using **only 16 class labels** — the distillation objective already biased features toward those 16 classes
2. Point-level distillation loses spatial context — per-voxel features don't capture object-level semantics well
3. The distillation process is lossy — 3D features are a degraded copy of 2D features

**2D CLIP features are the source of truth.** They haven't been degraded by distillation. A ViT-L/14 produces 768-dim patch tokens that are perfectly aligned to CLIP text space because they ARE the CLIP image encoder's native output.

### How It Works Step by Step

```
                     FULL PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Predict class-agnostic 3D masks (existing decoder)
═════════════════════════════════════════════════════════
  LiDAR → Voxelize → DisNet → Decoder → pred_masks (V, Q)

Step 2: Load calibration for this sample
═════════════════════════════════════════
  For each of 6 cameras, load:
    - K: (3,3) intrinsic matrix
    - T_lidar_to_cam: (4,4) extrinsic (lidar → camera coords)
  (Already available from nuScenes devkit)

Step 3: Project 3D voxels to 2D per camera
═══════════════════════════════════════════
  For each camera c in {front, front_left, front_right, back, back_left, back_right}:
    # Transform 3D points to camera frame
    pts_cam = T_lidar_to_cam[c] @ homogeneous(voxel_coordinates)  # (V, 3)

    # Project to pixel coordinates
    uv = K[c] @ pts_cam                                           # (V, 3)
    uv = uv[:, :2] / uv[:, 2:3]                                  # (V, 2) pixel coords

    # Validity mask: in front of camera + within image bounds
    valid_c = (pts_cam[:, 2] > 0) & (0 <= uv[:, 0] < W) & (0 <= uv[:, 1] < H)

    # Store: pixel_coords[c] = uv, valid_mask[c] = valid_c

Step 4: Extract CLIP ViT patch features per camera
════════════════════════════════════════════════════
  For each camera c:
    img_c = load_image(camera_path[c])                    # (3, 336, 336) after resize
    patch_feats_c = CLIP_ViT_visual(img_c)                # (576, 768) = 24×24 patches
    patch_feats_c = F.normalize(patch_feats_c, dim=-1)    # unit sphere

    # Map pixel coords to patch grid
    patch_uv = uv[valid_c] / patch_size                   # which patch each point falls in
    feat_per_point_c = bilinear_sample(patch_feats_c, patch_uv)  # (V_valid, 768)

Step 5: For each query/mask, pool 2D features across views
═══════════════════════════════════════════════════════════
  For each query q with mask pred_masks[:, q]:
    mask_probs = pred_masks[:, q].sigmoid()               # (V,) soft mask

    all_feats_for_q = []
    for c in cameras:
      # Get features for visible points in this camera
      visible_in_c = valid_mask[c]                        # which voxels are seen
      cam_feats = feat_per_point_c[visible_in_c]          # (V_c, 768)
      cam_weights = mask_probs[visible_in_c]              # (V_c,) mask weights

      if cam_weights.sum() > threshold:
        weighted_feat = (cam_weights.unsqueeze(-1) * cam_feats).sum(0) / cam_weights.sum()
        all_feats_for_q.append(weighted_feat)

    # Average across cameras
    if len(all_feats_for_q) > 0:
      query_feat_q = F.normalize(torch.stack(all_feats_for_q).mean(0), dim=-1)
    else:
      query_feat_q = zeros(768)  # fallback: no camera coverage

Step 6: Classify via text similarity
═════════════════════════════════════
  query_feat = stack([query_feat_0, ..., query_feat_Q])   # (Q, 768)
  text_feats = CLIP_text(["car", "pedestrian", ..., "pole", "bench"])
  logits = query_feat @ text_feats.T                       # (Q, C_open)

  # Panoptic assignment
  scores, labels = logits.max(dim=-1)
  # Greedy non-overlapping assignment...
```

### Implementation as Training-Time Camera Cross-Attention

Instead of only using cameras at inference, you can integrate them into training:

```python
class CameraEncoder(nn.Module):
    """Frozen CLIP ViT that extracts patch tokens from 6 cameras."""
    def __init__(self, clip_model="ViT-L/14@336px"):
        self.vit = open_clip.create_model(clip_model).visual
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, images_6views):
        """
        images_6views: (6, 3, 336, 336) — preprocessed camera images
        Returns: (6, 576, 768) — patch tokens per camera
        """
        with torch.no_grad():
            tokens = []
            for img in images_6views:
                t = self.vit(img.unsqueeze(0), output_hidden_states=True)
                tokens.append(t.last_hidden_state[:, 1:, :])  # skip CLS token
        return torch.cat(tokens, dim=1)  # (1, 3456, 768)

class CameraCrossAttentionLayer(nn.Module):
    """Queries attend to camera patch tokens (Phase 6 master plan)."""
    def __init__(self, d_model=256, nhead=8, cam_dim=768):
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.cam_proj = nn.Linear(cam_dim, d_model)  # project 768 → 256
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, cam_tokens):
        cam_projected = self.cam_proj(cam_tokens)  # (B, 3456, 256)
        q = queries.transpose(0, 1)
        kv = cam_projected.transpose(0, 1)
        out = self.cross_attn(q, kv, kv)[0].transpose(0, 1)
        return self.norm(queries + out)
```

### Why This Could Work

1. **Pure CLIP features** — not degraded by distillation, in perfect alignment with text space
2. **Multi-view aggregation** — same object seen from multiple angles gives robust representation
3. **Proven in OpenMask3D** — this exact pipeline works for indoor 3D open-vocab
4. **Thesis contribution** — camera cross-attention is the primary novel module in the master plan
5. **Addresses root cause** — classification uses IMAGE features, not 3D backbone features

### Why This Could Fail

1. **Computational cost** — CLIP ViT on 6 images ≈ 1s per sample (but only at inference, or can precompute)
2. **Projection accuracy** — nuScenes calibration isn't perfect, especially for far objects
3. **Occlusion** — some 3D points are behind other objects from camera's perspective
4. **Small objects** — a traffic cone covers maybe 5 patches in one camera view

### How to Address Those

- **Precompute** CLIP features for all training images offline (store as .pt files — similar to existing OpenSeg fusion pipeline)
- **Multi-view fusion** naturally handles occlusion (average over cameras where visible)
- **Patch interpolation** helps small objects (bilinear sampling rather than nearest-patch)
- **Start with inference-only** (no additional training cost), add cross-attention later if beneficial

---

## Strategy E: Geometric Ensemble (FC-CLIP Style)

### What It Is

A no-retraining, inference-only trick. Combine the existing strong linear head (PQ=0.476 on known classes) with mask-pooled backbone features via geometric mean of probabilities. The linear head handles known classes; CLIP features handle novel classes.

### Why It Is Useful

This is the **fastest thing to try** — zero training, zero code changes to the model. Just a new inference function. It costs nothing and tells you immediately whether the backbone features have ANY novel-class signal.

FC-CLIP (CVPR 2024) proved this works: a frozen CLIP feature pathway + a trained in-vocabulary head, combined via geometric mean, gives best of both worlds.

### How It Works Step by Step

```python
def panoptic_inference_ensemble_openvocab(
    pred_logits,     # (Q, 17) — linear head output (16 classes + no-object)
    pred_masks,      # (V, Q) — mask logits
    backbone_feats,  # (V, 768) — frozen backbone features
    text_protos_22,  # (22, 768) — 16 known + 6 novel class prototypes
    alpha=0.5,       # blend weight (0=pure CLIP, 1=pure linear)
    temperature=0.1  # softmax temperature for CLIP logits
):
    V, Q = pred_masks.shape
    num_known = 16

    # ═══ LINEAR HEAD SCORES (known classes only) ═══
    linear_probs = pred_logits[:, :num_known].softmax(dim=-1)  # (Q, 16)
    # Pad with zeros for novel classes
    linear_probs_22 = torch.zeros(Q, 22)
    linear_probs_22[:, :16] = linear_probs

    # ═══ CLIP POOL SCORES (all 22 classes) ═══
    backbone_normed = F.normalize(backbone_feats, dim=-1)        # NORMALIZE FIRST
    mask_probs = pred_masks.sigmoid()                            # soft masks
    weights = mask_probs / mask_probs.sum(0).clamp(min=1)        # per-query normalized
    pooled = F.normalize(weights.T @ backbone_normed, dim=-1)    # (Q, 768)

    clip_logits = pooled @ text_protos_22.T / temperature        # (Q, 22)
    clip_probs = clip_logits.softmax(dim=-1)                     # (Q, 22)

    # ═══ GEOMETRIC MEAN ═══
    # For known classes (0-15): blend both sources
    known_combined = (linear_probs_22[:, :16] ** alpha) * (clip_probs[:, :16] ** (1-alpha))
    # For novel classes (16-21): only CLIP (linear has no opinion)
    novel_combined = clip_probs[:, 16:]  # unblended

    combined = torch.cat([known_combined, novel_combined], dim=-1)  # (Q, 22)
    combined = combined / combined.sum(dim=-1, keepdim=True)  # renormalize

    # ═══ PANOPTIC ASSIGNMENT ═══
    scores, labels = combined.max(dim=-1)

    # Determine objectness: use linear head's no-object score
    obj_score = 1 - pred_logits[:, -1].softmax(dim=-1)  # 1 - P(no-object)
    final_score = scores * obj_score

    # Sort by score, greedy non-overlapping assignment
    order = final_score.argsort(descending=True)
    masks_binary = pred_masks[:, order].sigmoid() > 0.5

    sem_pred = torch.zeros(V, dtype=torch.long)
    inst_pred = torch.zeros(V, dtype=torch.long)
    occupied = torch.zeros(V, dtype=torch.bool)
    inst_id = 0

    for i in range(Q):
        q = order[i]
        mask = masks_binary[:, i] & ~occupied
        if mask.sum() / masks_binary[:, i].sum() < overlap_thr:
            continue  # too much overlap
        if mask.sum() < 50:
            continue  # too small

        sem_pred[mask] = labels[q]
        inst_id += 1
        inst_pred[mask] = inst_id
        occupied |= mask

    return sem_pred.numpy(), inst_pred.numpy()
```

### Key Difference from Previous Failed Ensemble

The previous ensemble (Phase 5) failed because:
1. It blended linear softmax with cosine similarity raw scores — incompatible scales
2. It didn't normalize backbone features before pooling
3. It used binary masks, not soft masks for pooling
4. Best α=0.0 (pure token) showed the scores couldn't be combined

**Fixed version:**
- Both pathways converted to proper probability distributions (softmax)
- Backbone features L2-normalized before pooling
- Soft mask-weighted pooling
- Temperature scaling on CLIP logits for calibration
- Novel classes get CLIP-only scores (not blended with zeros)

### Why This Could Fail

If the backbone features truly don't carry novel-class information (because distillation was biased to 16 classes), then no amount of ensemble tricks will help. But this is the cheapest way to find out.

---

## Strategy F: Vocabulary-Free Region-Text Alignment

### What It Is

Train the system using **image captions** rather than fixed class labels. For each predicted mask region, generate a pseudo-caption (using a VLM like BLIP-2), then align the mask's embedding to the text embedding of that caption. This mirrors how OpenSeg achieved open-vocab using image-caption pairs.

### Why It Is Useful

The fundamental reason the 16-prototype approach fails is **vocabulary poverty**. The token head only ever sees 16 concepts. Real open-vocab systems like CLIP were trained with 400M image-text pairs covering millions of concepts. OpenSeg used 4M image-caption pairs.

If you can supply diverse, per-region text supervision, the token head learns to span the full CLIP embedding space — not just 16 points.

### How It Would Work

```
OFFLINE PREPROCESSING:
═══════════════════════
  For each training sample:
    1. Load 6 camera images
    2. Project GT 3D panoptic masks onto camera views
    3. For each mask region visible in a camera:
       - Crop the image to the mask's bounding box
       - Run BLIP-2 / LLaVA: "What object is in this image?" → caption
       - Run CLIP text encoder on caption → text embedding (768-d)
    4. Store: per-mask text embeddings (diverse, per-instance)

TRAINING:
═════════
  Standard mask loss (BCE + Dice) for spatial learning
  Token loss: align pred_token → caption_embedding (per-instance text target)

  Key: each instance has a UNIQUE caption embedding, not a shared class prototype
  "Red Toyota sedan" ≠ "parked white SUV" ≠ "fast-moving delivery van"
  All are class "car" but have different CLIP embeddings
  → token head must learn the GENERAL mapping, not memorize 16 points
```

### Why This Could Work

- Creates effectively unlimited vocabulary diversity
- Per-instance supervision forces generalization
- OpenSeg proved this paradigm works for open-vocab

### Why This Would Likely Fail for This Thesis

- Requires running a VLM on possibly 50,000+ crop regions (massive compute)
- Captions for LiDAR-projected masks may be noisy (partial views, small crops)
- Engineering complexity is enormous — new preprocessing pipeline, new data format
- Well beyond scope of a single-GPU thesis project

### When to Consider This

Only if you have:
- Months of additional time
- Multi-GPU compute for the captioning pipeline
- A clear publication goal (not just thesis completion)

---

## Summary: Decision Matrix

```
┌──────────────────────────────────────────────────────────────────────┐
│ IF you want the FASTEST test of open-vocab signal:                    │
│   → Strategy E (geometric ensemble, no retraining)                   │
│                                                                        │
│ IF you want the SIMPLEST retraining approach:                         │
│   → Strategy A (remove CE, mask-pool backbone features)              │
│                                                                        │
│ IF you want the STRONGEST open-vocab with moderate effort:            │
│   → Strategy C1 (keep token head, add 200 negative bank)            │
│                                                                        │
│ IF you want the BEST thesis contribution + strongest open-vocab:      │
│   → Strategy D (camera CLIP features, OpenMask3D paradigm)           │
│                                                                        │
│ IF you want MAXIMUM theoretical open-vocab:                           │
│   → Strategy F (region-text alignment) — but likely out of scope     │
│                                                                        │
│ IF you want a BALANCED approach (good open-vocab + manageable):       │
│   → Strategy B (class-agnostic + diverse text adapter)               │
└──────────────────────────────────────────────────────────────────────┘
```

### Recommended Execution Order

```
Week 1: Strategy E → tells you if backbone has novel-class signal at all
         (2 hours implementation, diagnostic only)

Week 1: Fix backbone pooling bug → rerun evaluation with normalized features
         (1 hour, critical diagnostic)

Week 2: Strategy A → class-agnostic retraining + proper mask pooling
         (1-2 days retraining on A100)

Week 2: If A shows promise → Strategy D inference-only (cameras at eval)
         (3-4 days implementation + eval)

Week 3: Strategy D full → camera cross-attention in decoder (Phase 6)
         (1 week implementation + training)

Week 3: Strategy C1 as backup → if camera path is too expensive
         (2-3 days retraining)
```

---

## Strategy E Evaluation Results (Completed 2026-05-26)

Strategy E was implemented and evaluated on v1.0-trainval (6019 samples). Results:

| Mode | PQ | Novel Voxels |
|------|-----|------|
| Linear (baseline) | 0.477 | N/A |
| Backbone-Pool legacy | 0.209 | 25.4M (mostly "pole" FPs) |
| Backbone-Pool-v2 (fixed) | 0.221 | 23.9M (still "pole" dominated) |
| Geo-Ensemble α=0.5 (best) | 0.463 | 171K (748 police car, 7.5K trash cans, 163K pole) |

**Conclusion**: Strategy E validated the *framework* but novel detection signal is weak. The backbone features carry some open-vocab information (police car detected!) but feature-pooling (averaging) within masks destroys per-voxel CLIP alignment — this is the root cause of weak novel signal. Prototype bias correction was also implemented (±0.08 range) but proved too small to help, further confirming that pooling degradation, not prototype affinity, is the bottleneck. **Next step: Strategy K/L (per-voxel logit aggregation) to avoid the pooling problem entirely.**

---

## Strategy G: CLIP-Initialized Queries (Semantic Query Priors)

### What It Is

Replace the 100 randomly initialized learnable queries in the transformer decoder with queries initialized from CLIP text embeddings. Each query starts pre-aligned to a semantic concept, giving the decoder a "head start" on specialization.

### Why It Is Useful

Currently, the 100 queries are randomly initialized (`nn.Embedding(100, 256)`). During training, queries self-organize to specialize on different objects via Hungarian matching. But this specialization is arbitrary — there's no guarantee query #37 learns "car" and query #72 learns "pedestrian".

With CLIP-initialized queries:
1. Queries start in semantic space — query for "car" begins near the CLIP representation of "car"
2. The decoder potentially converges faster (queries already know what to look for)
3. At inference, adding new queries for novel classes (e.g., initialize from "police car" text embedding) could enable zero-shot query specialization
4. Compatible with existing training — just changes initialization, same loss

### How It Works Step by Step

```python
# Initialization (replace random nn.Embedding)
class CLIPInitializedQueries(nn.Module):
    def __init__(self, num_queries=100, hidden_dim=256, clip_dim=768):
        super().__init__()
        # 16 known class queries + 84 additional specific concepts
        concepts = [
            # 16 nuScenes known (one query per class, or multiple for things)
            "car", "car", "car", "car", "car",  # 5 car queries (many instances)
            "truck", "truck", "truck",
            "bus", "bus",
            "pedestrian", "pedestrian", "pedestrian", "pedestrian", "pedestrian",
            "bicycle", "motorcycle",
            "construction vehicle", "trailer",
            "traffic cone", "traffic cone",
            "barrier", "barrier",
            # 6 stuff queries
            "driveable surface", "sidewalk", "terrain",
            "manmade structure", "vegetation", "other flat surface",
            # Additional generic/novel concepts for generalization
            "pole", "bench", "trash can", "fire hydrant",
            "tree trunk", "street light", "stop sign",
            "parking meter", "fence", "guard rail",
            # ... fill to 100 with driving-relevant concepts
        ]
        # Encode with CLIP text encoder
        clip_text_feats = clip_encode(concepts)  # (100, 768)
        # Project to decoder dimension
        self.proj = nn.Linear(clip_dim, hidden_dim)
        self.register_buffer("clip_init", clip_text_feats)
        # Learnable residual (allows optimization while keeping prior)
        self.residual = nn.Parameter(torch.zeros(num_queries, hidden_dim))

    def forward(self, batch_size):
        base = self.proj(self.clip_init)  # (100, 256)
        queries = base + self.residual  # prior + learnable delta
        return queries.unsqueeze(0).expand(batch_size, -1, -1)
```

### Key Design Choices

1. **Residual design**: `query = proj(CLIP_text) + learnable_delta`. The CLIP prior provides a semantic starting point. The learnable delta allows the query to adapt to the decoder's learned attention patterns.
2. **Multiple queries per class**: Things (car, pedestrian) need multiple queries for multiple instances. Stuff classes need one query each.
3. **Extra concept queries**: Including novel driving concepts in the initial query set means the decoder has queries "ready" for those objects, even if no CE loss is applied for them.
4. **Open-vocab extension at inference**: Simply add new queries (project new text embeddings) without retraining.

### Why This Could Help Both Closed and Open Vocab

**Closed set improvement:**
- Faster convergence — queries start specialized instead of random
- Better instance coverage — number of queries per class can be proportional to frequency
- Less query competition — queries know their role from the start

**Open vocab:**
- Novel concept queries exist from initialization — decoder may learn generalizable attention patterns
- At inference, adding "police car" query (from CLIP text) could trigger detection if decoder learned concept-agnostic mask prediction
- The residual term preserves CLIP alignment — queries stay semantically meaningful

### Why This Could Fail

1. The decoder's cross-attention learns positional/geometric patterns, not semantic — initialization may get overwritten
2. By epoch 5, initial CLIP alignment may be completely lost (gradients dominate)
3. 100 queries for 100 concepts may be too rigid — current random init allows flexible specialization
4. The proj(768→256) compression may lose too much CLIP information

### Mitigation

- Use **cosine warmup** on the residual (small LR initially) to preserve CLIP prior longer
- Regularize: add loss `||residual||² * λ` to prevent drift from CLIP init
- Experiment with frozen proj vs learned proj
- Start with fewer concept-specific queries (30-40) and keep 60 "free" random queries

---

## Strategy H: End-to-End Encoder Unfreezing with CLIP Regularization

### What It Is

Unfreeze the DisNet backbone and fine-tune it end-to-end alongside the panoptic decoder, but with a regularization loss that prevents the features from drifting away from CLIP space.

### Why It Is Useful

Currently DisNet is frozen during panoptic training. This means:
- The backbone features are optimized for per-voxel OpenSeg alignment (pixel-level), NOT for object-level panoptic segmentation
- The 256-dim projected features (`backbone_proj`) are the only pathway for task adaptation
- Features in different parts of a mask may point in different directions (because they were trained per-pixel)

Unfreezing allows:
1. Features within the same object to become more **coherent** (pointing same direction) because the panoptic loss backpropagates through the backbone
2. Features to become more **discriminative** for the 16 panoptic classes (and potentially novel classes if combined with diverse-text loss)
3. The backbone to learn **spatial patterns** relevant to panoptic (e.g., object boundaries, stuff continuity)

### How It Works

```python
# Phase 1: Current (frozen backbone)
for p in backbone.parameters():
    p.requires_grad = False

# Phase 2: Unfrozen with CLIP regularization
for p in backbone.parameters():
    p.requires_grad = True  # Unfreeze!

# Add regularization loss during training:
def clip_alignment_regularization(curr_feats, frozen_feats, mask=None, lambda_reg=0.1):
    """Keep unfrozen features close to original CLIP-aligned features.

    curr_feats: (V, 768) — features from unfreezing backbone (current forward pass)
    frozen_feats: (V, 768) — cached features from frozen backbone (computed once per sample)
    mask: optional (V,) bool — only regularize on valid/foreground voxels
    """
    curr_normed = F.normalize(curr_feats, dim=-1)
    frozen_normed = F.normalize(frozen_feats, dim=-1)

    # Cosine similarity loss (keep features aligned)
    cos_sim = (curr_normed * frozen_normed).sum(dim=-1)  # (V,)
    if mask is not None:
        cos_sim = cos_sim[mask]
    loss = (1 - cos_sim).mean()  # push toward similarity=1

    return lambda_reg * loss
```

### Training Protocol

```
Phase 1 (Epochs 1-10): Freeze backbone, train decoder normally (current setup)
Phase 2 (Epochs 11-30): Unfreeze backbone with:
    - Backbone LR = 1e-6 (10x smaller than decoder)
    - CLIP alignment regularization (λ=0.1)
    - Gradient clipping on backbone (max_norm=1.0)
    - Monitor: cosine similarity between frozen vs current features (should stay >0.9)

Loss_total = L_panoptic + λ_clip * L_clip_alignment
           = (L_CE + L_mask + L_dice + L_token) + 0.1 * L_cosine_drift
```

### Expected Impact on Closed-Set PQ

This should **improve** closed-set PQ because:
- Backbone features become more coherent *within objects* → better mask pooling
- Cross-attention in decoder sees more task-relevant features → better queries
- Estimated improvement: PQ 0.477 → 0.50+ based on similar methods (Mask2Former fine-tuning gains)

### Expected Impact on Open-Vocab

**Risk**: Unconstrained fine-tuning will destroy CLIP alignment. Features drift toward 16-class discriminative space → open-vocab dies.

**Mitigation via regularization**: The cosine alignment loss keeps features on the CLIP manifold. Empirically, λ=0.1 allows features to adapt while maintaining >0.9 average cosine similarity to original CLIP features.

**Combined approach**: Unfreeze + diverse text negatives (Strategy C1). The backbone learns to be discriminative among 200+ concepts, not just 16.

### Why This Could Fail

1. Catastrophic forgetting — even with regularization, 768-dim space may distort subtly enough to kill novel-class similarity
2. Overfitting on trainval — backbone memorizes training set, doesn't generalize
3. Training instability — gradients through sparse convolutions + long backward pass may diverge
4. Memory cost — storing activations for 100K voxels × 768 dims for backprop is expensive

### Mitigation

- **LoRA/low-rank adaptation**: Instead of full unfreezing, add low-rank adapters to backbone layers (rank 16-32). Much less risk of catastrophic forgetting.
- **Gradient accumulation**: If memory is tight, accumulate over 4 samples
- **Periodic evaluation**: Run open-vocab eval every 2 epochs, stop if novel detection degrades
- **EMA teacher**: Keep exponential moving average of frozen backbone as regularization target

---

## Strategy I: Voxel Decomposition Loss (Per-Point CLIP Supervision)

### What It Is

Add a per-voxel contrastive loss that directly supervises backbone features against CLIP text embeddings at the point level. Instead of only supervising at the mask/query level, this ensures every voxel feature is individually aligned with its ground-truth class text embedding.

### Why It Is Useful

Current training supervises features only through:
1. Hungarian-matched queries (mask level) — 100 queries for ~30 objects per scene
2. Token head (query level) — maps query embedding to text space

This leaves ~50,000 voxels per sample completely *unsupervised* in terms of CLIP text alignment. The backbone features may be "close" to the right text embedding (from pre-training) but are never explicitly pushed toward them during panoptic training.

Voxel decomposition loss provides:
1. **Dense supervision** — every foreground voxel gets a gradient toward its class text embedding
2. **Feature coherence** — voxels within the same object are pushed toward the same text embedding
3. **Open-vocab foundation** — if features are tightly aligned to text per-voxel, mask-pooling becomes near-trivial for any text query

### How It Works

```python
def voxel_decomposition_loss(backbone_feats, gt_sem, text_protos, temperature=0.07):
    """Per-voxel contrastive loss against class text prototypes.

    backbone_feats: (V, 768) — backbone output (possibly unfrozen)
    gt_sem: (V,) — per-voxel GT semantic class (0-15, 255=ignore)
    text_protos: (C, 768) — text embeddings for C classes (could be 16 or 200+)
    """
    # Only supervise valid voxels (not ignore)
    valid = gt_sem != 255
    feats = F.normalize(backbone_feats[valid], dim=-1)  # (V', 768)
    labels = gt_sem[valid]  # (V',) class indices

    # Cosine similarity to all prototypes
    sim = feats @ text_protos.T / temperature  # (V', C)

    # Cross-entropy: each voxel should be closest to its GT class prototype
    loss = F.cross_entropy(sim, labels)

    return loss
```

### Extended Version: With Diverse Negative Bank

```python
def voxel_decomp_loss_diverse(backbone_feats, gt_sem, text_protos_16, negative_bank):
    """With 200+ negatives — forces fine-grained per-voxel CLIP alignment."""
    all_protos = torch.cat([text_protos_16, negative_bank])  # (216+, 768)
    valid = gt_sem != 255
    feats = F.normalize(backbone_feats[valid], dim=-1)
    labels = gt_sem[valid]  # targets are still 0-15 (known classes)

    sim = feats @ all_protos.T / 0.07  # (V', 216+)
    loss = F.cross_entropy(sim, labels)  # labels index into first 16 positions
    return loss
```

### Why This Is Different from Original OpenSeg Distillation

OpenSeg distillation (how DisNet was originally trained):
- Supervision: `L2_loss(backbone_feat, openseg_2d_feat)` — **per-pixel regression** to 2D features
- Problem: 2D features are already imperfect (projected from images), and L2 loss doesn't enforce text alignment directly

Voxel decomposition loss:
- Supervision: `CrossEntropy(backbone_feat @ text_protos.T, gt_class)` — **directly aligns to text embeddings**
- Advantage: Uses the CLIP text space directly, not degraded 2D features
- Advantage: With diverse negatives, forces navigation of full embedding space

### Training Integration

```python
# Combined loss
loss = (
    w_ce * loss_ce          # Query-level class loss (existing)
    + w_mask * loss_mask    # Mask BCE (existing)
    + w_dice * loss_dice    # Mask Dice (existing)
    + w_token * loss_token  # Token contrastive (existing)
    + w_voxel * loss_voxel  # NEW: per-voxel decomposition
)
# Suggested: w_voxel = 0.5 (after warmup of 5 epochs)
```

### Expected Impact

**Closed-set PQ improvement:**
- Per-voxel alignment makes mask-pooled features more coherent → better token head classification → higher RQ
- Backbone features become more discriminative → better linear head predictions
- Estimated: PQ +0.02-0.05

**Open-vocab improvement:**
- With diverse negatives: every voxel navigates full CLIP space → mask-pooling produces quality embeddings for any concept
- Geometric ensemble should work much better when per-voxel features are tightly aligned
- Strategy E (geo ensemble) could go from PQ_novel≈0 to measurable novel detection

### Why This Could Fail

1. **Computational cost**: 50K voxels per sample × 200+ prototypes = 10M similarity computations per sample
2. **Conflict with query-level loss**: Per-voxel loss may fight against Hungarian matching (different optimization targets)
3. **Class boundary issues**: Voxels near object boundaries may be ambiguous — forcing them to one class is wrong
4. **Background problem**: Most voxels are background/ignore. If GT coverage is sparse, most voxels get no gradient.

### Mitigation

- **Subsample**: Only compute on 10K random valid voxels per sample (not all 50K)
- **Soft labels**: Near boundaries, use soft targets based on distance to GT mask edges
- **Ramping weight**: Start with w_voxel=0, ramp to 0.5 over 5 epochs (let mask decoder stabilize first)
- **Only foreground**: Only apply to voxels within GT masks (not unlabeled background)

---

## Strategy J: Improved Backbone Training from Scratch (CLIP-Aligned Panoptic Encoder)

### What It Is

Instead of using the pre-trained DisNet (distilled from OpenSeg), train a new 3D backbone from scratch that is simultaneously optimized for:
1. CLIP text alignment (per-voxel)
2. Panoptic mask quality
3. Full vocabulary coverage (1000+ concepts in contrastive loss)

### Why It Is Useful

The current DisNet was distilled from OpenSeg with only a reconstruction objective (`L2 loss` to 2D features). It was never explicitly trained for:
- Text alignment (it's indirectly aligned through OpenSeg features, which are text-aligned)
- Panoptic awareness (object boundaries, instance separation)
- Diverse vocabulary (OpenSeg used specific segmentation labels for its 500 classes, but nuScenes distillation used point-wise features)

A purpose-built encoder could simultaneously serve as:
- A panoptic feature extractor (backbone for decoder)
- A CLIP-aligned per-voxel classifier (open-vocab ready)
- A semantic segmentation model (per-point classification)

### Architecture

```
Input: (V, 3) coords + (V, 3) XYZ features
    │
    ▼
MinkUNet18A with CLIP-aligned output head
    │
    ├── Sparse Conv Encoder (5 stages)
    │       │
    │       ▼
    ├── Sparse Conv Decoder (5 stages)
    │       │
    │       ▼
    │   feature_out: (V, 768) — main output
    │       │
    │       ├── panoptic_proj: Linear(768→256) → to panoptic decoder
    │       │
    │       └── clip_head: LayerNorm + L2Norm → to text contrastive loss
    │
    └── Training losses:
            ├── L_voxel_contrastive: per-voxel vs 1000 text embeddings
            ├── L_clip_distill: cosine similarity to OpenSeg 2D features (keep alignment)
            └── (optional) L_panoptic: from downstream decoder (end-to-end)
```

### Training Procedure

```
Stage 1: CLIP-aligned encoder pretraining (10-20 epochs)
    - Input: nuScenes point clouds with GT semantic labels
    - Loss: per-voxel cross-entropy against 200+ text prototypes
    - Data: full nuScenes trainval (28K samples)
    - Goal: backbone features directly in CLIP text space

Stage 2: Joint panoptic training (20 epochs)
    - Input: same point clouds
    - Forward: encoder → panoptic decoder → masks + classes + tokens
    - Loss: panoptic (CE+mask+dice+token) + α*L_voxel_contrastive (regularization)
    - Goal: panoptic-quality masks with CLIP-aligned features
```

### Why This Is Better Than Unfreezing DisNet

1. **No legacy bias**: DisNet was trained with L2 regression to imperfect 2D features. Starting fresh with direct text-alignment produces cleaner features.
2. **Architecture flexibility**: Can add multi-scale features, skip connections optimized for masks
3. **Full curriculum**: The backbone sees panoptic supervision from training, not just at fine-tuning time
4. **Publication value**: "CLIP-aligned 3D encoder for open-vocab panoptic" is a significant contribution

### Why This Could Fail

1. **Training cost**: MinkUNet18A on full nuScenes from scratch = 3-5 days on single A100
2. **Data limited**: Only 28K samples with panoptic labels (vs CLIP's 400M image-text pairs)
3. **May not beat DisNet**: DisNet benefits from OpenSeg's large-scale 2D pretraining; a from-scratch 3D model may underperform
4. **Risk of overfitting**: Small 3D dataset + large model = potential memorization

### When to Choose This

- If unfreezing DisNet (Strategy H) shows clear improvement and you establish that better backbone features = better panoptic
- If you have 1+ week of compute time available
- If this is targeted for a publication contribution

---

## Critical Finding #1: Backbone IS CLIP-Aligned — Pooling Degrades Signal

**Date**: 2026-05-27

### Evidence

Prototype bias calibration showed per-prototype background affinity in range ±0.08. This is **mathematically correct** for a CLIP-aligned space:
- Random groups of voxels (from different classes) → averaged → cancel to near-zero
- Near-zero cosine similarity to any specific text prototype = expected behavior in 768-dim space
- The backbone meets its training objective: per-voxel CLIP alignment via cosine distillation

### Root Cause of Weak Novel Detection

The problem is NOT that features aren't CLIP-aligned. It's that **averaging features over a mask** destroys the per-voxel signal:

```
Individual voxel (car):     cos_sim("car") ≈ 0.35, cos_sim("truck") ≈ 0.20  ← discriminative!
Individual voxel (road):    cos_sim("road") ≈ 0.30, cos_sim("car") ≈ 0.05   ← discriminative!

Averaged over mask (car+road+building voxels mixed):
  cos_sim("car") ≈ 0.08, cos_sim("road") ≈ 0.07, cos_sim("truck") ≈ 0.06  ← NOISE!
```

Averaging normalized vectors from different classes produces a near-zero vector. Re-normalizing it creates a random direction with no class specificity.

### Implication for All Strategies

Any approach that pools/averages backbone features before classifying will suffer from this. The correct approach is:
1. Classify per-voxel FIRST (each voxel is CLIP-aligned individually)
2. Aggregate DECISIONS (votes/logits) within masks

---

## Critical Finding #2: Novel Detections Are All False Positives

**Date**: 2026-05-27

### Evidence

Visual inspection of geo_ensemble_v2 outputs (multi-prompt + bias correction):
- **396K "novel" voxels detected** (24K police car, 352K ambulance, 17K trash cans, 280 pole, 2K bench)
- **Visual verification: ALL are false positives** — random objects (trucks, walls, buildings, road surfaces) incorrectly labeled as novel classes
- "Ambulance" (352K voxels) = white trucks/vans/buildings misclassified
- "Police car" (24K voxels) = dark cars/SUVs misclassified

### Root Cause Analysis

The backbone's per-voxel CLIP alignment is **too weak** for reliable novel-class discrimination:

1. **Distillation was lossy**: DisNet learned from per-pixel OpenSeg features (themselves projected from 2D). The chain is: `CLIP image → OpenSeg → 3D projection → DisNet distillation`. Each step degrades alignment.

2. **Cosine similarity margins are tiny**: For the backbone to reliably detect "police car" vs "car", the cosine similarity difference needs to be significant (>0.1). But 3D distilled features have much smaller margins (~0.01-0.03 between semantically similar classes).

3. **Text prototypes for novel classes are too similar to known classes**:
   - "police car" prototype ≈ "car" prototype (cosine sim ~0.85 between them in CLIP text space)
   - "ambulance" prototype ≈ "truck"/"bus" (white large vehicle descriptions overlap)
   - The backbone cannot distinguish what CLIP text space itself barely separates

4. **No negative training**: The backbone was NEVER trained to push features AWAY from "police car" or "ambulance" — it only learned to match OpenSeg features. Without explicit negatives, there's no decision boundary.

### What This Means for Strategy K/L/M

Per-voxel logit aggregation (K/L) addresses the **pooling degradation** identified in Finding #1. However, the underlying margin weakness may still cause FPs:
- Each car voxel has cos_sim("car") ≈ 0.35 and cos_sim("police car") ≈ 0.33
- Voting within a car mask: many voxels "vote" police car because the margin is noise-level
- **K/L/M MUST be tested on full trainval + 20-30 scene visual verification** before any conclusion

### Current Status (2026-05-28)

**K/L/M EVALUATED ON FULL TRAINVAL (6019 samples). RESULTS:**

| Strategy | PQ | Novel Voxels | vs Linear |
|----------|-----|-------------|-----------|
| Linear (baseline) | 0.4761 | N/A | — |
| geo_v2_a0.5 (E best) | 0.4697 | 330,658 | -0.006 |
| logit_agg_a0.5 (L best) | 0.4702 | 305,525 | -0.006 |
| confidence_gated (M) | 0.4633 | 274,038 | -0.013 |
| voxel_vote (K) | 0.4490 | 1,815,332 | -0.027 |

**Key result: Strategy L ≈ Strategy E.** Per-voxel logit aggregation produces nearly identical results to feature pooling — same top-20 scenes, same voxel counts, same "ambulance" dominance. This DISPROVES the hypothesis that pooling degrades signal. The bottleneck is the backbone features themselves.

**The inference-only chapter is effectively CLOSED:**
- ✗ Feature pooling (E): FPs in 10/10 visual samples
- ✗ Per-voxel logit aggregation (L): identical to E (disproves pooling hypothesis)
- ✗ Voxel vote with margin (K): 5.5× worse over-detection, PQ -0.027
- ✗ Confidence gating (M): reduces volume but same FPs
- ✗ No-bias ablation: DONE — pole catastrophe without bias (voxel_vote 18.6M, logit_agg 830K), L=E confirmed both ways

### Definitive Conclusion Threshold

~~Inference-only open-vocab will be **conclusively closed** when:~~
~~1. K/L/M all evaluated on full trainval (6019 samples)~~
~~2. Per-scene novel detection breakdown produced~~
~~3. Top 20-30 scenes by novel voxel count visually verified~~
~~4. Both with-bias and no-bias ablations tested~~
~~5. If ALL above show only FPs → close inference-only, move to training~~

**Status**: All items DONE (2026-05-29). No-bias ablation confirms pole catastrophe without bias, L=E in both configurations. Inference-only chapter is definitively CLOSED.

### What WILL Work (Requires Training)

1. **Strategy I (voxel decomposition loss with diverse negatives)**: Train the backbone to explicitly push features away from "police car" when the GT is "car". With 200+ negatives including confusable concepts, the backbone learns decision boundaries.

2. **Strategy D (camera CLIP features)**: Bypass the degraded 3D backbone entirely. Use fresh CLIP ViT features from the actual camera images — these have the full resolution of CLIP and can distinguish "police car" from "car".

3. **Strategy H + I combined**: Unfreeze backbone + per-voxel contrastive with diverse negatives. The backbone learns panoptic-aware, fine-grained CLIP alignment.

---

## Strategy K: Per-Voxel Majority Vote within Masks (NO TRAINING)

### What It Is

Instead of pooling backbone features then classifying the pool, classify EACH voxel independently using cosine similarity to text prototypes, then aggregate votes within each predicted mask.

### Why It Should Work

The backbone was trained per-voxel: `backbone_feat[v] → CLIP feature for voxel v`. Each voxel's feature is aligned to its class's text prototype. This is literally how OpenScene's "distill" mode works (achieving 35.4% mIoU). We just apply it *within* each predicted mask to get per-mask class labels.

Voting aggregates many weak per-voxel signals into strong per-mask decisions (law of large numbers). A car mask with 5000 voxels where 70% vote "car" gives high confidence even if individual voxels are noisy.

### How It Works

```python
def panoptic_inference_voxel_vote(pred_logits, pred_masks, backbone_feats,
                                   text_protos_all, num_known=16,
                                   overlap_thr=0.8, min_mask_pts=50):
    """Classify masks by per-voxel CLIP voting (not feature pooling).

    For known classes: use linear head (proven PQ=0.477).
    For novel classes: per-voxel cosine sim vote within each mask.
    """
    V, Q = pred_masks.shape
    masks_sig = pred_masks.sigmoid()  # (V, Q)

    # Per-voxel CLIP classification
    feats_normed = F.normalize(backbone_feats, dim=-1)  # (V, 768)
    per_voxel_logits = feats_normed @ text_protos_all.T  # (V, C_all)

    # Linear head scores for known classes
    linear_probs = pred_logits[:, :num_known].softmax(dim=-1)  # (Q, 16)
    obj_score = 1 - pred_logits.softmax(dim=-1)[:, -1]  # objectness

    # For each query: vote within mask
    scores = torch.zeros(Q)
    labels = torch.zeros(Q, dtype=torch.long)

    for q in range(Q):
        mask_weight = masks_sig[:, q]  # (V,) soft mask
        if mask_weight.sum() < min_mask_pts:
            scores[q] = 0
            continue

        # Weighted vote: sum of per-voxel logits weighted by mask confidence
        weighted_logits = (mask_weight.unsqueeze(-1) * per_voxel_logits).sum(0)  # (C_all,)
        weighted_logits = weighted_logits / mask_weight.sum()  # normalize

        # Decision: use linear head for known, CLIP vote for novel
        linear_score, linear_cls = linear_probs[q].max(dim=-1)
        novel_logits = weighted_logits[num_known:]  # novel class logits
        novel_score = novel_logits.max() if len(novel_logits) > 0 else 0

        # If novel signal is stronger → use it
        if novel_score > weighted_logits[:num_known].max() * 1.2:  # 20% margin
            labels[q] = num_known + novel_logits.argmax()
            scores[q] = novel_score * obj_score[q]
        else:
            labels[q] = linear_cls
            scores[q] = linear_score * obj_score[q]

    # Greedy panoptic assignment (same as before)
    return _greedy_assign(scores, labels, masks_sig, V, overlap_thr)
```

### Key Insight: Average LOGITS, Not FEATURES

```
WRONG (Strategy E pooling):   average(feat_v1, feat_v2, ...) → classify(averaged_feat)
RIGHT (Strategy K):           classify(feat_v1), classify(feat_v2), ... → average(logits)
```

Averaging logits preserves discriminability because each logit is already a class similarity score. Averaging features destroys discriminability because normalized vectors from different classes cancel out.

### Expected Results

- Known-class PQ: same as linear head (0.477) — falls back to linear for known
- Novel detection: significantly better than geo-ensemble because:
  - 5000 voxels voting "trash can" > 1 averaged feature near zero
  - Background voxels within mask get filtered by mask_weight (sigmoid)
  - No pooling degradation

### Why This Could Fail

1. Mask boundaries include voxels from neighboring objects (e.g., car mask extends over road) → wrong votes
2. Novel class prototypes still generic (e.g., "pole" matches many things) → need multi-prompt
3. Threshold for known-vs-novel decision is sensitive

---

## Strategy L: Soft Mask-Weighted Logit Aggregation (NO TRAINING)

### What It Is

A refined version of K. Instead of hard majority vote, use the mask sigmoid values as confidence weights for per-voxel logits. Voxels with high mask confidence contribute more to the class decision.

### How It Differs from Feature Pooling

| Approach | Formula | Problem |
|----------|---------|---------|
| Feature pool (E) | `classify(Σ(w_v * feat_v) / Σw_v)` | destroys per-voxel alignment |
| Logit aggregation (L) | `Σ(w_v * classify(feat_v)) / Σw_v` | preserves per-voxel alignment |

Mathematically: `E[f(x)] ≠ f(E[x])` — classification is nonlinear, so averaging before classifying ≠ classifying then averaging.

### Implementation

```python
def panoptic_inference_logit_aggregation(pred_logits, pred_masks, backbone_feats,
                                          text_protos_all, num_known=16,
                                          overlap_thr=0.8, temperature=0.1):
    V, Q = pred_masks.shape
    masks_sig = pred_masks.sigmoid()  # (V, Q)

    # Per-voxel logits for ALL classes (known + novel)
    feats_normed = F.normalize(backbone_feats.float(), dim=-1)
    voxel_logits = feats_normed @ text_protos_all.T / temperature  # (V, C_all)

    # For each query: soft-weighted logit sum
    # Efficient: matrix multiplication
    # weighted_logits[q, c] = Σ_v masks_sig[v,q] * voxel_logits[v,c] / Σ_v masks_sig[v,q]
    mask_sums = masks_sig.sum(dim=0).clamp(min=1)  # (Q,)
    aggregated_logits = (masks_sig.T @ voxel_logits) / mask_sums.unsqueeze(-1)  # (Q, C_all)

    # Combine with linear head for known classes
    linear_probs = pred_logits[:, :num_known].softmax(dim=-1)  # (Q, 16)
    clip_probs = aggregated_logits.softmax(dim=-1)  # (Q, C_all)

    # Geometric ensemble: known=blend, novel=CLIP-only
    alpha = 0.5
    combined_known = (linear_probs ** alpha) * (clip_probs[:, :num_known] ** (1-alpha))
    combined_novel = clip_probs[:, num_known:]
    combined = torch.cat([combined_known, combined_novel], dim=-1)
    combined = combined / combined.sum(dim=-1, keepdim=True)

    scores, labels = combined.max(dim=-1)
    obj_score = 1 - pred_logits.softmax(dim=-1)[:, -1]
    final_scores = scores * obj_score

    return _greedy_assign(final_scores, labels, masks_sig, V, overlap_thr)
```

### Why This Is the Best No-Training Approach

1. Uses backbone exactly as trained (per-voxel CLIP similarity)
2. Mask weighting ensures high-confidence mask voxels dominate (boundary voxels have low sigmoid)
3. Efficient: one matrix multiply `(Q,V) @ (V,C)` — same cost as current pooling
4. Naturally handles the "pole" problem: if a mask is truly car, most voxels will vote car, drowning out noise

### Computational Note

`masks_sig.T @ voxel_logits` is (Q, V) @ (V, C) = O(Q * V * C). For V=50K, Q=100, C=21, this is 100M operations — similar to current feature pooling. No extra cost.

---

## Strategy M: Confidence-Gated Novel Class Detection (NO TRAINING)

### What It Is

A simple but effective decision framework: use the linear head's confidence as a "gate" to decide whether a mask should be classified as known or novel. Only consider novel classes for masks where the linear head is uncertain.

### Motivation

The linear head achieves PQ=0.477 on known classes. When it's confident (max prob > 0.7), it's almost certainly correct. We only need novel-class detection for masks where the linear head says "I don't know" (low confidence or high no-object score).

### How It Works

```python
def panoptic_inference_confidence_gated(pred_logits, pred_masks, backbone_feats,
                                         text_protos_novel, num_known=16,
                                         confidence_threshold=0.5):
    V, Q = pred_masks.shape
    masks_sig = pred_masks.sigmoid()

    linear_probs = pred_logits[:, :num_known].softmax(dim=-1)  # (Q, 16)
    max_known_conf, known_labels = linear_probs.max(dim=-1)  # (Q,)
    obj_score = 1 - pred_logits.softmax(dim=-1)[:, -1]

    # Per-voxel novel-class logits (only novel prototypes)
    feats_normed = F.normalize(backbone_feats.float(), dim=-1)
    novel_logits_voxel = feats_normed @ text_protos_novel.T / 0.1  # (V, N_novel)

    labels = torch.zeros(Q, dtype=torch.long)
    scores = torch.zeros(Q)

    for q in range(Q):
        if max_known_conf[q] > confidence_threshold:
            # Linear head is confident → trust it
            labels[q] = known_labels[q]
            scores[q] = max_known_conf[q] * obj_score[q]
        else:
            # Linear head uncertain → check novel classes via per-voxel vote
            mask_w = masks_sig[:, q]
            if mask_w.sum() < 50:
                continue
            novel_vote = (mask_w.unsqueeze(-1) * novel_logits_voxel).sum(0) / mask_w.sum()
            best_novel_score = novel_vote.max()
            best_novel_cls = novel_vote.argmax()

            # Compare novel vs known
            known_vote = (mask_w.unsqueeze(-1) * (feats_normed @ text_protos_known.T / 0.1)).sum(0) / mask_w.sum()
            best_known_score = known_vote.max()

            if best_novel_score > best_known_score:
                labels[q] = num_known + best_novel_cls
                scores[q] = best_novel_score.sigmoid() * obj_score[q]
            else:
                labels[q] = known_vote.argmax()
                scores[q] = best_known_score.sigmoid() * obj_score[q]

    return _greedy_assign(scores, labels, masks_sig, V, overlap_thr)
```

### Why This Is Practical

- Known-class PQ preserved nearly perfectly (falls back to linear head when confident)
- Novel detection only triggers for uncertain queries → fewer false positives
- No "pole catastrophe" — only uncertain masks get novel evaluation
- Simple threshold tuning

---

## Updated Decision Matrix (2026-05-28)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CONCLUSION: Inference-only open-vocab DOES NOT WORK with current backbone.    │
│ K/L/M evaluation confirms: bottleneck is feature margins, NOT aggregation.   │
│ Strategy L = Strategy E (identical results) → pooling is NOT the problem.    │
│ TRAINING IS REQUIRED for real open-vocabulary capability.                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│ CONFIRMED FAILURES (inference-only, all tested on 6019 samples):             │
│   ✗ Strategy E (geo ensemble, feature pooling) — FPs, 330K novel voxels     │
│   ✗ Strategy E v2 (multi-prompt + bias) — FPs confirmed visually (10 scenes)│
│   ✗ Strategy K (voxel vote + 1.2× margin) — 1.8M FPs, PQ -0.027           │
│   ✗ Strategy L (logit aggregation) — IDENTICAL to E (305K novel, same scenes)│
│   ✗ Strategy M (confidence gated) — 274K FPs, reduces volume not quality    │
│   ✗ No-bias ablation — DONE: pole catastrophe returns, L=E both ways        │
│                                                                                │
│   ROOT CAUSE: Backbone features lack margin between similar classes.         │
│   "car" voxels have cos_sim("police car") ≈ cos_sim("car") ± noise.        │
│   No inference strategy can create margin that doesn't exist in features.    │
│                                                                                │
│ NEXT STEPS (training required, single A100 ~72h):                            │
│   → Strategy I (voxel decomp loss + 200 diverse negatives)                  │
│     - Teaches backbone to push "car" features AWAY from "police car"        │
│     - Directly addresses root cause (no inter-class margin)                  │
│     - Can start from current best checkpoint                                 │
│                                                                                │
│   → Strategy C1 (token head + 200 negatives)                                │
│     - Fix the token head's 16-class collapse                                │
│     - Minimal code change, can run simultaneously                            │
│                                                                                │
│ MEDIUM-TERM (better features):                                                │
│   → Strategy H + I (unfreeze backbone + voxel decomp)                       │
│     - Backbone learns panoptic-aware boundaries                              │
│     - With diverse negatives: fine-grained CLIP alignment                    │
│                                                                                │
│ BEST LONG-TERM (bypass 3D backbone limitation):                              │
│   → Strategy D (camera CLIP features at inference)                           │
│     - Fresh ViT-L/14 features have full CLIP resolution                     │
│     - Can distinguish "police car" from "car" directly                       │
│     - No 3D distillation degradation                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Updated Recommended Execution Order (2026-05-28)

```
DONE: Strategy E / geo_v2 — FPs in 10 scenes (visual verification)
DONE: Strategy K/L/M evaluated on full trainval:
  - L (logit aggregation) = identical to E → pooling NOT the bottleneck
  - K (voxel vote) = 5.5× worse, PQ degraded
  - M (confidence gated) = reduces volume, same FPs
DONE: No-bias ablation (2026-05-29):
  - voxel_vote: PQ=0.3935, 18.6M novel (pole catastrophe — 10× explosion)
  - logit_agg: PQ=0.4642, 830K novel (pole returns)
  - confidence_gated: PQ=0.4634, 364K novel (slightly more pole)
  - Conclusion: bias prevents pole catastrophe, doesn't fix margin issue

INFERENCE-ONLY CHAPTER: CLOSED

NEXT (training path):
  Week 1: Strategy I implementation
    - Generate diverse text negative bank (200+ concepts including confusables)
    - Add voxel decomposition loss to training code
    - Train from best checkpoint (20 epochs, single A100)

  Week 2: Strategy C1 (token head + negatives)
    - Minimal change: replace 16-negative loss with 200+
    - Can train simultaneously with I

  Week 3+: Strategy H + I (unfreeze) or Strategy D (camera features)
```

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

2. **Point-level distillation ≠ object-level features** — DisNet was trained to match per-pixel OpenSeg features. Averaging many voxels might "blur" the semantic signal if different parts of an object have different feature directions.

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

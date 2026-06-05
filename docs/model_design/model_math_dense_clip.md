# Mathematical Formulation: Dense Camera-CLIP Supervision

**Companion to:** `model_arch_dense_clip.md`
**Notation conventions:** Bold lowercase = vectors, bold uppercase = matrices, calligraphic = sets.

---

## 1. Input Representation

Let a scene contain V voxels after voxelization at resolution δ = 0.05m.

- **Voxel coordinates:** `C ∈ ℤ^{V×3}` (integer grid coordinates)
- **Voxel features:** `X ∈ ℝ^{V×3}` (normalized XYZ in meters)
- **Sparse tensor:** `S = SparseTensor(X, C)` (MinkowskiEngine format)

---

## 2. Backbone Forward Pass

### 2.1 Frozen Backbone

The DisNet backbone f_θ (MinkUNet18A, θ frozen) maps sparse input to dense features:

```
f_θ: S → F_backbone ∈ ℝ^{V×768}
```

Internally: encoder (4 stages, stride 2 each) → decoder (4 stages, transposed conv) → final 1×1 conv (96→768).

### 2.2 LoRA Adaptation

After each target decoder block b ∈ {block5, block6, block7, block8, final}, a LoRA adapter g_φ_b modifies the features:

```
g_φ_b(h) = h + Dropout(W_up · ReLU(W_down · h))
```

where:
- h ∈ ℝ^{V×C_b} is the block output feature matrix (extracted via SparseTensor.F)
- W_down ∈ ℝ^{r×C_b}, W_up ∈ ℝ^{C_b×r} (rank r = 16)
- W_up initialized to zero → g_φ_b(h) = h at epoch 0 (exact identity, no warm-start disruption)
- Dropout(p=0.05) for regularization
- No LayerNorm (removed — it re-scaled features even at epoch 0)

The adapted backbone output:

```
F = f_{θ,φ}(S) ∈ ℝ^{V×768}
```

where φ = {W_down^b, W_up^b} for all adapter blocks. Total |φ| = 41,344 params.

---

## 3. Panoptic Decoder

### 3.1 Query Initialization

- Q = 106 queries: q_1, ..., q_Q ∈ ℝ^{256}
- 100 thing queries (learnable embeddings)
- 6 stuff queries (learnable embeddings, one per stuff class)

### 3.2 Cross-Attention Decoding (L = 3 blocks)

For each block l = 1, ..., L:

```
q^l = CrossAttn(q^{l-1}, F_down) + SelfAttn(q^{l-1})
```

where F_down is the backbone features projected to 256-d via a learned linear layer.

Output: decoder features D ∈ ℝ^{Q×256}

### 3.3 Mask Prediction

Binary mask logits via dot product:

```
M = F_proj · D^T ∈ ℝ^{V×Q}
```

where F_proj = Linear(F, 256→256). Each column M[:,q] is the per-voxel mask logit for query q.

### 3.4 Single CLIP Embedding Head (A4)

```
E = L2Norm(MLP(D)) ∈ ℝ^{Q×768}
```

MLP: Linear(256→512) → ReLU → Linear(512→512) → ReLU → Linear(512→768)

### 3.5 Objectness Head

```
o = σ(Linear(D)) ∈ ℝ^Q    (sigmoid output)
```

### 3.6 Classification (Inference)

Given text prototypes P ∈ ℝ^{K×768} (K classes, L2-normalized):

```
class(q) = argmax_k (E_q · P_k^T / τ)
```

where τ = exp(log_τ) is the learnable temperature.

---

## 4. Hungarian Matching

### 4.1 Cost Matrix (No CE — Single Head Setup)

For B batch items, compute assignment cost between Q queries and N_gt ground truth masks:

```
C_match(i, j) = λ_mask · C_BCE(M[:,i], GT_mask_j) + λ_dice · C_dice(M[:,i], GT_mask_j)
```

where λ_mask = λ_dice = 5.0.

Note: NO class-embedding cost in matching. The single CLIP head produces embeddings in a continuous space — distance between pred_embed and text_proto is not a reliable matching signal early in training.

### 4.2 Assignment

```
σ* = argmin_σ Σ_j C_match(σ(j), j)
```

Solved via scipy.optimize.linear_sum_assignment.

---

## 5. Loss Functions

### 5.1 Mask Losses (Standard)

**Binary cross-entropy:**
```
L_mask = (1/N_matched) Σ_{(i,j)∈σ*} BCE(σ(M[:,i]), GT_mask_j)
```

**Dice loss:**
```
L_dice = (1/N_matched) Σ_{(i,j)∈σ*} (1 - 2|M_i ∩ GT_j| / (|M_i| + |GT_j|))
```

### 5.2 Objectness Loss

```
L_obj = BCE(o, y_obj)
```

where y_obj_q = 1 if query q is matched, 0 otherwise.

### 5.3 L5/C1: Token Contrastive Loss with 216+ Negatives (v2 — Margin + Soft Negatives)

Let P_all ∈ ℝ^{(16+N_neg)×768} be the concatenation of 16 class prototypes (including 4 novel as soft negatives) and N_neg=200 negative concept embeddings. All L2-normalized.

For each matched query (i, j) where GT class c_j ∈ base_ids (12 training classes, excludes novel):

```
s_k = E_i · P_k                           (cosine similarity to k-th proto)
s'_k = s_k - m · 𝟙[k = c_j]              (subtract margin m from GT class)
L_C1 = (1/N_matched) Σ_{(i,j)∈σ*} -log(exp(τ · s'_{c_j}) / Σ_k exp(τ · s'_k))
```

where:
- τ = exp(log_τ), clamped to [1.0, 5.0] (prevents saturation)
- m = 0.1 (margin penalty — forces cos_sim(GT) - cos_sim(confuser) > m)
- k ranges over all 216 protos (12 base + 4 novel + 200 negatives)
- Novel protos (bus, pedestrian, manmade, vegetation) are NEVER targets, only denominators

**With multi-prompt (T4):** At each training step, P_{c_j} is randomly sampled from 10 template-generated embeddings for class c_j.

**Gradient effect of margin:** The model receives zero reward for achieving sim(GT) > sim(confuser). It only gets reward for sim(GT) - m > sim(confuser). This directly creates the cosine separation needed for open-vocab generalization.

### 5.4 L_O: Object-Level Camera CLIP Distillation

Let fv_clip ∈ ℝ^{V×768} be per-voxel camera CLIP features (from cached patch grid + bilinear projection).

For each matched query (i, j):

```
w_i = (σ(M[:,i]) / ||σ(M[:,i])||_1) · fv_clip    ∈ ℝ^{768}   (soft-pooled target)
w_i = L2Norm(w_i)

L_O = (1/N_covered) Σ_{(i,j): ||w_i||>ε} (1 - E_i · w_i)
```

where ε = 0.05 filters out queries with no camera coverage.

### 5.5 L_V: Voxel-Level Camera CLIP Distillation

Reconstruct per-voxel features from all queries:

```
F_rec = σ(M)^T · E ∈ ℝ^{V×768}
```

where σ(M) ∈ ℝ^{V×Q} is sigmoid of mask logits. Each voxel v gets:

```
F_rec[v] = Σ_q σ(M[v,q]) · E_q
```

Loss (L1, only on covered voxels):

```
L_V = (1/|covered|) Σ_{v: ||fv_clip[v]||>ε} |F_rec[v] - fv_clip[v]|
```

### 5.6 CASD Modification to L_V

For CASD classes C_casd = {11, 12, 13}, replace the L_V target:

```
fv_target[v] = {
    CLIP_CLS(crop(img, bbox(class_v) × scale_c))   if sem(v) ∈ C_casd
    fv_clip[v]                                       otherwise
}
```

where scale_c ∈ {3.0, 2.5, 2.0} for {other_flat, sidewalk, terrain}.

The CLS token of the context-expanded crop gives CLIP's holistic understanding of the stuff region in context, which is what CLIP text embeddings expect.

### 5.7 L7: Voxel Decomposition Loss (Root Cause Fix)

Let P_text ∈ ℝ^{(16+200)×768} be the full text bank (base + negatives), L2-normalized.

For valid voxels (sem ≠ ignore), subsample V' = min(V_valid, 10000):

```
sim = L2Norm(F[v]) · P_text^T / τ_7 ∈ ℝ^{V'×216}

L7 = (1/V') Σ_v CrossEntropy(sim[v], c_v)
```

where c_v ∈ {0, ..., 15} is the GT class (indexed into first 16 positions of P_text), and τ_7 = 0.07.

**Gradient flow:** L7 → F[v] → LoRA adapters φ. This is the ONLY loss that trains the backbone path.

---

## 6. Total Loss

### Phase 1 (epochs 1-2): Foundation

```
L_total = 5.0·L_mask + 5.0·L_dice + 2.0·L_obj + L_C1
```

Only LoRA + heads train. Decoder frozen.

### Phase 2 (epochs 3+): Full

```
L_total = 5.0·L_mask + 5.0·L_dice + 2.0·L_obj + L_C1
         + w_O(t)·L_O + w_V(t)·L_V + w_7(t)·L7
```

where warmup weights:
```
w_O(t) = min(1.0, (t - t_phase1) / 2)      ramp 0→1.0 over 2 epochs
w_V(t) = min(0.1, 0.1·(t - t_phase1) / 5)  ramp 0→0.1 over 5 epochs
w_7(t) = min(0.5, 0.5·(t - t_phase1) / 5)  ramp 0→0.5 over 5 epochs
```

---

## 7. Camera Feature Projection (Detailed)

### 7.1 LiDAR → Camera Transform Chain

For voxel at position p_lidar ∈ ℝ^3 in LiDAR sensor frame:

```
p_ego_L = R_lidar · p_lidar + t_lidar          (LiDAR sensor → ego at LiDAR time)
p_global = R_ego_L · p_ego_L + t_ego_L         (ego → global)
p_ego_C = R_ego_C^T · (p_global - t_ego_C)     (global → ego at camera time)
p_cam = R_cam^T · (p_ego_C - t_cam)            (ego → camera sensor)
```

### 7.2 Camera Projection

```
[u', v', w]^T = K · p_cam
u = u'/w,  v = v'/w,  depth = w
valid = (depth > 0.5) ∧ (0 ≤ u < W) ∧ (0 ≤ v < H)
```

### 7.3 Patch Grid Sampling

Patch grid G ∈ ℝ^{16×16×768} (cached, L2-normalized). For pixel (u, v):

```
patch_u = u / (W/16),  patch_v = v / (H/16)    (continuous patch coordinates)
feature = BilinearSample(G, patch_u, patch_v)   ∈ ℝ^{768}
```

### 7.4 Multi-Camera Averaging

For voxel v visible in cameras C_v ⊆ {1, ..., 6}:

```
fv_clip[v] = L2Norm((1/|C_v|) Σ_{c∈C_v} feature_c[v])
```

Zero vector if C_v = ∅ (no camera coverage).

---

## 8. Gradient Flow Summary

```
L_mask, L_dice, L_obj  →  Decoder params, SingleCLIPEmbeddingHead, LoRA adapters (via decoder→backbone_proj→F)
L_C1                    →  SingleCLIPEmbeddingHead, log_τ (token_criterion), LoRA adapters
L_O                     →  SingleCLIPEmbeddingHead (via E_i), Decoder (via M[:,i]), LoRA adapters
L_V                     →  SingleCLIPEmbeddingHead (via E_q), Decoder (via σ(M)), LoRA adapters
L7                      →  LoRA adapters φ DIRECTLY (via F[v], strongest per-voxel signal)
```

**Key insight (corrected from v1):** ALL losses train LoRA adapters because backbone features flow through backbone_proj into the decoder with `requires_grad=True`. L7 adds a direct per-voxel path that bypasses the decoder entirely — making it the strongest signal for backbone feature improvement. But LoRA is NOT exclusive to L7.

**Temperature note:** A single shared temperature (on `token_criterion`) is used for both training (C1 loss) and inference (panoptic_inference_clip). The decoder's `clip_embed_head.log_temperature` is vestigial and not used.

---

## 9. B12/N4 Training Setup

### Class Partitioning

- **B12 (base):** {barrier, bicycle, car, construction_vehicle, motorcycle, traffic_cone, trailer, truck, driveable_surface, other_flat, sidewalk, terrain}
- **N4 (novel):** {bus, pedestrian, vegetation, manmade}

### Training Implications

- GT masks for N4 classes are **excluded from all training losses** (mask, dice, token, L_O)
- No class labels, no mask supervision for novel classes during training
- L_V supervises ALL covered voxels regardless of class (camera features are class-agnostic) — this is the only signal that touches novel voxels
- Novel protos remain in C1 denominator as soft negatives (pushes embeddings away)

This ensures PQ_novel > 0 can only come from genuine CLIP-space generalization, not from mask supervision leakage.

### Inference on Novel Classes

```
P_eval = [P_base12 ; P_novel4] ∈ ℝ^{16×768}
class(q) = argmax_k (E_q · P_eval_k^T / τ)
```

Novel classes are classified purely by proximity in CLIP embedding space. The L_O and L_V losses ensure E_q is CLIP-aligned; L7 ensures backbone features have margin; L_C1 ensures the head respects CLIP geometry.

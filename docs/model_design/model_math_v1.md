# Mathematical Formulation: Base Panoptic Model (v1)

**Companion to:** `model_arch_v1.md`
**Context:** Closed-set 16-class panoptic segmentation with Mask2Former-style decoder on frozen DisNet backbone.

---

## 1. Input Representation

Let a scene contain V voxels after voxelization at resolution δ = 0.05m.

- **Raw LiDAR:** `P ∈ ℝ^{N×5}` (x, y, z, intensity, ring)
- **After range filter + voxelization:** `X ∈ ℝ^{V×3}` (XYZ in meters), `C ∈ ℤ^{V×3}` (grid coords)
- **Sparse tensor:** `S = SparseTensor(X, C)` (MinkowskiEngine format)

---

## 2. Backbone Forward Pass

The DisNet backbone f_θ (MinkUNet18A, θ frozen) maps sparse input to dense features:

```
f_θ: S → F ∈ ℝ^{V×768}
```

Architecture internally:
```
Encoder: 3 → 32 → 64 → 128 → 256 (stride 2 per stage, sparse conv)
Decoder: 256 → 128 → 128 → 96 (transposed sparse conv + skip connections)
Final:   96 → 768 (1×1 sparse conv)
```

All parameters θ are frozen (`requires_grad=False`). Forward pass under `torch.no_grad()`.

---

## 3. Panoptic Decoder

### 3.1 Projections

```
src          = LayerNorm(Linear(F, 768→256)) ∈ ℝ^{V×256}     (cross-attention K/V)
mask_feat    = LayerNorm(Linear(F, 768→256)) ∈ ℝ^{V×256}     (mask dot product)
pe           = SinusoidalPE(coords)           ∈ ℝ^{V×256}     (positional encoding)
mask_features = mask_feat + pe                ∈ ℝ^{V×256}
```

### 3.2 Sinusoidal Positional Encoding

For coordinate vector c = (x, y, z) normalized to [-1, 1]:
```
PE(c) = concat_{d∈{x,y,z}} concat_{k=1}^{42} [sin(2^{k·Δ} · πc_d), cos(2^{k·Δ} · πc_d)]
```
where Δ = log₂(2500)/42. Output: 3 × 42 × 2 = 252 features, zero-padded to 256.

### 3.3 Query Initialization

```
Q = 106 queries = 100 thing + 6 stuff
output       ∈ ℝ^{Q×256}  (learnable content embeddings)
query_embed  ∈ ℝ^{Q×256}  (learnable positional embeddings)
```

### 3.4 Decoder Layers (L = 9 layers, 3 blocks × 3 levels)

For each layer l = 0, ..., 8:
```
level = l mod 3
src_l = src + level_embed[level]          (virtual multi-scale)

// Cross-attention (masked)
q = LayerNorm(output) + query_embed
k = src_l + pe
v = src_l + pe
attn_mask = (σ(M_prev) < 0.5)            (attend only to predicted foreground)
output = output + Dropout(MultiHeadAttn(q, k, v, attn_mask))

// Self-attention
q = k = output + query_embed
v = output
output = LayerNorm(output + Dropout(MultiHeadAttn(q, k, v)))

// FFN
output = output + Dropout(Linear(1024→256, ReLU(Linear(256→1024, LayerNorm(output)))))
```

### 3.5 Prediction Heads (applied after each layer)

```
D = LayerNorm(output)                     ∈ ℝ^{Q×256}
logits = Linear(D, 256→17)               ∈ ℝ^{Q×17}     (16 classes + no-object)
mask_embed = MLP(D, 256→256→256)          ∈ ℝ^{Q×256}
M = mask_features · mask_embed^T          ∈ ℝ^{V×Q}      (mask logits)
```

---

## 4. Hungarian Matching

For B=1 batch, match Q=106 predictions to M ground truth segments.

### 4.1 Cost Matrix

```
C(i,j) = λ_cls · C_class(i,j) + λ_mask · C_mask(i,j) + λ_dice · C_dice(i,j)
```

where:
- `C_class(i,j) = -softmax(logits_i)[c_j]` (negative probability of GT class)
- `C_mask(i,j) = BCE(σ(M[:,i][S]), GT_j[S])` (S = random 75% subsample of V)
- `C_dice(i,j) = 1 - Dice(σ(M[:,i][S]), GT_j[S])`
- λ_cls = 2.0, λ_mask = 5.0, λ_dice = 5.0

### 4.2 Assignment

```
σ* = argmin_σ Σ_j C(σ(j), j)
```

Solved via `scipy.optimize.linear_sum_assignment(C)`.

---

## 5. Loss Functions

### 5.1 Classification Loss (Cross-Entropy)

```
target_i = c_j        if query i matched to GT j with class c_j
         = 16         if query i is unmatched (no-object)

L_CE = F.cross_entropy(logits, target, weight=[1,...,1, 0.1])
                                                 ↑16    ↑eos_coef
```

### 5.2 Mask BCE Loss

For each matched pair (i, j), subsample n = min(50000, V) voxels:
```
L_mask = (1/|matched|) Σ_{(i,j)∈σ*} mean_v[BCE(σ(M[v,i]), GT_j[v])]
```

### 5.3 Dice Loss

```
L_dice = (1/|matched|) Σ_{(i,j)∈σ*} (1 - (2·Σ_v σ(M[v,i])·GT_j[v] + 1) / (Σ_v σ(M[v,i]) + Σ_v GT_j[v] + 1))
```

### 5.4 Total Loss (with Auxiliary)

Applied to all 10 prediction layers (1 final + 9 auxiliary), each with independent matching:

```
L_total = Σ_{l=0}^{9} [2.0·L_CE^l + 5.0·L_mask^l + 5.0·L_dice^l]
```

= 30 individual loss terms.

---

## 6. Optimizer

```
AdamW(decoder.parameters(), lr=1e-4, weight_decay=0.01)
```

### Learning Rate Schedule (Cosine Warmup)

```
lr(t) = {
    t/t_warmup · lr_max                              if t < t_warmup
    lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·(t-t_warmup)/(t_total-t_warmup)))   otherwise
}
```

where t_warmup = 2 epochs × steps_per_epoch, lr_max = 1e-4, lr_min = 1e-6.

### Gradient Accumulation

Effective batch size = batch_size × grad_accum = 1 × 8 = 8.
Loss divided by grad_accum before backward. Optimizer steps every grad_accum forward passes.

### Mixed Precision (AMP)

```
with autocast():
    outputs = decoder(feats, coords)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(params, 0.5)
scaler.step(optimizer)
```

---

## 7. Panoptic Inference

From final layer predictions:

```
scores, labels = softmax(logits, dim=-1).max(-1)     (Q,)
valid = labels ≠ 16                                   (filter no-object)

For each query in score-descending order:
    mask = σ(M[:,q]) ≥ 0.5                           (binary mask)
    mask = mask & ~used                              (remove assigned voxels)
    if |mask| < 50: skip                             (min_points filter)

    pan[mask] = label × 1000 + instance_id
    used |= mask
```

---

## 8. Evaluation: Panoptic Quality

For each class c:
```
Match pred segments to GT segments where IoU > 0.5
(Segments with < 50 voxels ignored)

TP = matched pairs
FP = unmatched predictions
FN = unmatched GT

SQ(c) = mean(IoU of TP pairs)           (segmentation quality)
RQ(c) = 2·TP / (2·TP + FP + FN)        (recognition quality, F1)
PQ(c) = SQ(c) × RQ(c)

PQ = (1/16) Σ_c PQ(c)
```

---

## 9. Key Numbers

| Quantity | Value |
|---|---|
| V (voxels/scene) | ~20,000 |
| Q (queries) | 106 |
| M (GT segments/scene) | ~15-30 |
| Hidden dim | 256 |
| Backbone dim | 768 |
| Decoder layers | 9 |
| Loss terms | 30 (3 × 10 layers) |
| Trainable params | ~10M (decoder only) |
| Frozen params | 15.6M (backbone) |
| Best PQ | 0.473 (epoch 23, full trainval) |

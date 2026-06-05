# Mathematical Formulation: CLIP Token Head (Token v1)

**Companion to:** `model_arch_token_v1.md`
**Context:** Extends base model (PQ=0.473) with a CLIP token head for open-vocabulary classification. Dual-head: linear (closed-set) + token (CLIP-space).

---

## 1. Architecture Extension

The decoder produces output D ∈ ℝ^{Q×256} (same as base model). Three heads operate in parallel:

```
D = LayerNorm(output)  ∈ ℝ^{Q×256}

// Existing heads (from v1)
logits     = Linear(D, 256→17)            ∈ ℝ^{Q×17}      (closed-set classification)
mask_embed = MLP(D, 256→256→256)          ∈ ℝ^{Q×256}
M          = mask_features · mask_embed^T  ∈ ℝ^{V×Q}       (mask logits)

// New token head
E = L2Norm(MLP_token(D))                   ∈ ℝ^{Q×768}     (CLIP embeddings)
```

where `MLP_token`: Linear(256→512) → ReLU → Linear(512→768).

---

## 2. Text Prototypes

Pre-computed (frozen during training):

```
P ∈ ℝ^{K×768}     (K text prototypes, L2-normalized)
P_k = L2Norm(CLIP_text("a {class_k} in a scene"))
```

For training: K = 16 (nuScenes panoptic classes)
For open-vocab inference: K = 22 (16 known + 6 novel concepts)

---

## 3. Token Loss Functions

### 3.1 Cosine Similarity Loss (v1 default)

For each matched pair (i, j) from Hungarian matching:

```
L_cos = (1/|matched|) Σ_{(i,j)∈σ*} (1 - E_i · P_{c_j})
```

where c_j is the GT class of segment j, and both E_i and P_{c_j} are L2-normalized (so dot product = cosine similarity).

**Range:** [0, 2], typically converges to ~0.02.
**Gradient:** ∂L/∂E_i = -P_{c_j} / |matched| (pulls embedding toward GT prototype)

### 3.2 Contrastive Loss / InfoNCE (v2)

```
L_NCE = (1/|matched|) Σ_{(i,j)∈σ*} -log(exp(E_i · P_{c_j} / τ) / Σ_{k=1}^{K} exp(E_i · P_k / τ))
```

where τ = exp(log_τ) is a learnable temperature parameter.

**Key difference from cosine loss:**
- Cosine: only pulls toward GT prototype
- InfoNCE: pulls toward GT AND pushes away from all 15 other prototypes
- Magnitude: InfoNCE ~ 2-3 (for 16 classes) vs cosine ~ 0.02

**Temperature:**
- Initialized at τ = 0.07 (CLIP default, log_τ = log(1/0.07) ≈ 2.66)
- Clamped: max=100 for stability
- Higher τ → sharper softmax → stronger gradients on hard examples

### 3.3 Auxiliary Token Loss

Applied to all L=9 intermediate decoder layers (not just final):

```
L_token_total = L_token^{final} + Σ_{l=0}^{8} L_token^{l}
```

Each auxiliary layer l has:
- Its own prediction: E^l = L2Norm(MLP_token(LayerNorm(output^l)))
- Its own matching σ*^l (same matching used for panoptic aux losses)

---

## 4. Total Loss

```
L_total = L_panoptic + w(t) · L_token_total

where:
  L_panoptic = Σ_{l=0}^{9} [2.0·L_CE^l + 5.0·L_mask^l + 5.0·L_dice^l]  (same as base)
  w(t) = min(1, t/t_warmup) · w_max                                        (linear warmup)
```

| Config | w_max | t_warmup | Loss type |
|---|---|---|---|
| v1 (cosine) | 0.5 | 100 steps | cosine |
| v2 (contrastive) | 1.0 | 500 steps | InfoNCE |

---

## 5. Optimizer (Differential LR — v2)

Two parameter groups:

```
Group 1 (decoder): lr = 1e-5, weight_decay = 0.01
  - All existing decoder parameters (backbone_proj, layers, class_embed, mask_embed, queries)

Group 2 (token_head): lr = 5e-5, weight_decay = 0.01
  - MLP_token parameters (256→512, 512→768)
  - log_temperature parameter
```

**Rationale:** Lower decoder LR preserves mask quality (PQ=0.473). Higher token head LR allows fresh parameters to converge quickly.

---

## 6. Inference Modes

### Mode A: Linear (Closed-Set Baseline)

```
class(q) = argmax_k softmax(logits_q)[k]     k ∈ {0,...,15}
score(q) = max_k softmax(logits_q)[k]
```

### Mode B: Token-16 (CLIP on Known Classes)

```
sim_q = E_q · P^T ∈ ℝ^{16}                   (cosine similarity to 16 protos)
class(q) = argmax_k sim_q[k]
score(q) = max_k sim_q[k]
filter: score > 0.1                           (no explicit no-object class)
```

### Mode C: Ensemble

```
s_lin = softmax(logits_q)[:16]                (linear scores, 16-dim)
s_tok = (sim_q - min) / (max - min)           (normalized token scores)
score = α · s_lin + (1-α) · s_tok
class(q) = argmax score
```

### Mode D: Token-OpenVocab (Primary Open-Vocab Test)

```
P_ext ∈ ℝ^{22×768}                           (16 known + 6 novel prototypes)
sim_q = E_q · P_ext^T ∈ ℝ^{22}
class(q) = argmax_k sim_q[k]                  k ∈ {0,...,21}
```

Novel classes detected when argmax falls in positions 16-21.

### Mode E: Backbone Pooling (Lower Bound)

```
For predicted mask m_q:
  f_pool = L2Norm(mean(F[v] for v where σ(M[v,q]) > 0.5))  ∈ ℝ^{768}
  class(q) = argmax_k (f_pool · P_ext_k)
```

Uses frozen backbone features directly — no learned token head needed.

---

## 7. Gradient Flow

```
L_panoptic → decoder params, class_embed, mask_embed  (same as base)
L_token    → MLP_token, log_τ, decoder params (via D = LayerNorm(output))
```

With auxiliary token loss, every decoder layer receives token gradients at every forward pass:
```
Layer l: L_token^l → output^l → CrossAttn^l, SelfAttn^l, FFN^l
```

Without auxiliary: only final layer output receives token gradients directly.

---

## 8. Key Results

| Metric | Base (linear only) | Token v1 | Token v2 |
|---|---|---|---|
| PQ (linear) | 0.473 | 0.429 | **0.476** |
| PQ (token) | — | 0.410 | **0.468** |
| SQ | 0.793 | 0.772 | **0.794** |
| mIoU (linear) | 0.433 | 0.405 | **0.431** |
| Novel detections | 0 | 0 | 0 |

**Conclusion:** Token v2 achieves near-parity with linear head (0.468 vs 0.476 PQ) while operating in CLIP embedding space. However, novel detection = 0 indicates the InfoNCE loss collapsed the token head to a 16-class classifier rather than learning generalizable CLIP alignment. This motivates the dense CLIP supervision approach (Job 1a/1b/1c).

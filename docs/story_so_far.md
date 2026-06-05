# The Story So Far: Open-Vocabulary 3D Panoptic Segmentation

## Abstract

This document traces the development of an open-vocabulary 3D panoptic segmentation system built on nuScenes LiDAR data. Beginning from a closed-set Mask2Former-style decoder achieving PQ=0.476, we explored inference-time open-vocabulary strategies (geometric ensemble, per-voxel voting, logit aggregation), proved their fundamental limitations through a novel three-stage voxel margin diagnostic, and pivoted to a training-based pipeline (Jobs 1a/1b/1c) that addresses the root cause: cosine geometry collapse in the CLIP-distilled backbone.

---

## 1. Phase 1 — Closed-Set Panoptic Baseline (v1)

### 1.1 Architecture

The system builds on a frozen DisNet backbone (MinkUNet18A, 15.6M parameters) that was pre-trained via knowledge distillation from OpenSeg, which itself distills from CLIP ViT-L/14@336px. The backbone produces 768-dimensional features per voxel that nominally reside in CLIP's embedding space.

A trainable Mask2Former-style decoder (~10M parameters) processes these features:

$$
\mathbf{F}_{\text{src}} = \text{BackboneProj}(\mathbf{F}_{\text{backbone}}) \in \mathbb{R}^{V \times 256}
$$

$$
\mathbf{Q} = [\mathbf{q}_1^{\text{thing}}, \ldots, \mathbf{q}_{100}^{\text{thing}}, \mathbf{q}_1^{\text{stuff}}, \ldots, \mathbf{q}_6^{\text{stuff}}] \in \mathbb{R}^{106 \times 256}
$$

Nine masked cross-attention layers refine queries, producing per-query class logits and per-voxel mask predictions via dot-product:

$$
\hat{y}_{\text{class}} = \text{Linear}_{256 \to 17}(\mathbf{q}_i) \quad \text{(16 classes + no-object)}
$$

$$
\hat{m}_i = \sigma(\mathbf{F}_{\text{mask}} \cdot \mathbf{q}_i^{\text{mask}}) \in [0,1]^V
$$

### 1.2 Training Losses

Hungarian matching assigns predictions to ground truth:

$$
\mathcal{C}_{\text{match}} = 2.0 \cdot \mathcal{C}_{\text{CE}} + 5.0 \cdot \mathcal{C}_{\text{BCE}} + 5.0 \cdot \mathcal{C}_{\text{Dice}}
$$

Supervision losses on matched pairs:

$$
\mathcal{L} = 2.0 \cdot \mathcal{L}_{\text{CE}} + 5.0 \cdot \mathcal{L}_{\text{mask}} + 5.0 \cdot \mathcal{L}_{\text{dice}}
$$

Applied to all 10 prediction layers (final + 9 auxiliary).

### 1.3 Result

**PQ = 0.473** on nuScenes val (full trainval training, 30 epochs, A100). Strong closed-set performance: car PQ=0.845, pedestrian PQ=0.647. Weak on rare/ambiguous classes: other_flat PQ=0.082.

> **Full mathematical formulation:** See `docs/model_design/model_math_v1.md`

---

## 2. Phase 2 — CLIP Token Head Extension (token v1/v2)

### 2.1 Motivation

The closed-set decoder cannot recognize classes beyond its 16 training labels. Since the backbone is CLIP-distilled, we hypothesised that projecting decoder queries back into CLIP's 768-d space would enable text-prototype-based classification for arbitrary classes.

### 2.2 Architecture

A CLIPTokenHead MLP is attached to the decoder:

$$
\mathbf{e}_i = \frac{\text{MLP}_{256 \to 512 \to 768}(\mathbf{q}_i)}{\|\text{MLP}(\mathbf{q}_i)\|_2} \in \mathbb{R}^{768}
$$

Five inference modes were tested:
- **Mode A (Linear):** Standard CE argmax — baseline
- **Mode B (Token-16):** Classify via $\arg\max_c \; \mathbf{e}_i \cdot \mathbf{p}_c$ where $\mathbf{p}_c$ are 16 text prototypes
- **Mode C (Ensemble):** Geometric mean of linear and token probabilities
- **Mode D (Open-Vocab):** Token head with expanded prototype set (known + novel)
- **Mode E (Backbone-Pool):** Pool raw backbone features through masks

### 2.3 Token Loss (v2: InfoNCE)

Initial cosine loss (v1) was too weak (~0.02 magnitude vs ~34 panoptic loss). Replaced with contrastive InfoNCE:

$$
\mathcal{L}_{\text{token}} = -\log \frac{\exp(\mathbf{e}_i \cdot \mathbf{p}_{y_i} / \tau)}{\sum_{c=1}^{16} \exp(\mathbf{e}_i \cdot \mathbf{p}_c / \tau)}
$$

With learnable temperature $\tau$ (init $1/0.07 = 14.3$) and linear warmup.

### 2.4 Results

| Mode | PQ | Novel PQ |
|------|-----|----------|
| Linear (CE) | 0.476 | — |
| Token-16 | 0.468 | 0.000 |
| Backbone-Pool | 0.209 | — |

**Critical finding:** Token head PQ competitive with CE (within 2%), but novel detection = **zero**. The InfoNCE loss with 16 prototypes collapsed the 768-d space to a 16-class subspace. The head learned a good 16-way classifier, not generalised CLIP geometry.

> **Full mathematical formulation:** See `docs/model_design/model_math_token_v1.md`

---

## 3. Phase 3 — Inference-Only Open-Vocabulary Strategies

### 3.1 Strategy E: Geometric Ensemble

Pool backbone features through predicted masks, classify against expanded prototype set:

$$
\mathbf{f}_i^{\text{pool}} = \frac{\sum_v \sigma(\hat{m}_{i,v}) \cdot \bar{\mathbf{f}}_v}{\sum_v \sigma(\hat{m}_{i,v})} \quad \text{where } \bar{\mathbf{f}}_v = \frac{\mathbf{f}_v}{\|\mathbf{f}_v\|}
$$

$$
P_{\text{clip}}(c|i) = \text{softmax}\left(\frac{\mathbf{f}_i^{\text{pool}} \cdot \mathbf{p}_c}{\tau}\right)
$$

Ensemble for known classes: $\; s_c = P_{\text{linear}}(c)^\alpha \cdot P_{\text{clip}}(c)^{1-\alpha}$

Novel classes: $\; s_c = P_{\text{clip}}(c)$ only.

**Result:** PQ=0.468 at $\alpha=0.5$. Visual inspection revealed **all novel detections were false positives** — "ambulance" was white trucks, "police car" was dark sedans.

### 3.2 Strategy K: Per-Voxel Majority Vote

Hypothesis: pooling features before classification destroys discriminability ($\mathbb{E}[f(x)] \neq f(\mathbb{E}[x])$). Instead, classify each voxel independently, then aggregate:

$$
\mathbf{L}_i = \frac{\sum_v \sigma(\hat{m}_{i,v}) \cdot (\bar{\mathbf{f}}_v \cdot \mathbf{P}^T / \tau)}{\sum_v \sigma(\hat{m}_{i,v})} \in \mathbb{R}^{C}
$$

Novel override: assign novel class only if $P_{\text{novel}}^{\max} > 1.2 \cdot P_{\text{known}}^{\max}$.

**Result:** PQ=0.449, 1.8M novel voxels (massive over-detection, all false positives).

### 3.3 Strategy L: Soft Logit Aggregation

Same per-voxel classification as K, but with geometric ensemble instead of margin threshold. Mathematically equivalent to testing whether the aggregation order matters:

$$
\text{Strategy E: } \; f(\text{avg}(\mathbf{x}_v)) \quad \text{vs} \quad \text{Strategy L: } \; \text{avg}(f(\mathbf{x}_v))
$$

**Result:** PQ=0.470, 305K novel voxels — **virtually identical to Strategy E**.

### 3.4 The Critical Finding

**Strategy L ≈ Strategy E** proves that pooling degradation is NOT the bottleneck. The features within a correctly-predicted mask are sufficiently homogeneous that classify-then-average ≈ average-then-classify. The problem is the features themselves.

**Inference-only chapter: CLOSED.**

> **Full strategy details:** See `docs/model_design/model_openvocab_strategyE.md` and `model_openvocab_strategyK_L_M.md`

---

## 4. Phase 4 — Three-Stage Voxel Margin Diagnostic

### 4.1 Motivation

All inference strategies failed despite the backbone being nominally "CLIP-aligned." We designed a diagnostic to precisely locate where in the processing chain the CLIP geometry degrades.

### 4.2 Methodology

For each ground-truth instance with class $y$, measure the cosine margin:

$$
\Delta_y = \cos(\mathbf{e}, \mathbf{p}_y) - \max_{c \neq y} \cos(\mathbf{e}, \mathbf{p}_c)
$$

at three stages:

- **Stage 1 (Raw Backbone):** $\mathbf{e} = \bar{\mathbf{f}}_v$ for each voxel with GT label
- **Stage 2 (Mask-Pooled Backbone):** $\mathbf{e} = \text{norm}(\sum_v w_v \mathbf{f}_v)$ pooled by decoder's predicted mask
- **Stage 3 (Token Head):** $\mathbf{e} = \text{TokenHead}(\mathbf{q}_i)$ from the learned projection

### 4.3 Results (on `checkpoints_token_full_v2/best.pth`)

| Stage | Mean Margin | Correct (>0) |
|-------|-------------|-------------|
| 1. Raw backbone (per-voxel) | **−0.021** | 27.9% |
| 2. Mask-pooled backbone | **−0.019** | 35.4% |
| 3. Token head output | **+0.225** | 82.3% |

### 4.4 Interpretation

1. **The OpenSeg → DisNet distillation destroyed fine-grained CLIP geometry.** Mean margin is negative — the backbone cannot reliably distinguish classes via prototype matching.

2. **Spatial pooling is neutral** (+0.002 gain). The mask doesn't help or hurt — the underlying features are the bottleneck.

3. **A learned projection recovers strong separation.** The token head proves the decoder queries encode sufficient semantic information — a 3-layer MLP can map it to CLIP space with +0.22 margin.

4. **The chain of accountability is clear:** distillation broke geometry (S1), pooling cannot fix it (S2), but a trained head can bypass it (S3).

### 4.5 Root Cause: Distillation Chain Degradation

The 4-step pipeline CLIP → OpenSeg → multi-view projection → DisNet applies successive compressions:

- OpenSeg regression treats all angular directions equally — inter-class gaps are not preserved
- Multi-view averaging into 3D blurs view-specific discriminative cues
- MinkUNet capacity limitations smooth out fine structure in feature space
- Result: cosine margin between classes (e.g., car vs truck: +0.01) approaches noise level

This is why no inference strategy can succeed: the discriminative signal does not exist at the backbone level for similar classes.

---

## 5. Phase 5 — Training-Based Open-Vocabulary Pipeline (Job 1a/1b/1c)

### 5.1 Design Principle

Open-vocabulary capability cannot be achieved at inference time when backbone margins are negative. Training-time alignment is required, but full model retraining is impractical. Solution: a staged fine-tuning pipeline, each job targeting a specific stage of the margin diagnostic.

### 5.2 Class Split: B12/N4

| Role | Classes | Training supervision |
|------|---------|---------------------|
| Base (B12) | barrier, bicycle, car, construction vehicle, motorcycle, traffic cone, trailer, truck, driveable surface, other flat, sidewalk, terrain | Full (mask + dice + token) |
| Novel (N4) | bus, pedestrian, manmade, vegetation | **None** (excluded from ALL losses) |

Novel classes appear in the data but receive zero supervision. Their mask targets are filtered via `NOVEL_IDS_TRAIN = {2, 6, 14, 15}`. They exist in the token loss denominator as soft negatives only.

### 5.3 Job 1a: Token Head Alignment (Targets Stage 3)

**Architecture change:** Replace dual CE + token head with a single `SingleCLIPEmbeddingHead`:

$$
\mathbf{e}_i = \frac{\text{MLP}_{256 \to 512 \to 512 \to 768}(\mathbf{q}_i)}{\|\text{MLP}(\mathbf{q}_i)\|_2}
$$

Plus objectness head: $o_i = \text{Linear}_{256 \to 1}(\mathbf{q}_i)$

**Loss: InfoNCE with 216-class denominator and margin penalty:**

$$
\mathcal{L}_{\text{C1}} = -\log \frac{\exp((\mathbf{e}_i \cdot \mathbf{p}_{y_i} - m) / \tau)}{\sum_{c=1}^{216} \exp(\mathbf{e}_i \cdot \mathbf{p}_c / \tau)}
$$

where:
- $m = 0.1$ (margin penalty, applied to GT class only)
- $\tau$ is learnable (init 3.0, clamped [1.0, 5.0], applied as division)
- Denominator includes 12 base + 4 novel (soft negatives) + 200 hard negatives (driving-domain confusers)
- Only base classes (B12) used as targets; novel positions never receive positive gradient

**LoRA adaptation:** Rank-16 adapters (~41K params) on backbone decoder blocks:

$$
\mathbf{h}' = \mathbf{h} + \frac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{h}, \quad \mathbf{A} \in \mathbb{R}^{r \times d}, \; \mathbf{B} \in \mathbb{R}^{d \times r}
$$

**Warm-start:** From closed-set best (PQ=0.476). Two-phase training: Phase 1 (2 epochs, freeze decoder, train only LoRA + new heads) → Phase 2 (18 epochs, all trainable, differential LR).

**Bug fixes from v1 → current:**
- Temperature: `sim * τ` (temperature escapes to ceiling) → `sim / τ` (temperature descends toward sharpness)
- Temperature init: log(1/0.07) = 2.66 → exp = 14.3 (above clamp, zero gradient) → log(3.0) = 1.1 (inside clamp)
- Novel class leakage: novel had full mask/dice supervision → filtered from ALL losses
- LoRA grad norm: read before AMP unscale → moved after unscale (true norm ~0.2, not 914K)

**Expected outcome:** Stage 3 margin > +0.10 for base classes. Novel generalization through broad 200-negative bank preventing subspace collapse.

### 5.4 Job 1b: Backbone Feature Alignment (Targets Stage 1)

**Data:** Precomputed dense camera CLIP features using ViT-L/14@224px with sliding window (stride=112, 50% overlap), stored as fp16. Total: 186GB for full trainval.

**Object-level distillation (L_O):**

$$
\mathcal{L}_O = 1 - \cos\left(\mathbf{e}_i^{\text{pool}}, \; \mathbf{c}_i^{\text{CLIP}}\right)
$$

where $\mathbf{e}_i^{\text{pool}}$ is mask-pooled backbone features and $\mathbf{c}_i^{\text{CLIP}}$ is the corresponding camera CLIP feature.

**Voxel-level distillation (L_V):**

$$
\mathcal{L}_V = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \left(1 - \cos(\hat{\mathbf{f}}_v, \mathbf{c}_v^{\text{CLIP}})\right)
$$

where $\hat{\mathbf{f}}_v = \sigma(\hat{m}_v)^T \mathbf{E}$ is the reconstructed per-voxel feature from mask predictions and query embeddings.

**Key property:** Both losses are class-agnostic — they supervise ALL voxels including novel classes. CLIP doesn't know class labels; it just provides a visual feature target. This is what allows Stage 1 margins to improve for novel classes without explicit novel supervision.

**Warm-start:** From Job 1a best checkpoint.

### 5.5 Job 1c: Novel Class Mask Bootstrapping (Targets Novel SQ)

**Problem:** Job 1a excludes novel classes from mask/dice losses → model never learns novel class mask shapes. Job 1b aligns features but doesn't produce novel-specific masks.

**Approach (planned):** Class-Agnostic Self-Distillation (CASD):
1. Run inference on training data using Job 1b checkpoint
2. High-confidence novel class predictions (via CLIP prototype matching) become pseudo-labels
3. Retrain with pseudo-labels included for novel class mask supervision
4. Iterate if needed

**Condition:** Only proceed if after Job 1b, novel classes show good voxel margins (>+0.05) but poor mask quality (low SQ).

> **Full mathematical formulation:** See `docs/model_design/model_math_dense_clip.md`

---

## 6. Summary: The Chain of Evidence

```
Phase 1: Closed-set baseline established (PQ=0.473)
    │
Phase 2: Token head added — competitive but novel=0 (16-class collapse)
    │
Phase 3: Inference strategies (E, K, L, M) — all produce false positives
    │     Key finding: Strategy L ≈ Strategy E → pooling is NOT the issue
    │
Phase 4: Voxel margin diagnostic reveals root cause:
    │     Backbone margin = -0.02 (distillation destroyed CLIP geometry)
    │     Token head margin = +0.22 (learned projection WORKS when supervised)
    │
Phase 5: Training-based pipeline to fix margins at source:
          Job 1a → Fix Stage 3 (token head, B12/N4, 216 negatives)
          Job 1b → Fix Stage 1 (backbone, dense CLIP distillation, class-agnostic)
          Job 1c → Fix novel masks (self-distillation, conditional)
```

### Progression Logic

| Transition | Evidence | Decision |
|-----------|----------|----------|
| v1 → token head | Backbone is CLIP-aligned, decoder should project back | Add CLIPTokenHead (auxiliary loss) |
| Token head → Strategy E | Token novel=0, but backbone has CLIP features | Try inference-only: pool + ensemble |
| Strategy E → K/L/M | Pool might destroy signal; per-voxel may preserve | Test aggregation order hypothesis |
| K/L/M → Diagnostic | L≈E disproves pooling hypothesis; ALL strategies fail | Need to measure WHERE signal is lost |
| Diagnostic → Job pipeline | S1=-0.02 (backbone broken), S3=+0.22 (head works when trained) | Must fix backbone via training |
| Job 1a → 1b | Token head alignment alone won't fix backbone (S1 stays negative) | Need dense voxel-level CLIP supervision |
| Job 1b → 1c | Features aligned but novel masks never trained | Bootstrap via self-distillation |

---

## 7. Current Status

| Component | Status | Key Metric |
|-----------|--------|-----------|
| Closed-set baseline | **Complete** | PQ = 0.476 |
| Token head (all-16-class) | **Complete** | Margin = +0.225 (proves architecture) |
| Inference strategies (E/K/L/M) | **Complete, abandoned** | All produce false positives |
| Voxel margin diagnostic | **Complete** | Backbone = -0.021, head = +0.225 |
| Precompute CLIP features | **Complete** | 186GB, stride=112, fp16 |
| Job 1a (B12/N4 token training) | **Running** | Temperature descending correctly (3.0→2.91) |
| Job 1b (dense voxel distillation) | **Ready** | Code tested, mini smoke passed |
| Job 1c (self-distillation) | **Planned** | Conditional on Job 1b results |

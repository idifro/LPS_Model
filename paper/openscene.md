# OpenScene: 3D Scene Understanding with Open Vocabularies

**Authors:** Songyou Peng, Kyle Genova, Chiyu "Max" Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser  
**Affiliations:** Google Research, ETH Zurich, MPI Tübingen, Waymo LLC, Simon Fraser University  
**Published:** CVPR 2023 (arXiv: 2211.15654)

---

## 1. Problem Statement

Traditional 3D scene understanding systems are trained with supervision on closed label sets (e.g., 20 ScanNet classes). This means they can only answer pre-defined queries like "is this a chair or a table?" and fail on:

- **Rare/long-tail objects** with few training examples
- **Non-categorical queries** — materials ("metal"), affordances ("sit"), activities ("cook"), room types ("kitchen"), physical properties ("soft")

The goal of OpenScene is to enable **arbitrary, open-vocabulary text queries** on 3D point clouds **without any labeled 3D data**.

---

## 2. Core Idea

Co-embed every 3D surface point into the **CLIP feature space**, so that 3D points, image pixels, and text strings all live in the same embedding space. Once this alignment is achieved, any text query can be answered by computing cosine similarity between the query's CLIP text embedding and each point's feature vector.

---

## 3. Method (Three-Stage Pipeline)

### 3.1 Multi-View Image Feature Fusion (f²ᴰ)

1. **Extract per-pixel CLIP-aligned features** from each posed RGB image using a frozen 2D segmentation model (OpenSeg or LSeg). For image $i$, this produces $\mathbf{I}_i \in \mathbb{R}^{H \times W \times C}$.

2. **2D–3D pairing:** For each 3D point $\mathbf{p}$, project it into every camera using the pinhole model $\tilde{\mathbf{u}} = \mathbf{I}_i \cdot \mathbf{E}_i \cdot \tilde{\mathbf{p}}$. An occlusion test (using depth maps when available) ensures only visible correspondences are kept.

3. **Average pooling** across $K$ visible views:
$$\mathbf{f}^{2D} = \phi(\mathbf{f}_1, \dots, \mathbf{f}_K) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{f}_k$$

This produces a feature point cloud $\mathbf{F}^{2D} \in \mathbb{R}^{M \times C}$.

### 3.2 3D Distillation (f³ᴰ)

The fused 2D features can be noisy (inconsistent 2D predictions) and require images at test time. To address this, a **MinkowskiNet18A** (sparse 3D CNN) is trained to predict per-point features from **geometry alone**:

$$\mathbf{F}^{3D} = \mathcal{E}^{3D}(\mathbf{P}), \quad \mathcal{E}^{3D}: \mathbb{R}^{M \times 3} \rightarrow \mathbb{R}^{M \times C}$$

**Loss:** Cosine similarity loss between distilled and fused features:
$$\mathcal{L} = 1 - \cos(\mathbf{F}^{2D}, \mathbf{F}^{3D})$$

Since OpenSeg/LSeg features are already aligned with CLIP, the distilled 3D model inherits CLIP alignment — enabling **3D-only open-vocabulary understanding** even without images at inference.

### 3.3 2D–3D Feature Ensemble (f²ᴰ³ᴰ)

The 2D and 3D features have complementary strengths:
- **2D features** excel at small objects and geometrically ambiguous items (e.g., paintings on walls)
- **3D features** excel at objects with distinctive geometry (e.g., walls, floors)

**Ensemble procedure:**
1. Compute CLIP text embeddings $\mathbf{T} = \{\mathbf{t}_1, \dots, \mathbf{t}_N\}$ for $N$ query prompts.
2. For each point, compute max cosine similarity across all prompts for both feature types:
   - $s^{2D} = \max_n \cos(\mathbf{f}^{2D}, \mathbf{t}_n)$
   - $s^{3D} = \max_n \cos(\mathbf{f}^{3D}, \mathbf{t}_n)$
3. Select whichever feature (2D or 3D) has the higher max similarity as the final feature $\mathbf{f}^{2D3D}$.

### 3.4 Inference

For any task, simply compute:
$$\text{label} = \arg\max_n \cos(\mathbf{f}^{2D3D}, \mathbf{t}_n)$$

This works for semantic segmentation, object search, scene exploration, etc. — all zero-shot.

---

## 4. Key Experimental Results

### 4.1 Zero-Shot 3D Semantic Segmentation

**ScanNet 4 unseen classes** (comparison with 3DGenZ which trains on 16 seen classes):

| Method | mIoU | mAcc |
|--------|------|------|
| 3DGenZ | 7.7 | 23.8 |
| OpenScene (LSeg) | **62.8** | **81.0** |

OpenScene massively outperforms despite using **zero** labeled 3D data.

### 4.2 Standard Benchmarks (Full Label Sets)

| Dataset | SOTA Supervised mIoU | OpenScene mIoU |
|---------|---------------------|----------------|
| nuScenes | 82.0 | 42.1 |
| ScanNet (20) | 73.6 | 54.2 |
| Matterport3D (21) | 54.2 | 43.4 |

Gap is smallest on Matterport3D, likely because its diversity makes supervised training harder.

### 4.3 Scaling to Many Classes (Key Result)

On Matterport3D, as the number of classes increases:

| K classes | Supervised mAcc | OpenScene mAcc |
|-----------|----------------|----------------|
| 21 | **64.5** | 59.2 |
| 40 | 50.8 | **50.9** |
| 80 | 33.4 | **34.6** |
| 160 | 18.4 | **23.1** |

**OpenScene overtakes the fully-supervised approach at ≥40 classes.** The supervised model collapses on tail classes (only 1.7% mAcc on the rarest 20 classes), while OpenScene maintains ~16% because it doesn't depend on labeled 3D examples.

### 4.4 Ablation: 2D vs 3D vs Ensemble

The ensemble consistently outperforms either alone. ~70% of points use 3D features, but the fraction using 2D features increases with more classes (28.6% at K=21 → 32.5% at K=160), confirming 2D features matter more for fine-grained/long-tail categories.

### 4.5 OpenSeg vs LSeg

OpenSeg generally produces better features than LSeg across datasets and metrics. OpenSeg is used as default.

---

## 5. Applications (All Zero-Shot)

### 5.1 Open-Vocabulary 3D Object Search
Query the entire Matterport3D test set (18 buildings, 406 regions) with arbitrary text. Results for specific queries like "teddy bear," "globe," "yellow egg-shaped vase" achieve near-perfect precision. The system even discovers 26 instances that were **missing from the ground truth annotations**.

### 5.2 Image-Based 3D Object Detection
Encode a query image with CLIP's image encoder, compute cosine similarity with 3D point features, and threshold. Successfully distinguishes "pool table" from "dining table" despite both being tables.

### 5.3 Open-Vocabulary Scene Exploration
Query scenes for:
- **Properties:** "comfy" → beds, couches; "hot" → ovens, fireplaces
- **Materials:** "fabric," "wood," "metal," "glass"
- **Activities:** "sleep," "cook," "wash," "dine," "play"
- **Room types:** "kitchen," "bathroom," "bedroom"
- **Abstract concepts:** "cluttered," "outdoor," "open"

This is arguably the paper's most compelling contribution — a single model handles all these diverse queries without any task-specific training.

---

## 6. Implementation Details

- **3D backbone:** MinkowskiNet18A
- **Voxel size:** 2cm (indoor), 5cm (outdoor/nuScenes)
- **Optimizer:** Adam, lr=1e-4, 100 epochs
- **Supervision sampling:** 20K points per iteration (memory constraint)
- **Hardware:** 1× A100 (indoor), 4× A100 (nuScenes)
- **Training time:** ~24 hours; **Inference:** ~0.1 seconds
- **Prompt engineering:** Wrapping class names as "a XX in a scene" gives +2.3 mIoU boost
- **Multi-view fusion:** Average pooling outperforms random selection and median selection
- **Occlusion test threshold:** Proportional to depth — 0.2D for ScanNet (noisy depth), 0.02D for Matterport3D

---

## 7. Limitations

1. **Test-time image fusion** could be improved — early fusion was tried but didn't help much.
2. **Only qualitative evaluation** for non-segmentation tasks (affordance, materials, activities) due to lack of 3D benchmarks with such annotations.
3. **Gap to supervised methods** remains on standard closed-set benchmarks with few classes.

---

## 8. Architectural Perspective

From an attention/embedding standpoint, OpenScene is notable for what it does **not** use: there is no cross-attention between modalities and no end-to-end multimodal training. Instead, it relies on a **transitive alignment** strategy:

$$\text{3D points} \xrightarrow{\text{distillation}} \text{2D pixels} \xrightarrow{\text{CLIP pre-training}} \text{text}$$

The 3D network never sees text during training — it only learns to mimic pixel features that happen to be CLIP-aligned. This simplicity is both a strength (no 3D labels needed, trivial to swap in better 2D models) and a limitation (the 3D features are bounded by the quality of the 2D backbone's alignment with CLIP).

---

## 9. Significance and Takeaways

1. **Foundation model transfer to 3D:** Demonstrates that CLIP's rich semantic space can be effectively transferred to 3D via a simple distillation pipeline, without any 3D annotations.

2. **Long-tail advantage:** The zero-shot approach fundamentally sidesteps the data scarcity problem that plagues supervised methods on rare classes.

3. **Task agnosticism:** A single set of per-point features supports semantic segmentation, object search, property estimation, and more — you just change the text query.

4. **Precursor to modern 3D foundation models:** OpenScene laid groundwork for subsequent methods (LERF, LangSplat, etc.) that embed language features into 3D representations like NeRFs and Gaussian Splats.
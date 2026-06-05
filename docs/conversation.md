Option B, but with mini smoke testing gating each stage.
The key insight is: the camera distillation losses (L_O, L_V, CASD) are where the real unknowns live. You haven't run EVA-CLIP or the nuScenes projection code yet — both are untested. These are the high-risk components. LoRA, the negative bank, and the single head change are relatively well-understood and low-risk.

Pre-flight (no GPU, ~2 days code):
  - Write all scripts: verify_clip_model.py, precompute pipeline
  - Run prerequisites P1–P3 on interactive session
  - Fix any issues before launching jobs

Job 1a: Foundation (mini smoke 3 epochs → if pass, full 20 epochs)
  Components: B1 (LoRA) + L5 (C1 negatives) + A4 (single head) + T4
  Risk: LOW — no camera, no projection, just loss function changes
  Failure mode: clearly LoRA instability or head change
  If fail: fix and rerun — still have 2 weeks left

Job 1b: Camera distillation (mini smoke → if pass, full 15 epochs warm-start)
  Components: L_O + L_V standard only (NO CASD, NO L7 initially)
  Risk: MEDIUM — first real camera integration
  Failure mode: camera projection noise, coverage rate too low
  Gate: check camera coverage rate per sample (should be >70%)

Job 1c: Novel contributions (10 epochs warm-start from 1b)
  Components: + CASD (modify L_V targets) + L7 (voxel decomp)
  Risk: MEDIUM — your novel components, need to validate independently
  If fail: you still have Job 1b checkpoint as a strong result
  This is the paper contribution row in the ablation table

  Total compute: ~25 + ~20 + ~14 = ~59 hours of GPU time. With queue waits, roughly 2 weeks of calendar time, which fits the June 15 deadline if started now.


The ablation table then writes itself:
AblationPQPQ_novel_thingsPQ_novel_stuffWhat changedBaseline0.476~0~0—+ LoRA + L5 + A4~0.48–0.50~3–5 (token head improvement)~0Foundation+ L_O + L_V~0.52–0.55~30–45~10–20Camera distillation+ CASD + L7~0.54–0.58~35–50~20–35Novel contributions


The Job 1 document should be split into three markdown files:

job1a_foundation.md — LoRA + L5 + A4 + T4, 20 epochs, standalone
job1b_camera_distillation.md — L_O + L_V, 15 epochs, warm-start from 1a
job1c_novel_contributions.md — CASD + L7, 10 epochs, warm-start from 1b

Each file is short enough to give Claude Code as a single context document without overwhelming it. Each produces a checkpoint that can be evaluated independently.

DAY 0-1: Pre-flight (interactive GPU node, ~4 hours total)
│
├── Run verify_clip_model.py              (~10 min)
├── Run verify_camera_projection.py       (~5 min)
├── Run generate_negative_bank.py         (~5 min)
├── Run generate_text_prototypes.py       (update multi-prompt, ~10 min)
├── Run Job 1a MINI smoke test            (~20 min on v1.0-mini, 3 epochs)
│   └── If fail: fix and re-smoke before submitting anything
│
└── On smoke test PASS → submit TWO batch jobs simultaneously:
    │
    ├── ┌─────────────────────────────────────────────────────────┐
    │   │ BATCH JOB A: CLIP precompute  (-W 10:00)               │
    │   │ Precomputes camera patch grids for train + val          │
    │   │ ~8 hours, ~62 GB written                                │
    │   │ No training — pure feature extraction                   │
    │   └─────────────────────────────────────────────────────────┘
    │
    └── ┌─────────────────────────────────────────────────────────┐
        │ BATCH JOB B: Job 1a  (-W 36:00)                        │
        │ LoRA + L5 (C1 negatives) + A4 (single head) + T4       │
        │ 20 epochs, ~25 hours                                    │
        │ Does NOT need camera cache — safe to start immediately  │
        └─────────────────────────────────────────────────────────┘

DAY 1-2: While CLIP precompute + Job 1a are both running
│
├── Write Job 1b code: L_O + L_V camera loader integration
├── Write Job 1c code: CASD + L7
└── (No GPU needed — pure coding)

DAY 2: CLIP precompute finishes (~8h after submit)
│
└── Camera cache is ready, waiting for Job 1a to finish

DAY 2-3: Job 1a finishes (~25h after submit)
│
├── Evaluate Job 1a checkpoint on val set (interactive, ~30 min)
├── Check: PQ ≥ 0.47 (should not regress), LoRA grad norms healthy
│
├── IF PASS → immediately submit:
│   └── ┌──────────────────────────────────────────────────────────┐
│       │ BATCH JOB C: Job 1b  (-W 30:00)                         │
│       │ + L_O + L_V (standard, no CASD yet)                     │
│       │ 15 epochs, ~20 hours                                     │
│       │ Warm-start from Job 1a checkpoint                        │
│       │ Uses CLIP cache from Batch Job A ← already ready         │
│       └──────────────────────────────────────────────────────────┘
│
└── IF FAIL → diagnose (see failure modes), fix, resubmit 1a

DAY 3-4: Job 1b finishes
│
├── Evaluate Job 1b checkpoint
├── Run B12/N4 evaluation → check novel things PQ
│   └── Expected: novel things PQ 25–45, novel stuff PQ 5–15
│
├── IF PASS → immediately submit:
│   └── ┌──────────────────────────────────────────────────────────┐
│       │ BATCH JOB D: Job 1c  (-W 20:00)                         │
│       │ + CASD (L_V stuff modification) + L7 (voxel decomp)     │
│       │ 10 epochs, ~14 hours                                     │
│       │ Warm-start from Job 1b checkpoint                        │
│       └──────────────────────────────────────────────────────────┘
│
└── IF FAIL → CASD or L7 is the problem, diagnose independently

DAY 4-5: Job 1c finishes
│
└── Final evaluation — all four rows of ablation table done

Evaluation Plan
Training setup         → Evaluation              → Comparable to
─────────────────────────────────────────────────────────────────
All 16 classes         → PQ/SQ/RQ (16 classes)   → SAL, MaskPLS,
(current model         → mIoU                        P3Former
+ Job 1a losses)       → Closed-set benchmark
                         Table 1 in paper

B12/N4 (12 base)       → PQ_base (12 classes)    → 3D2D distillation
(Jobs 1a + 1b + 1c)    → PQ_novel_things          → Direct row-for-row
                       → PQ_novel_stuff               comparison
                         Table 2 in paper          Novel-4 = {bus,
                                                   pedestrian,
                                                   motorcycle,
                                                   vegetation}

B12/N4 model           → Qualitative overlays     → Novel concept
at inference with      → Voxel count stats           generalization
novel-5 text protos    → Manually verified scenes    (no quantitative
                         Figure in paper              benchmark exists)


You do not need to retrain on all 16 classes to show novel-5 capability. The B12/N4 model is actually better for novel-5 demonstration than the all-16-class model, for a specific reason.
The B12/N4 model was forced to classify bus, pedestrian, motorcycle, and vegetation purely via CLIP features during training — it never saw their class labels. That training pressure makes the model's CLIP alignment more generalizable. When you then ask it to classify a police car or an ambulance at inference by adding those text prototypes to the vocabulary, the model's backbone has been trained to produce CLIP-space features with better inter-class margin (via L7 and L_V from Jobs 1b and 1c).
The all-16-class model, by contrast, had a linear head giving it a shortcut to classify all 16 classes — it never needed to rely purely on CLIP features.

1. Train on B12/N4 with distillation losses (Jobs 1a + 1b + 1c)

2. Quantitative: evaluate B12/N4 novel-4 PQ → compare with 3D2D
   (this is your main technical comparison)

3. Qualitative: add novel-5 to text vocabulary at inference
   → run logit aggregation (Strategy L) with 16+5=21 classes
   → show camera overlays of detected novel objects
   → this demonstrates generalization BEYOND 3D2D's 4 novel classes

4. Separately: evaluate closed-set on all 16 classes
   (the B12/N4 model classifies novel-4 via CLIP at inference,
   base-12 via the trained head → reconstruct 16-class PQ)
   → compare with SAL in Table 1


   CLIP Stratergies:

   Background: What CLIP ViT Actually Produces
A CLIP ViT divides the image into a grid of non-overlapping patches, linearly embeds each patch, adds a learnable CLS token, and runs the full transformer over the sequence. After the final layer you have:
Input image (224×224)
    ↓ patchify (16px patches)
[CLS, patch_1, patch_2, ..., patch_256]   ← 257 tokens total
    ↓ transformer (attention between all tokens)
[CLS_out, p_1_out, p_2_out, ..., p_256_out]  ← 257 × 768-d vectors
The CLS token attends to all patch tokens — it is a global scene summary. Each patch token attends to all others — it contains local information but is also contextually influenced by the whole image. They are not independent.

All Six Methods
Method 1 — Global CLS token (one vector per image)
Take only the final CLS token. Discard all patch tokens. For 6 cameras: 6 vectors, each 768-d. Assign this single vector as the supervision target for every voxel visible in that camera.
Image → CLIP → CLS token (768-d)
Every voxel in that camera's field of view gets the same 768-d target
Used in: LidarCLIP, early CLIP-based 3D methods, image-level retrieval.
Pros:

Trivially simple — one forward pass, one vector per camera
Minimal storage (6 × 768 floats per sample = 36KB)
Fast at training time (just index into a 6-element array)
CLS is the most CLIP-aligned vector — it was trained to match text in contrastive learning

Cons:

Zero spatial discrimination. A car voxel and a pedestrian voxel in the same image get an identical supervision target. L_V with CLS targets would teach backbone features to look like "the whole scene" regardless of what class the voxel belongs to.
Fails entirely for L_O: pooling CLS over a predicted mask returns the same vector as pooling over the whole image.
Not used by any serious 3D dense prediction paper for per-voxel or per-object supervision.

Verdict: Not suitable for L_V or L_O. Only useful for whole-scene retrieval or coarse alignment.

Method 2 — Dense patch tokens with projection sampling ← 3D2D approach
Run one forward pass per image. Extract all 256 patch tokens (for ViT-L/14@224). Store them as a 16×16 spatial grid. For each LiDAR voxel, project it to pixel coordinates, then bilinear sample from the patch grid at that pixel location.
Image → CLIP → patch_grid (16×16×768)
Voxel v → project → pixel (u, v) → bilinear sample → 768-d vector
This is exactly what 3D2D distillation does. From their paper: "we compute per-pixel embeddings for each RGB image using a frozen segmentation model, then back-project them onto 3D surface points."
Pros:

One forward pass per image — same cost as Method 1 for the forward pass
Spatially localised — each voxel gets the feature corresponding to its image region
Cacheable as a (6, 256, 768) grid per sample (~2.4MB, ~62GB total)
At training time: just bilinear sample from the cached grid using precomputed pixel coords — negligible cost
The bilinear interpolation between adjacent patches handles voxels falling between patch boundaries
Natural fallback: if a voxel projects to the edge of the image, just use the nearest patch

Cons:

Spatial resolution is coarse: for ViT-L/14@224 each patch covers 14×14 pixels = roughly 0.7m × 0.7m on the road at 30m range. A traffic cone (0.3m wide) may fall entirely within one patch shared with its surroundings.
Patch tokens are contextually influenced by the whole image via attention — they are NOT purely local. The patch token for a "sidewalk" region is partly influenced by the road next to it and buildings behind it. This is actually a feature for CASD but a mild noise source for things.
Voxels occluded by closer objects still project to a valid pixel but get the occluder's features — no occlusion handling.

Verdict: Best general-purpose method. Use this for L_V standard (all voxels) and as the basis for L_O (pool within mask). This is the industry standard for dense 3D distillation.

Method 3 — ROI-pooled patch tokens (average within projected mask)
For each predicted or GT mask, project all its voxels to camera coordinates, collect the patch tokens at those locations, and average them. One 768-d vector per mask.
Mask M → project voxels → pixel coords → sample patch tokens → average → 768-d
This is the exact computation for L_O. It is not a separate image extraction method but rather a downstream aggregation of Method 2's patch tokens. You still run one forward pass per image and cache the patch grid; ROI pooling happens at training time from the cached grid.
Pros:

Object-level semantics — averages over the object's spatial extent, cancelling out noise from individual patches
More robust to small projection errors — a few misaligned patches are diluted
Natural fit for L_O: matched query → GT mask → project → pool → target
No additional forward passes beyond Method 2

Cons:

Same underlying issue as Method 2: pooling patch tokens from different classes within a mask dilutes the semantic signal (the pooling degradation we confirmed). However, for L_O this matters less because the mask quality at matched queries is decent (SQ=0.79 confirms mask boundaries are good).
For stuff classes (large regions), pooling many patches from different sub-regions averages out fine-grained spatial variation.

Verdict: This IS how L_O is computed. Not a separate extraction method — it is Method 2 applied at the mask level. Good for things. CASD replaces this for ambiguous stuff classes.

Method 4 — Cropped region with re-forward pass ← CASD and OpenMask3D
For each mask region, crop and resize the bounding box from the camera image to CLIP's input size (224×224), then run a completely new forward pass of CLIP on that crop. Take the CLS token of the crop.
Mask M → project → image bounding box → expand → crop → resize to 224px
       → NEW CLIP forward pass → CLS token (768-d)
Used in OpenMask3D, ODISE, and it is exactly what CASD does for ambiguous stuff classes (with an expanded bounding box to include scene context).
Pros:

CLIP's full attention mechanism runs on the object in its local context — every patch token in the crop attends to every other patch token in the crop, giving richer representation than sampling patch tokens from a full-image forward pass
The crop can be resized to use CLIP's full spatial resolution on just the object region — a traffic cone that was 1 patch in the full image becomes the entire 16×16 grid in the crop
For CASD specifically: expanding the bounding box before cropping gives you the exact "sidewalk in context" or "other_flat in context" embedding that CLIP text matches against
CLS token of a focused crop is semantically much cleaner than a patch token from a full-image pass

Cons:

Expensive: one CLIP forward pass per mask per camera. If a scene has 30 masks, each visible in 3 cameras, that is 90 CLIP forward passes per sample. At ~30ms per forward pass that is ~2.7 seconds per sample versus ~0.03s per camera for Method 2.
Cannot be trivially cached at precompute time if using PREDICTED masks (which change during training). However if using GT masks for CASD (which do not change), it CAN be cached.
For small or distant objects, the crop may contain mostly background padding after resize.
Different crops see different context, making the features less consistent across training batches.

Verdict: Only practical for a small number of masks and only when GT masks define the crop (CASD). Not suitable as the primary per-voxel feature extraction method. Used specifically for CASD's contextual stuff targets.

Method 5 — CLS + patch hybrid (weighted combination)
Used in FC-CLIP and ClearCLIP. Compute attention weights between the CLS token and each patch token using the final attention layer's weights, then take a weighted sum of patch tokens.
Image → CLIP → [CLS_out, patch_tokens]
attention_weights = softmax(CLS_out @ patch_tokens.T)
hybrid_feature = attention_weights @ patch_tokens  ← (768-d, spatially aware)
This gives a 768-d vector per image that is globally aware (like CLS) but spatially weighted toward the most semantically salient patches.
Pros:

Combines global context (CLS direction) with spatial weighting (patch distribution)
Better than raw CLS for open-vocabulary dense prediction (FC-CLIP shows this on 2D benchmarks)
Same cost as Method 1/2 — one forward pass

Cons:

Produces one vector per image, not per patch — still cannot be used directly for per-voxel supervision without a mapping
The attention-weighted combination depends on the image content globally — a busy street scene will have diffuse attention, making the output noisy
Implementation requires access to attention weights which not all open_clip versions expose cleanly
Not used in any LiDAR 3D distillation paper as the primary approach

Verdict: Interesting but not suitable for L_V (need per-voxel features, not per-image). Could potentially replace the CLS token in L_O but no evidence it helps over standard ROI pooling.

Method 6 — Multi-scale crops + averaging
Take the full image plus several crops at different scales or positions, run CLIP on each, average the resulting CLS tokens.
[full_image, crop_1, crop_2, ...] → CLIP (N forward passes) → average CLS tokens
Used in some zero-shot classification work. Not used in any LiDAR distillation paper.
Pros:

More robust embeddings — captures the object at multiple scales
Reduces sensitivity to exact crop boundaries

Cons:

N× the computation cost
Averages global CLS tokens which already have the averaging problem
Produces one vector per region, not per voxel
Not practical at training time at scale

Verdict: Not relevant for this project.

Summary Table
MethodVectors per imageCostSpatially localisedCacheableUsed for1 — Global CLS11 forward pass✗✓ (tiny)Image retrieval2 — Dense patches2561 forward pass✓✓ (62 GB)L_V, basis for L_O3 — ROI pool1 per maskDerived from 2✓ (mask-level)Via Method 2L_O (at training time)4 — Crop re-forward1 per cropN forward passes✓ (focused)✓ if GT masksCASD only5 — CLS + patch hybrid11 forward passPartial✓2D open-vocab6 — Multi-scale1N forward passes✗ExpensiveClassification


L_V (per-voxel supervision):
  → Method 2 (dense patch tokens, bilinear sampling)
  → One forward pass per image, cached as (6, 256, 768)
  → At training time: project voxels → bilinear sample from patch grid

L_O (per-object/mask supervision):
  → Method 3, derived from Method 2 cache
  → Project matched mask's voxels to camera → average patch tokens within mask
  → No additional forward passes

CASD (context-aware stuff targets):
  → Method 4 (cropped region + re-forward)
  → Only for other_flat, sidewalk, terrain (3 classes)
  → GT masks define the crop (can be precomputed and cached separately)
  → Expanded bounding box gives the contextual embedding CLIP text expects

  
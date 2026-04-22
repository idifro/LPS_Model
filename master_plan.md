# Master Plan — OpenScene Panoptic Phase A

## Purpose

This document is the **single source of truth** for implementing the supervised panoptic segmentation project built on top of the existing OpenScene setup.

It is written to be directly usable by an autonomous coding agent such as Claude Code, Claude Agent, or a VSCode GitHub Agent.

It resolves and consolidates:

- the current working state described in `Progress.md`
- the target implementation direction described in `openscene_panoptic_ablation.html`
- the practical execution constraints of the HPC single-node A100 setup

This file is intended to live at the repo root as `master_plan.md`.

---

# 1. Executive Summary

## 1.1 Final objective

Build a **supervised panoptic segmentation system for nuScenes** that:

1. uses a **frozen OpenScene/OpenSeg-distilled DisNet backbone** as the 3D feature extractor
2. uses a **MaskPLS-style transformer decoder** to predict panoptic masks
3. adds a **CLIP token head** for text-aligned mask classification
4. adds **camera cross-attention** as the main contribution
5. runs on a **single-node A100 HPC environment** with `num_workers=0`
6. is evaluated through a staged ablation sequence

## 1.2 Non-goals

This plan does **not** do any of the following in Phase A:

- retrain OpenScene from scratch
- reuse the old scene-aggregated OpenScene data format for panoptic training
- depend on multi-GPU distributed training
- start from pseudo-label SAL training
- redesign the backbone before validating the panoptic path

## 1.3 High-level implementation order

1. Freeze current working OpenScene repo and environment
2. Create a **new panoptic data path** based on **per-keyframe samples**
3. Prove MaskPLS-style panoptic training works on nuScenes mini
4. Prove frozen OpenScene DisNet works on the same per-keyframe input
5. Swap backbone into the panoptic decoder
6. Add CLIP token head
7. Add camera cross-attention
8. Run staged mini and then full ablations
9. Produce final checkpoints, metrics, visuals, and thesis-ready artifacts

---

# 2. Ground Truth Facts and Constraints

These are treated as hard facts unless directly disproven by code inspection.

## 2.1 HPC constraints

- Execution target: **single-node A100**
- Use `num_workers=0` in all DataLoaders
- Expect long-run issues from **MinkowskiEngine memory fragmentation**
- Design jobs so they can be **checkpointed and resumed**
- Prefer exclusive GPU access
- Batch size should be assumed to be **1 or 2**, not larger

## 2.2 Current OpenScene facts

- Existing OpenScene setup is already functional for semantic evaluation
- Current pretrained backbone of interest is **DisNet**
- Expected output dimension for OpenSeg-distilled variant: **768-d**
- Current progress validates checkpoint loading and semantic evaluation flow

## 2.3 Critical data-pipeline fact

The current OpenScene preprocessing pipeline aggregates **all keyframes of a scene** into one global point cloud and pairs them with images from only the last keyframe.

That representation is acceptable for the current OpenScene semantic pipeline, but it is **not acceptable for panoptic training**.

### Rule

For panoptic segmentation, **do not reuse** the old aggregated scene-level training representation.

Panoptic training must use:

- one nuScenes sample token at a time
- the matching `LIDAR_TOP` sweep for that sample
- the matching panoptic label file for that sample
- the matching 6 camera images for that sample

## 2.4 Panoptic label decoding rule

The panoptic encoding must be treated as:

- `panoptic_id = semantic_id * 1000 + instance_id`
- `semantic = panoptic // 1000`
- `instance = panoptic % 1000`

This is the format that must be used in the new panoptic dataloader and evaluation pipeline.

---

# 3. Guiding Design Principles

## 3.1 Preserve what already works

The existing OpenScene repo already provides value:

- checkpoint loading
- MinkowskiEngine integration
- current semantic evaluation
- semantic feature-space intuition
- HPC execution experience

Do not rewrite or migrate the environment unless required by a concrete blocker.

## 3.2 Separate semantic legacy pipeline from new panoptic pipeline

The old semantic pipeline remains a reference and backbone source.

The new panoptic system must be a **parallel pipeline**, not a fragile mutation of the existing scene-aggregation path.

## 3.3 Validate each layer before stacking complexity

The required order is:

1. data correctness
2. panoptic smoke test
3. backbone compatibility
4. token head
5. camera fusion
6. large-scale experiments

## 3.4 Optimize for thesis reliability, not architectural maximalism

The best thesis outcome comes from a clean ablation story with stable runs, not from attempting too many speculative changes at once.

---

# 4. Repository Strategy

## 4.1 Branching policy

Create a dedicated branch for the panoptic project:

- `feature/panoptic-openscene-phaseA`

Never develop this work directly on the current working semantic branch.

## 4.2 Top-level directory strategy

Keep existing OpenScene code intact as much as possible.

Add new panoptic code under clearly separated folders.

Recommended additions:

```text
openscene/
├── panoptic/
│   ├── dataset/
│   ├── models/
│   ├── losses/
│   ├── train/
│   ├── eval/
│   ├── scripts/
│   ├── configs/
│   └── utils/
├── third_party/
│   └── MaskPLS/
└── docs/
    └── panoptic_master_plan.md
```

## 4.3 File ownership policy

### Existing files that may be read but should not be heavily rewritten initially

- `models/disnet.py`
- `models/mink_unet.py`
- current semantic data loaders
- current evaluation scripts for semantic tasks

### New files to create

- `panoptic/dataset/nuscenes_panoptic.py`
- `panoptic/dataset/collate.py`
- `panoptic/models/openscene_backbone.py`
- `panoptic/models/mask_decoder.py`
- `panoptic/models/clip_token_head.py`
- `panoptic/models/camera_encoder.py`
- `panoptic/models/cam_decoder_layer.py`
- `panoptic/losses/panoptic_loss.py`
- `panoptic/losses/token_loss.py`
- `panoptic/eval/eval_pq.py`
- `panoptic/train/train_maskpls_smoke.py`
- `panoptic/train/train_stage2_backbone.py`
- `panoptic/train/train_stage3_token.py`
- `panoptic/train/train_stage4_camera.py`
- `panoptic/scripts/inspect_openscene_checkpoint.py`
- `panoptic/scripts/verify_nuscenes_panoptic_sample.py`
- `panoptic/scripts/precompute_text_prototypes.py`
- `panoptic/scripts/profile_camera_encoder.py`
- `panoptic/scripts/run_mini_ablation.sh`
- `panoptic/scripts/run_full_ablation.sh`

---

# 5. Phase Structure

The project is divided into 10 phases.

1. Phase 0 — Freeze baseline and verify assumptions
2. Phase 1 — Build the panoptic data pipeline
3. Phase 2 — Smoke-test MaskPLS-style panoptic training on mini
4. Phase 3 — Validate OpenScene DisNet on per-keyframe panoptic input
5. Phase 4 — Swap in frozen OpenScene backbone
6. Phase 5 — Add CLIP token head and text supervision
7. Phase 6 — Add camera cross-attention
8. Phase 7 — Run mini ablations end to end
9. Phase 8 — Run full-data experiments on single-node A100
10. Phase 9 — Consolidate outputs for thesis and future agents

Each phase has:

- objective
- implementation tasks
- files to create/modify
- commands to run
- acceptance criteria
- rollback / fallback if the phase fails

---

# 6. Phase 0 — Freeze Baseline and Verify Assumptions

## Objective

Create a stable starting point using the current working OpenScene code, environment, and checkpoint knowledge.

## Required outputs

- a frozen git branch or tag for the current working state
- a verified checkpoint dimension inspection result
- a validated list of hard constraints
- a short baseline status report

## Tasks

### 0.1 Freeze the repo state

- create feature branch for panoptic work
- ensure current semantic evaluation still runs before new changes start
- record exact commit hash in `docs/panoptic_master_plan.md`

### 0.2 Inspect the OpenScene checkpoint directly

Implement and run:

- `panoptic/scripts/inspect_openscene_checkpoint.py`

It must:

- load the OpenScene checkpoint
- identify the correct state dict root
- print possible projection and output layer shapes
- confirm whether the final output dimension is 768 or 512
- confirm expected loading path into `DisNet`

### 0.3 Validate per-sample inference compatibility assumptions

Use raw nuScenes sample-level LiDAR input, not aggregated scene files.

Goal:

- instantiate DisNet
- run one raw per-keyframe sparse tensor through it
- verify output tensor shape and numerical sanity

This is only a forward-pass compatibility check, not training.

### 0.4 Record environment invariants

Create `docs/panoptic_runtime_constraints.md` containing:

- GPU type
- queue command
- conda env name
- `PYTHONPATH`
- `num_workers=0`
- checkpoint resume requirement
- expected A100 memory envelope

## Acceptance criteria

- current semantic evaluation remains intact
- checkpoint inspection confirms output dim
- per-keyframe forward pass succeeds on one nuScenes sample
- branch is frozen and documented

## If Phase 0 fails

Stop all downstream panoptic work until the following is resolved:

- checkpoint cannot load into DisNet
- raw per-sample forward pass fails
- output dim assumption is wrong and unresolved

---

# 7. Phase 1 — Build the Panoptic Data Pipeline

## Objective

Create a brand-new **per-sample nuScenes panoptic dataset pipeline** for training and evaluation.

This phase is foundational.

## Why this phase exists

The existing scene-aggregated OpenScene data representation is not valid for panoptic training. This phase creates the proper representation.

## Required dataset behavior

Each dataset item must correspond to one nuScenes sample token and return:

- voxel coordinates for the current LiDAR sweep only
- voxel features
- GT panoptic masks
- GT classes
- semantic targets for evaluation
- instance metadata
- camera image paths for the same sample
- optional bookkeeping fields for debugging

## Files

Create:

- `panoptic/dataset/nuscenes_panoptic.py`
- `panoptic/dataset/collate.py`
- `panoptic/scripts/verify_nuscenes_panoptic_sample.py`

## Tasks

### 1.1 Implement raw nuScenes sample indexing

The dataset must:

- use nuScenes devkit directly
- map split names correctly for mini and full
- collect sample tokens by scene membership
- support `train`, `val`, and `mini` subsets cleanly

### 1.2 Load LiDAR from the current sample only

For each item:

- load `LIDAR_TOP`
- keep one sweep only
- keep intensity if available
- apply configurable range filtering
- do not aggregate across the whole scene

### 1.3 Load panoptic labels

For the same sample token:

- load the matching panoptic label file
- decode using the correct rule
- verify point count alignment with the LiDAR sample
- ignore or flag malformed cases explicitly

### 1.4 Voxelize the sample

Use the same voxel-size assumptions required by the OpenScene checkpoint.

Default target:

- `voxel_size = 0.05`

The dataset must return:

- voxel coordinates
- voxel features
- mapping from points to voxels where needed
- data needed to project GT masks onto voxels

### 1.5 Build GT thing and stuff masks

Construct:

- one binary voxel mask per thing instance
- one binary voxel mask per stuff class present

Define a single class-index convention for the full project and document it.

### 1.6 Return camera paths aligned to the same sample

For each sample, return the 6 camera image paths:

- front
- front_left
- front_right
- back
- back_left
- back_right

These paths must correspond to the exact same sample token.

### 1.7 Write custom collate function

Collate must:

- batch sparse coordinates correctly
- leave variable-length GT masks unstacked if necessary
- preserve metadata per sample

### 1.8 Verification script

`verify_nuscenes_panoptic_sample.py` must check:

- dataset length
- first few sample tokens
- LiDAR count
- voxel count
- semantic range
- number of thing masks
- number of stuff masks
- presence of all 6 cameras
- absence of shape mismatches

## Acceptance criteria

- 10 random mini samples load without exception
- all returned fields are consistent
- camera paths exist
- panoptic label decoding is correct
- collate function builds a valid SparseTensor batch

## Failure policy

Do not move to Phase 2 if any of these are unresolved:

- point/panoptic alignment mismatch
- broken semantic/instance decoding
- camera mismatch to sample token
- voxel masks all-zero or malformed

---

# 8. Phase 2 — Smoke-Test MaskPLS-Style Panoptic Training on nuScenes Mini

## Objective

Prove that the panoptic decoder, matcher, and losses work on nuScenes mini **before** integrating OpenScene.

## Required outputs

- working mini training loop
- decreasing loss over 1–2 epochs
- successful matcher behavior
- successful PQ evaluation run

## Files

Create:

- `panoptic/models/mask_decoder.py`
- `panoptic/losses/panoptic_loss.py`
- `panoptic/train/train_maskpls_smoke.py`
- `panoptic/eval/eval_pq.py`

If using MaskPLS code directly, isolate it cleanly in `third_party/MaskPLS` and wrap only what is needed.

## Tasks

### 2.1 Decide implementation strategy

Choose one of two options and document the choice.

#### Option A — vendor core MaskPLS modules

Pros:

- faster startup
- closer to known implementation

Cons:

- more brittle imports
- likely more adaptation work

#### Option B — reimplement the minimal decoder/loss stack

Pros:

- clearer control
- easier long-term maintenance

Cons:

- more code to write

For agent execution, Option A is acceptable if cleanly isolated.

### 2.2 Implement the baseline panoptic decoder

The first version must support:

- learnable queries
- self-attention
- LiDAR cross-attention
- mask prediction
- objectness prediction
- auxiliary outputs per layer if possible

Do not add camera or token head yet.

### 2.3 Implement matching and loss

Required loss components:

- mask BCE or equivalent
- Dice loss
- objectness or confidence loss
- Hungarian matching between predicted and GT masks

### 2.4 Build the mini smoke training loop

Run on nuScenes mini only.

Required runtime settings:

- `batch_size=1`
- `num_workers=0`
- AMP if stable
- gradient clipping
- high-frequency logging

### 2.5 Implement mini PQ evaluation

Need at minimum:

- convert predictions to panoptic IDs
- compare to GT
- compute PQ, SQ, RQ or a simplified verified variant
- save a few qualitative outputs

## Acceptance criteria

- training runs for at least 1–2 epochs on mini without crash
- loss decreases across the smoke run
- matcher produces non-empty matches in most frames
- evaluation script runs to completion
- at least some non-trivial PQ is observed

## Failure policy

If smoke test fails:

1. inspect GT masks for degeneracy
2. inspect matcher cost matrix
3. reduce query count
4. run one-sample overfit test
5. do not touch the backbone until smoke training is stable

---

# 9. Phase 3 — Validate OpenScene DisNet on Per-Keyframe Panoptic Input

## Objective

Verify that the pretrained OpenScene backbone behaves sensibly on the new per-sample panoptic input distribution.

## Why this phase matters

The current project already proved OpenScene on its semantic pipeline, but the panoptic system uses different sample granularity.

This phase de-risks the backbone swap.

## Files

Create:

- `panoptic/models/openscene_backbone.py`
- `panoptic/scripts/check_per_sample_semantic_alignment.py`

## Tasks

### 3.1 Wrap frozen DisNet cleanly

Implement a backbone wrapper that:

- loads pretrained weights
- strips prefixes if necessary
- freezes all backbone params
- returns both:
  - normalized CLIP-space features
  - adapter-ready features

### 3.2 Run raw feature sanity checks

On several mini samples:

- inspect output shape
- inspect norms
- confirm no NaNs
- log voxel count vs feature count

### 3.3 Run quick semantic proxy check

Without any panoptic decoder, use the raw features to compute a rough semantic proxy:

- encode text prompts
- compute cosine similarity to voxel features
- inspect top predicted classes qualitatively

This does not need to be a polished benchmark. It is only a compatibility test.

### 3.4 Confirm memory behavior

Measure:

- single-sample forward memory
- average runtime
- stability across repeated forwards

## Acceptance criteria

- pretrained DisNet loads successfully
- forward pass on per-sample voxels works reliably
- output dimension is confirmed in actual execution
- features look numerically stable
- semantic proxy is plausible enough to justify proceeding

## Failure policy

If Phase 3 fails, do not continue with backbone swap until resolved.

Potential remediation:

- verify input feature shape expected by backbone
- verify voxel size
- verify sparse tensor coordinate conventions
- inspect whether the checkpoint expects different input channels

---

# 10. Phase 4 — Swap in Frozen OpenScene Backbone

## Objective

Replace the baseline geometry-only backbone with the pretrained frozen OpenScene DisNet and a trainable adapter.

## Files

Create or modify:

- `panoptic/models/openscene_backbone.py`
- `panoptic/models/mask_decoder.py`
- `panoptic/train/train_stage2_backbone.py`
- `panoptic/configs/stage2_backbone.yaml`

## Tasks

### 4.1 Implement adapter layer

Use a trainable mapping from backbone output dim to decoder dim.

Default:

- `Linear(768 -> 256)`
- `LayerNorm(256)`

### 4.2 Define interface between backbone and decoder

Standardize the contract:

- backbone outputs per-voxel features in CLIP space
- adapter maps them to decoder feature space
- decoder consumes only the adapted features for K,V

### 4.3 Preserve frozen-backbone behavior

Training must update:

- adapter
- decoder

Training must not update:

- frozen DisNet backbone

### 4.4 Warm-start policy

If the Phase 2 decoder is stable, warm-start decoder weights where possible.

### 4.5 Run mini training

Train on mini first before touching full data.

Measure:

- loss stability
- memory usage
- PQ delta vs baseline panoptic smoke model

## Acceptance criteria

- training is stable on mini
- no gradient flows into backbone parameters
- adapted features produce valid decoder outputs
- evaluation runs end to end

## Failure policy

If instability occurs:

- reduce LR on adapter
- inspect feature normalization
- reduce query count
- verify backbone output ordering matches decoder expectation

---

# 11. Phase 5 — Add CLIP Token Head and Text Supervision

## Objective

Predict a CLIP-space embedding per query so mask predictions become text-filterable and zero-shot capable.

## Files

Create:

- `panoptic/models/clip_token_head.py`
- `panoptic/losses/token_loss.py`
- `panoptic/scripts/precompute_text_prototypes.py`
- `panoptic/train/train_stage3_token.py`

## Tasks

### 5.1 Build text prototype generation

Precompute class prototypes using CLIP text encoder.

Implementation requirements:

- support 1 or more prompts per class
- normalize embeddings
- save to disk
- document prompt template version used

Recommended improvement:

- use multiple prompt templates per class and average them

### 5.2 Implement token head

Default design:

- 3-layer MLP from decoder query embedding to CLIP embedding dimension
- L2-normalized output

### 5.3 Implement token supervision loss

For matched thing queries:

- supervise token toward the GT class prototype

For stuff queries:

- supervise either fixed query slots or matched structure depending on decoder design

### 5.4 Add training schedule for token loss weight

Start with a low token-loss weight and ramp it up gradually.

Reason:

- prevent early degradation of mask learning

### 5.5 Add monitoring for token collapse

Log:

- nearest prototype histogram
- cosine similarity distribution
- per-class prototype assignment frequency

## Acceptance criteria

- mask quality does not collapse after token head is added
- predicted tokens remain non-degenerate
- query-token similarities are sensible
- text-based filtering works on mini examples

## Failure policy

If token collapse occurs:

- lower token loss weight
- delay token loss warmup
- check prompt quality
- inspect normalization consistency between tokens and prototypes

---

# 12. Phase 6 — Add Camera Cross-Attention

## Objective

Inject camera patch features into each decoder layer via cross-attention and measure the contribution.

This is the primary novel component.

## Files

Create:

- `panoptic/models/camera_encoder.py`
- `panoptic/models/cam_decoder_layer.py`
- `panoptic/scripts/profile_camera_encoder.py`
- `panoptic/train/train_stage4_camera.py`

## Tasks

### 6.1 Implement frozen camera encoder abstraction

Support two backends:

- primary: EVA-CLIP ViT-L/14
- fallback or ablation: CLIP ViT-L/14

The abstraction must:

- load model through `open_clip`
- return patch tokens, not just CLS token
- normalize output tokens
- run in eval mode with no gradients

### 6.2 Implement camera token extraction

For each sample:

- load 6 images
- preprocess to model input size
- extract patch tokens for each camera
- concatenate all patch tokens

### 6.3 Implement decoder camera cross-attention block

Insert into each decoder layer after LiDAR cross-attention and before FFN.

### 6.4 Support camera dropout

During training, randomly drop camera features on some fraction of steps.

Reason:

- maintain graceful LiDAR-only fallback
- reduce over-dependence on cameras

### 6.5 Profile runtime and memory

Before large runs, measure:

- encoder latency
- decoder memory growth
- total step time
- batch-size feasibility

### 6.6 Mini verification

Run short mini training with:

- camera enabled
- camera disabled
- EVA-CLIP and CLIP variants if possible

## Acceptance criteria

- camera tokens are extracted reliably
- total training step fits in single A100 memory envelope
- camera-on training is stable
- camera path meaningfully changes outputs

## Failure policy

If memory is too high:

- reduce batch to 1
- use gradient accumulation
- reduce image resolution
- switch from EVA-CLIP to CLIP baseline for initial validation

---

# 13. Phase 7 — Run Mini Ablations End to End

## Objective

Complete the entire ablation chain on nuScenes mini before committing long full-dataset runs.

## Required ablation sequence on mini

1. Panoptic baseline decoder
2. + frozen OpenScene backbone
3. + token head
4. + camera cross-attention (CLIP)
5. + camera cross-attention (EVA-CLIP if feasible)

## Files

Create:

- `panoptic/scripts/run_mini_ablation.sh`
- `panoptic/configs/*.yaml`
- `panoptic/utils/experiment_registry.py`

## Tasks

### 7.1 Standardize experiment naming

Use deterministic names, for example:

- `mini_a1_maskpls`
- `mini_a2_openscene`
- `mini_a2_token`
- `mini_a2_cam_clip`
- `mini_a2_cam_eva`

### 7.2 Standardize saved outputs

Each run must save:

- config snapshot
- log file
- train curves
- checkpoint
- mini PQ summary
- sample qualitative visualizations

### 7.3 Compare results in one table

For each run report:

- PQ
- SQ
- RQ
- runtime per step
- max memory
- notes or failures

## Acceptance criteria

- all mini variants run end to end
- deltas between variants are measurable
- final full-data plan is justified by mini evidence

## Decision gate

Do not launch full long runs unless mini experiments support the design.

---

# 14. Phase 8 — Run Full-Data Experiments on Single-Node A100

## Objective

Execute the prioritized full-data experiments within the constraints of one A100 node.

## Key principle

Because compute is limited, experiment order matters.

## Priority order

### Tier 1 — must-run experiments

1. OpenScene backbone + panoptic decoder
2. + token head
3. + camera cross-attention using the fastest stable camera option

### Tier 2 — strongly desirable

4. second camera variant for ablation

### Tier 3 — optional if time permits

5. full plain geometry-only panoptic baseline if not already run at sufficient quality

## Files

Create:

- `panoptic/scripts/run_full_ablation.sh`
- `panoptic/scripts/resume_training.sh`
- `panoptic/train/common_trainer.py`

## Tasks

### 8.1 Make resume-first training mandatory

Every training script must support:

- saving checkpoint every N steps or end of epoch
- resume from checkpoint
- recovering optimizer, scaler, and scheduler state

### 8.2 Add auto-restart policy for long runs

Because ME fragmentation is a known risk, long jobs should be segmented.

Practical pattern:

- run 5–10 epochs per job
- save checkpoint
- relaunch next segment

### 8.3 Fix experiment budget up front

For each full experiment, define before launch:

- maximum epochs
- early-stop heuristic
- batch size
- accumulation factor
- expected wall clock

### 8.4 Capture resource telemetry

Log per run:

- peak VRAM
- median step time
- checkpoint size
- epochs completed before restart

## Acceptance criteria

- Tier 1 full experiments complete successfully
- restarts are recoverable
- final checkpoints are usable for evaluation and visualizations

---

# 15. Phase 9 — Consolidate Outputs for Thesis and Future Agents

## Objective

Produce a clean, restartable, well-documented project state that future agents can continue from without ambiguity.

## Required outputs

- final experiment table
- directory map of checkpoints and outputs
- exact commands for reproduce, resume, and eval
- concise status file for the next agent

## Files

Create:

- `docs/panoptic_status.md`
- `docs/panoptic_experiments.md`
- `docs/panoptic_reproduce.md`
- `docs/panoptic_results_table.csv`

## Tasks

### 9.1 Write reproducibility commands

For each completed experiment include:

- training command
- resume command
- evaluation command
- visualization command

### 9.2 Write status handoff file

The handoff file must contain:

- what is done
- what is partially done
- what is blocked
- where checkpoints live
- what the next agent should do first

### 9.3 Standardize result packaging

For each final run store:

- final checkpoint
- best checkpoint
- metrics JSON
- qualitative images
- config file
- log file

---

# 16. Detailed Run Order for the Agent

This section is the minimal execution order the agent must follow.

## Step 1

Freeze current branch and inspect checkpoint.

## Step 2

Implement per-sample nuScenes panoptic dataset and verification script.

## Step 3

Implement baseline panoptic decoder + loss + PQ eval and run smoke test on mini.

## Step 4

Wrap frozen DisNet and test it on per-sample mini data.

## Step 5

Swap DisNet into the decoder through a trainable adapter and rerun mini training.

## Step 6

Add token head + text prototype generation + token loss.

## Step 7

Add camera encoder + camera cross-attention + camera profiling.

## Step 8

Run mini ablations in standard order.

## Step 9

Promote stable configs to full-data runs using resume-first policy.

## Step 10

Generate final result summary and handoff docs.

---

# 17. Strict Do-Not-Do Rules

The agent must not do the following unless explicitly instructed by a human after explaining the tradeoff.

## 17.1 Do not reuse scene-aggregated OpenScene `.pth` data for panoptic training

Those files are semantic-pipeline artifacts, not valid panoptic training inputs.

## 17.2 Do not set `num_workers > 0`

Keep `num_workers=0` everywhere unless cluster behavior is revalidated.

## 17.3 Do not begin with camera cross-attention

Camera fusion is Phase 6, not Phase 1.

## 17.4 Do not unfreeze the OpenScene backbone early

Backbone freezing is part of the design and resource strategy.

## 17.5 Do not migrate environments without a blocker

Use the current working environment as long as possible.

## 17.6 Do not launch full runs before mini validation

Mini is the gating mechanism.

## 17.7 Do not allow unchecked long jobs without checkpoint resume

All long jobs must be restartable.

---

# 18. Standard Logging and Output Schema

Every training run must write:

- `config.yaml`
- `train.log`
- `metrics.json`
- `checkpoint_last.pth`
- `checkpoint_best.pth`
- `system_info.json`
- qualitative outputs in `vis/`

Recommended experiment directory format:

```text
exp/panoptic/<experiment_name>/
├── config.yaml
├── train.log
├── metrics.json
├── checkpoint_last.pth
├── checkpoint_best.pth
├── system_info.json
└── vis/
```

---

# 19. Acceptance Gates Summary

## Gate A — after Phase 1

- per-sample panoptic dataset is correct

## Gate B — after Phase 2

- panoptic decoder trains on mini

## Gate C — after Phase 3

- frozen DisNet works on per-sample panoptic input

## Gate D — after Phase 4

- OpenScene backbone swap is stable on mini

## Gate E — after Phase 5

- token head works without collapse

## Gate F — after Phase 6

- camera path fits memory and trains on mini

## Gate G — after Phase 7

- mini ablation results justify full runs

Only after Gate G should long full-data experiments begin.

---

# 20. Prioritized Implementation Checklist

## Priority 0 — must do first

- [ ] freeze repo state
- [ ] inspect checkpoint output dim
- [ ] verify per-sample raw DisNet forward
- [ ] create panoptic dataset
- [ ] create dataset verification script

## Priority 1 — must do next

- [ ] implement baseline panoptic decoder
- [ ] implement matching and panoptic losses
- [ ] run mini smoke training
- [ ] implement basic PQ evaluation

## Priority 2 — core thesis path

- [ ] implement frozen OpenScene backbone wrapper
- [ ] add adapter
- [ ] run mini backbone-swap training
- [ ] implement token head and text prototypes
- [ ] run mini token training

## Priority 3 — novel module

- [ ] implement camera encoder abstraction
- [ ] implement camera cross-attention
- [ ] profile camera runtime and memory
- [ ] run mini camera ablations

## Priority 4 — full experiments and packaging

- [ ] run full prioritized experiments
- [ ] save stable checkpoints
- [ ] create final result tables
- [ ] create handoff docs

---

# 21. Agent Operating Instructions

This section is written directly for an autonomous coding agent.

## 21.1 Before changing code

Read in order:

1. `master_plan.md`
2. `Progress.md`
3. checkpoint inspection output
4. current repo tree

## 21.2 When implementing a phase

For each phase:

1. create or modify only the files needed for that phase
2. keep commits small and phase-scoped
3. run the smallest validating command possible
4. record outputs before moving forward

## 21.3 When blocked

If blocked by environment or import issues:

- prefer local wrappers and path fixes over broad refactors
- do not rewrite working semantic code unless unavoidable

## 21.4 When a result is ambiguous

Prefer stronger instrumentation over speculative code changes.

Examples:

- print tensor shapes
- log class distributions
- inspect matching outputs
- save one batch worth of debug tensors

## 21.5 When moving to the next phase

Do not advance merely because code runs.
Advance only when the phase acceptance criteria are met.

---

# 22. Final Recommended Timeline

## Week 1

- Phase 0
- Phase 1

## Week 2

- Phase 2
- Phase 3

## Week 3

- Phase 4
- Phase 5

## Week 4

- Phase 6
- Phase 7

## Week 5+

- Phase 8 full runs
- Phase 9 packaging

This timeline is conceptual. Actual HPC scheduling and runtime segmentation may stretch later phases.

---

# 23. Definition of Success

The project is successful if all of the following are true:

1. a per-sample nuScenes panoptic pipeline exists and is correct
2. a MaskPLS-style panoptic decoder trains and evaluates on mini
3. frozen OpenScene DisNet integrates successfully into the panoptic decoder
4. a CLIP token head enables text-aligned mask representations
5. camera cross-attention runs stably on a single-node A100
6. at least one complete full-data OpenScene panoptic experiment is finished
7. the experiment outputs are packaged so a future agent can continue without re-deriving the plan

---

# 24. Final Directive

If there is ever a conflict between:

- the old semantic OpenScene data pipeline
- this panoptic project plan

then this panoptic plan wins for all panoptic-related code.

The old semantic pipeline remains useful only as:

- a checkpoint source
- a reference for DisNet behavior
- a reference for HPC execution patterns

It is not the training data pipeline for panoptic segmentation.

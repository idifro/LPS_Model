# Job 1 v4 Ablation: Open-Vocabulary 3D Panoptic — Full Implementation Plan

**Branch:** `feature/job1-v4-clip-teacher`
**Objective:** Fix the three root causes identified by Script A/B that prevent novel PQ > 0, then re-run the full job1a → job1b → job1c pipeline.

---

## Root Causes Being Fixed

| # | Problem | Severity | Fix in this plan |
|---|---|---|---|
| 1 | L_O teacher (patch-pooled obj_feats) has negative margin — the teacher signal points to the wrong class in text space | **Critical** | Precompute v2: crop-CLS encoding |
| 2 | Prototype margins fragile: motorcycle (0.087), vegetation (0.124), bus (0.136) | High | PR-B: discriminative multi-prompt templates |
| 3 | Neg-bank contains sub-types of all 16 classes (truck→"pickup truck" 0.94) — distorts C1 training geometry | Medium | NB-C: rebuild with sim < 0.75 filter |

---

## Overview: Phases and Gates

```
Phase 0  Stop current job1a (record checkpoint)
Phase 1  Rebuild neg-bank v2          → GATE: Script B — all novel-neg max sims < 0.80
Phase 2  Re-engineer prompts v2       → GATE: Script B — motorcycle margin > 0.12
Phase 3  Rewrite precompute script    → GATE: unit test shapes + norm check on 1 sample
Phase 4  Mini precompute + Script A   → GATE: novel margins positive, top-1 > 50%
Phase 5  Full trainval precompute     → GATE: spot-check 10 samples, no NaN
Phase 6  Job1a-v2 mini smoke          → GATE: loss_token decreasing, no NaN, lora_gnorm in [1e-4, 10]
Phase 7  Job1a-v2 full               → GATE: PQ_base ≥ 0.35 at epoch 10, temperature in [0.1, 3.0]
Phase 8  Job1b-v2 ablation A smoke    → GATE: loss_lo decreasing, novel PQ > 0 at epoch 3
Phase 9  Job1b-v2 ablation A full     → GATE: novel PQ > 0.05 at final epoch
Phase 10 Job1b-v2 ablation B smoke    → GATE: loss_lv decreasing, novel PQ ≥ ablation A
Phase 11 Job1b-v2 ablation B full     → GATE: novel PQ > ablation A
Phase 12 Job1c smoke                  → GATE: L7 loss decreasing, no NaN
Phase 13 Job1c full                   → GATE: novel PQ > 0.10 (things), voxel margin > 0.05
```

Each phase only starts after the previous phase's gate passes. If a gate fails, follow the documented failure remediation for that phase.

---

## File Map

### Files to CREATE
```
panoptic/scripts/generate_negative_bank_v2.py      Phase 1
panoptic/scripts/generate_prototypes_v2.py         Phase 2
panoptic/scripts/verify_neg_bank.py                Phase 1
panoptic/scripts/verify_crop_quality.py            Phase 4
panoptic/configs/job1a_v2.yaml                     Phase 6
panoptic/configs/job1a_v2_mini.yaml                Phase 6
panoptic/configs/job1b_v2_lo_only.yaml             Phase 8
panoptic/configs/job1b_v2_lo_only_mini.yaml        Phase 8
panoptic/configs/job1b_v2_lo_lv.yaml               Phase 10
panoptic/configs/job1b_v2_lo_lv_mini.yaml          Phase 10
panoptic/configs/job1c.yaml                        Phase 12
panoptic/configs/job1c_mini.yaml                   Phase 12
panoptic/scripts/job1a_v2_full.sh                  Phase 7
panoptic/scripts/job1b_v2_lo_smoke.sh              Phase 8
panoptic/scripts/job1b_v2_lo_full.sh               Phase 9
panoptic/scripts/job1b_v2_lo_lv_full.sh            Phase 11
panoptic/scripts/job1c_full.sh                     Phase 13
panoptic/scripts/precompute_v2_mini.sh             Phase 4
panoptic/scripts/precompute_v2_full.sh             Phase 5
panoptic/losses/voxel_decomp_loss.py               Phase 12
docs/job1_v4_ablation.md                           This file
```

### Files to MODIFY
```
panoptic/scripts/precompute_clip_voxel_features.py Phase 3
panoptic/losses/distill_voxel_loss.py              Phase 10
```

### Files NOT modified
```
panoptic/train/train_panoptic_job1.py              (compatible with all phases via config)
panoptic/losses/token_loss_c1.py                   (no changes needed)
panoptic/losses/distill_object_loss.py             (no changes needed)
panoptic/dataset/nuscenes_panoptic.py              (no changes needed)
```

### New artifacts
```
panoptic/clip_prototypes/negative_bank_v2.pt
panoptic/clip_prototypes/negative_bank_v2_concepts.txt
panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt
/fs/scratch/.../clip_features_v2/v1.0-mini/        ~50MB
/fs/scratch/.../clip_features_v2/v1.0-trainval/    ~170GB
panoptic/checkpoints_job1a_v2/
panoptic/checkpoints_job1b_v2_lo/
panoptic/checkpoints_job1b_v2_lo_lv/
panoptic/checkpoints_job1c/
```

---

## Phase 0: Stop Current Job1a

**Purpose:** Current job1a is using the old neg-bank (with sub-types) and old prompts. Re-running with v2 prototypes ensures job1b warm-starts from a correctly trained foundation.

### 0.1 Record current checkpoint

```bash
# Check job status
bjobs
# Note the epoch reached:
ls -lh panoptic/checkpoints_job1a_v1/
# Copy best checkpoint as reference before cancelling
cp panoptic/checkpoints_job1a_v1/best.pth panoptic/checkpoints_job1a_v1/best_before_cancel.pth
```

### 0.2 Cancel job

```bash
bkill <JOB_ID>
```

**Note:** If job1a has completed more than 10 epochs and PQ_base ≥ 0.30, the checkpoint is usable as a baseline reference. The v2 re-run will produce a better warm-start for job1b.

---

## Phase 1: Rebuild Negative Bank (NB-C)

**Goal:** Replace `negative_bank_200.pt` (188 entries, many near class sub-types) with a v2 bank where every entry has cosine sim < 0.75 to all 16 class prototypes.

**Why:** Script B showed novel-neg max sims: bus=0.926 ("school bus"), motorcycle=0.914 ("motorcycle rider"), vegetation=0.895 ("shrub"). These create contradictory training pressure in C1 InfoNCE. Base classes are also affected (truck=0.940 "pickup truck").

### 1.1 Create `panoptic/scripts/generate_negative_bank_v2.py`

Key changes from `generate_negative_bank.py`:
- **Remove** all vehicle sub-types: school bus, coach bus, pickup truck, SUV, minivan, delivery van, cargo van, motorcycle rider, bicycle rider, electric bicycle, scooter, moped
- **Remove** person variants: person standing, person walking, child walking, jogger, skateboarder, group of people
- **Remove** vegetation sub-types: bush, shrub, tree trunk, grass (all too close to vegetation=15)
- **Remove** surface sub-types: concrete road, asphalt road (close to driveable surface)
- **Remove** barrier near-synonyms: sound barrier, guard rail (close to barrier=0)
- **Remove** sidewalk near-synonyms: curb, crosswalk (close to sidewalk=12)
- **Remove** traffic cone near-synonyms: bollard, cone (close to traffic cone=7)
- **Keep and expand:** genuinely non-class objects: fire hydrant, parking meter, bench, trash bin, billboard, traffic light, stop sign, shopping cart, bicycle rack, dumpster, construction barrel, manhole cover, street lamp, utility pole, gas station, parking lot, bus stop (as infrastructure, not vehicle)
- **Add new safe categories:** playground equipment, vending machine, newspaper stand, ATM kiosk, flower pot, public phone booth, post box, outdoor seating, awning, sandwich board, election sign, art installation

```python
# generate_negative_bank_v2.py
NEGATIVE_CONCEPTS_V2 = [
    # Street furniture (safe — not near any class)
    "a fire hydrant in a scene", "a parking meter in a scene",
    "a bench in a scene", "a trash bin in a scene", "a mailbox in a scene",
    "a post box in a scene", "a public telephone in a scene",
    "a vending machine in a scene", "a newspaper stand in a scene",
    "a bicycle rack in a scene", "a shopping cart in a scene",
    "a dumpster in a scene", "a construction barrel in a scene",
    "a road cone in a scene",  # generic enough to stay
    "a portable toilet in a scene", "a skip container in a scene",
    # Traffic infrastructure (safe)
    "a traffic light in a scene", "a stop sign in a scene",
    "a yield sign in a scene", "a street lamp in a scene",
    "a lamp post in a scene", "a utility pole in a scene",
    "a power line in a scene", "a traffic sign in a scene",
    "a road sign in a scene", "a billboard in a scene",
    "an advertising sign in a scene",
    # Vehicles that are genuinely different from all 16 classes
    "a fire truck in a scene", "an ambulance in a scene",
    "a police car in a scene", "a tow truck in a scene",
    "a golf cart in a scene", "a forklift in a scene",
    "a bulldozer in a scene", "an excavator in a scene",
    "a road roller in a scene", "a tractor in a scene",
    "a crane in a scene", "a cement mixer in a scene",
    "an ATV in a scene", "a go-kart in a scene",
    # Animals (clearly not any class)
    "a dog in a scene", "a cat in a scene",
    "a bird in a scene", "a pigeon in a scene",
    # Structures
    "a building in a scene", "a house in a scene",
    "a tower in a scene", "a bridge in a scene",
    "a tunnel in a scene", "a parking garage in a scene",
    "a gas station in a scene", "a car wash in a scene",
    "a loading dock in a scene", "a toll booth in a scene",
    "a bus stop shelter in a scene", "a train station in a scene",
    # Infrastructure
    "a manhole cover in a scene", "a drainage grate in a scene",
    "road markings in a scene", "lane markings in a scene",
    "a speed bump in a scene", "a pothole in a scene",
    "a fence post in a scene", "a chain link fence in a scene",
    "a stone wall in a scene", "a retaining wall in a scene",
    "a jersey barrier in a scene",
    # Urban objects
    "a rooftop in a scene", "a balcony in a scene",
    "a garden in a scene", "a playground in a scene",
    "a fountain in a scene", "a statue in a scene",
    "a monument in a scene", "a sculpture in a scene",
    "an antenna in a scene", "a satellite dish in a scene",
    "a solar panel in a scene", "a wind turbine in a scene",
    "a transformer box in a scene", "a scaffolding in a scene",
    "a fire escape in a scene",
    # Shipping/industrial
    "a shipping container in a scene", "a cargo crate in a scene",
    "a conveyor belt in a scene", "a pallet in a scene",
    # Weather/surface materials (safe ones)
    "snow in a scene", "puddles in a scene",
    "a wet road in a scene",
    # Driving infrastructure
    "a roundabout in a scene", "a highway overpass in a scene",
    "a pedestrian bridge in a scene", "a railroad crossing in a scene",
    "a subway entrance in a scene",
    # Miscellaneous urban
    "a swimming pool in a scene", "a sports field in a scene",
    "a park in a scene", "a parking lot in a scene",
    "an outdoor café in a scene", "a market stall in a scene",
    "a construction site in a scene",
]
# Filter by sim to class prototypes happens in main() — see below
```

**Critical addition in `main()`:** Automatic filtering against class prototypes:

```python
def main():
    # ... load CLIP model ...
    # Load class prototypes for filtering
    class_protos = load_prototypes("panoptic/clip_prototypes/text_prototypes_vitl14_224.pt")  # (16, 768)

    # Encode all candidates
    all_embeddings = encode_concepts(NEGATIVE_CONCEPTS_V2, model, device)  # (N, 768)

    # Filter: remove any entry with sim > 0.75 to any class prototype
    sim_to_classes = all_embeddings @ class_protos.T   # (N, 16)
    max_sim = sim_to_classes.max(dim=1).values         # (N,)
    keep_mask = max_sim < 0.75
    filtered_embeddings = all_embeddings[keep_mask]
    filtered_concepts   = [c for c, k in zip(NEGATIVE_CONCEPTS_V2, keep_mask.tolist()) if k]

    print(f"Kept {keep_mask.sum()}/{len(NEGATIVE_CONCEPTS_V2)} concepts (sim < 0.75 to all class protos)")
    print(f"Rejected: {[c for c, k in zip(NEGATIVE_CONCEPTS_V2, keep_mask.tolist()) if not k]}")

    # Save
    torch.save({"embeddings": filtered_embeddings, "concepts": filtered_concepts,
                "model_name": "ViT-L/14", "filter_threshold": 0.75}, args.output)
```

**Run:**
```bash
python panoptic/scripts/generate_negative_bank_v2.py \
    --output panoptic/clip_prototypes/negative_bank_v2.pt \
    --device cuda
```

### 1.2 Create verification script `panoptic/scripts/verify_neg_bank.py`

```python
"""Quick verification of negative bank quality.

Checks:
1. No entry has sim > threshold to any class prototype
2. Bank has enough entries (>= 100)
3. Internal diversity: avg pairwise sim of bank entries
"""
import torch, sys
import torch.nn.functional as F

def verify(bank_path, proto_path, sim_threshold=0.75, min_entries=100):
    bank_data   = torch.load(bank_path,  map_location="cpu")
    proto_data  = torch.load(proto_path, map_location="cpu")

    bank   = F.normalize(bank_data["embeddings"].float(), dim=-1)   # (N, 768)
    protos = F.normalize(proto_data["embeddings"].float(), dim=-1)  # (16, 768)
    concepts = bank_data.get("concepts", [f"neg_{i}" for i in range(len(bank))])

    sim = bank @ protos.T   # (N, 16)
    max_sim, max_cls = sim.max(dim=1)

    violations = [(concepts[i], max_sim[i].item(), max_cls[i].item())
                  for i in range(len(bank)) if max_sim[i] > sim_threshold]

    print(f"Bank entries: {len(bank)}")
    print(f"Violations (sim > {sim_threshold}): {len(violations)}")
    for conc, s, cls in violations:
        print(f"  '{conc}' sim={s:.4f} to class {cls}")

    # Diversity
    bank_sim = bank @ bank.T
    avg_off_diag = (bank_sim.sum() - bank_sim.trace()) / (len(bank) * (len(bank)-1))
    print(f"Internal avg pairwise sim: {avg_off_diag:.4f}")

    # PASS / FAIL
    passed = len(violations) == 0 and len(bank) >= min_entries
    print(f"\n{'PASS' if passed else 'FAIL'}: {'All entries clean' if passed else 'Fix violations above'}")
    return passed
```

**Run:**
```bash
python panoptic/scripts/verify_neg_bank.py \
    --bank panoptic/clip_prototypes/negative_bank_v2.pt \
    --protos panoptic/clip_prototypes/text_prototypes_vitl14_224.pt
```

### GATE 1: Pass criteria
- [ ] Zero entries with sim > 0.75 to any class prototype
- [ ] Bank has ≥ 100 entries
- [ ] `verify_neg_bank.py` prints `PASS`

### GATE 1: Failure remediation
- If entries fail: add more clearly-negative concepts, re-run generation
- If bank < 100: lower filter threshold to 0.77, or add more concept categories
- Do NOT proceed to Phase 2 until gate passes

---

## Phase 2: Re-Engineer Text Prototypes (PR-B)

**Goal:** Add discriminative multi-prompt templates for the fragile class pairs. Script B showed motorcycle margin=0.087 (vs bicycle), vegetation=0.124 (vs terrain), bus=0.136 (vs truck).

**Why prompt engineering helps:** Averaging over 10 diverse templates per class produces a more stable prototype that reduces the influence of any single template that happens to be close to a competing class.

### 2.1 Create `panoptic/scripts/generate_prototypes_v2.py`

Based on `generate_prototypes_224.py` with updated multi-prompt templates:

```python
# Key change: MULTI_TEMPLATES replaces existing templates for fragile classes
# Uses ViT-L/14 (224px) to match training

CLASS_TEMPLATES = {
    "barrier": [
        "a barrier in a scene", "a concrete barrier on a road",
        "a road barrier in a driving scene", "a traffic barrier",
        "a barrier separating lanes", "a road divider barrier",
        "a safety barrier in a scene", "a highway barrier",
        "a barrier in an urban scene", "a concrete road barrier",
    ],
    "bicycle": [
        "a bicycle in a scene", "a non-motorized pedal bicycle",
        "a bicycle without an engine", "a push bicycle on a road",
        "a human-powered bicycle", "a pedal bicycle in a city",
        "a bicycle parked or moving", "a manual bicycle",
        "a bicycle in a driving scene", "a cyclist on a pedal bicycle",
    ],
    "bus": [
        "a bus in a scene", "a large passenger transit bus",
        "a public transit bus on a road", "a city bus",
        "a large bus for many passengers", "a transit bus in urban traffic",
        "a bus with many windows in a scene", "a public bus",
        "a passenger bus in a driving scene", "a full-size transit bus",
    ],
    "car": [
        "a car in a scene", "a passenger car on a road",
        "a car in a driving scene", "a regular passenger vehicle",
        "a car parked or driving", "a sedan or hatchback in a scene",
        "a personal car on the street", "a car in urban traffic",
        "a car in an urban environment", "a small personal vehicle",
    ],
    "construction vehicle": [
        "a construction vehicle in a scene",
        "a heavy construction machine", "a construction site vehicle",
        "a large yellow construction machine", "a construction equipment vehicle",
        "a machine for construction work", "a construction vehicle on a road",
        "a construction machinery in a scene", "a digger or construction machine",
        "a construction vehicle in an urban scene",
    ],
    "motorcycle": [
        "a motorcycle in a scene", "a motorized two-wheeled vehicle",
        "a motorcycle with an engine on a road",
        "a gas-powered motorbike", "a motorcycle in urban traffic",
        "a motorized bike in a driving scene",
        "a motorcycle parked or riding", "a motorbike with engine",
        "a powered two-wheeler in a scene",
        "a motorcycle or motorbike in a city",
    ],
    "pedestrian": [
        "a pedestrian in a scene", "a person walking on a street",
        "a person on foot in a driving scene", "a walking person",
        "a pedestrian in urban traffic", "a person in a city scene",
        "a pedestrian crossing or walking", "a person on a sidewalk",
        "a person walking in an urban scene", "a pedestrian in a scene",
    ],
    "traffic cone": [
        "a traffic cone in a scene", "an orange safety cone on a road",
        "a traffic cone marking a hazard", "a road safety cone",
        "a bright orange cone on the street", "a traffic cone in a driving scene",
        "a safety cone in a scene", "a road cone in an urban scene",
        "a traffic cone directing traffic", "a cone on a road",
    ],
    "trailer": [
        "a trailer in a scene", "a towed trailer on a road",
        "a trailer attached to a vehicle", "a cargo trailer",
        "a large trailer in a driving scene", "a semi-trailer without a cab",
        "a detached trailer", "a trailer in an urban scene",
        "a trailer in a driving scene", "a vehicle trailer",
    ],
    "truck": [
        "a truck in a scene", "a cargo delivery truck",
        "a large cargo truck on a road", "a truck for transporting goods",
        "a freight truck in a driving scene", "a delivery truck on the street",
        "a commercial truck carrying cargo", "a goods truck in traffic",
        "a truck for cargo transport", "a heavy goods truck",
    ],
    "driveable surface": [
        "driveable surface in a scene", "a paved road surface",
        "a road you can drive on", "a flat driveable road",
        "a road surface in a driving scene", "paved ground for driving",
        "a driving lane in a scene", "a road pavement",
        "a driveable road in an urban scene", "an asphalt driving surface",
    ],
    "other flat": [
        "other flat surface in a scene", "a flat non-road surface",
        "a flat paved area not for driving", "a flat ground surface",
        "a level non-road pavement", "a flat area in a scene",
        "flat ground near a road", "a flat surface in a scene",
        "a flat paved region", "a non-driveable flat surface",
    ],
    "sidewalk": [
        "a sidewalk in a scene", "a pedestrian walkway",
        "a pavement for pedestrians", "a sidewalk in an urban scene",
        "a footpath along a road", "a sidewalk in a driving scene",
        "a pedestrian path beside a road", "a paved walkway",
        "a sidewalk for people", "a pedestrian sidewalk",
    ],
    "terrain": [
        "terrain in a scene", "rough unpaved natural ground",
        "dirt and earth ground surface", "natural terrain in a scene",
        "off-road natural ground", "rough earth and soil",
        "natural ground without pavement", "outdoor terrain ground",
        "rough natural earth surface", "terrain outside paved roads",
    ],
    "manmade": [
        "a manmade structure in a scene", "a man-made object in a scene",
        "an artificial structure", "a built structure in a driving scene",
        "a constructed surface or object", "a manufactured structure",
        "a built environment element", "an engineered structure in a scene",
        "a constructed object in an urban scene", "a human-built structure",
    ],
    "vegetation": [
        "vegetation in a scene", "trees and plants in a driving scene",
        "green leafy trees and vegetation", "tall trees along a road",
        "natural vegetation in a scene", "trees and shrubs in an urban scene",
        "green tree canopy in a scene", "leafy vegetation in a driving scene",
        "trees and plants along a street", "natural plant life in a scene",
    ],
}
```

**Run:**
```bash
python panoptic/scripts/generate_prototypes_v2.py \
    --output_single panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --output_multi  panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt \
    --device cuda
```

### 2.2 Run Script B on v2 prototypes

```bash
python panoptic/scripts/diag_prototype_similarity.py \
    --proto_path    panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --neg_bank_path panoptic/clip_prototypes/negative_bank_v2.pt \
    --out_dir       panoptic/diagnostics/v2
```

### GATE 2: Pass criteria
- [ ] motorcycle proto_margin > 0.12
- [ ] vegetation proto_margin > 0.10
- [ ] bus proto_margin > 0.12
- [ ] All 4 novel max neg-bank sim < 0.80
- [ ] No novel class has HIGH neg-bank risk flag

### GATE 2: Failure remediation
- If motorcycle margin still < 0.10: revise motorcycle templates to more aggressively emphasize "motorized" / "engine-powered". Rerun prototype generation.
- If vegetation still confuses with terrain: add more visually distinct descriptions ("tall trees", "leafy canopy"). Rerun.
- If novel-neg sims still high: add those specific concepts to the removal list in Phase 1 and regenerate neg-bank.
- Do NOT proceed to Phase 3 until gate passes.

---

## Phase 3: Rewrite Precompute Script

**Goal:** Replace patch-pooled `obj_feats` with object-crop CLS encoding. This is the highest-priority fix.

### 3.1 Rewrite `panoptic/scripts/precompute_clip_voxel_features.py`

**Key architectural changes:**

#### New class: `CLIPCropEncoder`

```python
import torchvision.transforms as T

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
MIN_PTS_THING = 5
MIN_PTS_STUFF = 20
N_STUFF_CROPS = 6
THING_IDS_SET = set(range(1, 11))   # class ids 1-10 (1-indexed)

class CLIPCropEncoder:
    """Encodes object image crops using CLIP encode_image (CLS token)."""

    def __init__(self, clip_model, device):
        self.model  = clip_model
        self.device = device
        # Resize to 224x224 WITHOUT CenterCrop — preserves full object bbox
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def encode_crop(self, img_pil, x0, y0, x1, y1):
        """Returns (768,) float32 L2-normalized CLS feature, or None if crop invalid."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None
        crop = img_pil.crop((x0, y0, x1, y1))
        t    = self.transform(crop).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            feat = self.model.encode_image(t)         # (1, 768)
            feat = F.normalize(feat.float(), dim=-1)
        return feat.squeeze(0).cpu().numpy()
```

#### New function: `encode_instance_crop()`

```python
def encode_instance_crop(nusc, sample_token, pt_xyz, cam_name, img_pil,
                          crop_encoder, is_stuff):
    """Encode a single GT instance across one camera view.

    Returns (feat: np.ndarray (768,) or None, n_pts: int)
    """
    uv, depths, valid = project_lidar_to_camera(nusc, sample_token, pt_xyz, cam_name)
    n_pts = int(valid.sum())

    min_pts = MIN_PTS_STUFF if is_stuff else MIN_PTS_THING
    if n_pts < min_pts:
        return None, n_pts

    uv_v = uv[valid]   # (n_pts, 2) — valid projected pixel coords

    if is_stuff:
        # Sample N_STUFF_CROPS random centers from the projected stuff points
        half = 112
        sample_idx = np.random.choice(len(uv_v), min(N_STUFF_CROPS, len(uv_v)), replace=False)
        feats = []
        for si in sample_idx:
            cx, cy = uv_v[si, 0], uv_v[si, 1]
            x0 = max(0, cx - half);  x1 = min(CAM_W, x0 + 224)
            y0 = max(0, cy - half);  y1 = min(CAM_H, y0 + 224)
            f  = crop_encoder.encode_crop(img_pil, x0, y0, x1, y1)
            if f is not None:
                feats.append(f)
        if not feats:
            return None, n_pts
        avg = np.stack(feats).mean(axis=0)
        return (avg / max(np.linalg.norm(avg), 1e-8)), n_pts

    else:
        # Compute 2D bbox, expand 1.3x, make square
        u_min, u_max = uv_v[:, 0].min(), uv_v[:, 0].max()
        v_min, v_max = uv_v[:, 1].min(), uv_v[:, 1].max()
        cx = (u_min + u_max) / 2;  cy = (v_min + v_max) / 2
        hw = (u_max - u_min) / 2 * 1.3
        hh = (v_max - v_min) / 2 * 1.3
        half = max(hw, hh)                             # square crop
        x0 = max(0, cx - half);  x1 = min(CAM_W, cx + half)
        y0 = max(0, cy - half);  y1 = min(CAM_H, cy + half)
        feat = crop_encoder.encode_crop(img_pil, x0, y0, x1, y1)
        return feat, n_pts
```

#### Rewritten `process_sample()` core loop

```python
def process_sample(nusc, sample_token, dataroot, extractor, crop_encoder,
                   device, protos=None):
    # 1. Voxelize
    vox_xyz, vox_idx, inv_idx, sem, inst, N = voxelize_sample(...)
    V = len(vox_idx)

    # 2. Build GT masks (all instances, in GT order)
    gt_instances = build_gt_masks(sem, inst, inv_idx, V)
    n_inst = len(gt_instances)

    # Pre-allocate accumulators
    obj_feat_accum  = np.zeros((n_inst, 768), dtype=np.float64)
    obj_weight_sum  = np.zeros(n_inst,        dtype=np.float64)
    obj_num_views   = np.zeros(n_inst,        dtype=np.int16)
    obj_num_points  = np.zeros(n_inst,        dtype=np.int32)

    # Pre-allocate output
    obj_feats   = np.zeros((n_inst, 768), dtype=np.float16)
    obj_classes = np.zeros(n_inst,        dtype=np.int32)
    obj_valid   = np.zeros(n_inst,        dtype=np.bool_)

    # Pre-fill classes
    for k, (mask, cls_id) in enumerate(gt_instances):
        obj_classes[k] = cls_id

    # Also keep vox_feats (patch-pooled) for backward compat
    voxel_feats_accum = np.zeros((V, 768), dtype=np.float32)
    voxel_cam_count   = np.zeros(V, dtype=np.int32)

    # 3. For each camera: crop-encode instances + patch-project voxels
    sample = nusc.get("sample", sample_token)
    for cam_name in CAMERAS:
        cam_sd   = nusc.get("sample_data", sample["data"][cam_name])
        cam_path = os.path.join(dataroot, cam_sd["filename"])
        img_pil  = Image.open(cam_path).convert("RGB")

        # (a) Crop-CLS for each GT instance
        for k, (mask, cls_id) in enumerate(gt_instances):
            is_stuff  = (cls_id + 1) in STUFF_IDS  # cls_id is 0-indexed, STUFF_IDS is 1-indexed
            pt_xyz_k  = vox_xyz[mask]               # voxel XYZ for this instance
            if len(pt_xyz_k) == 0:
                continue
            feat, n_pts = encode_instance_crop(
                nusc, sample_token, pt_xyz_k, cam_name, img_pil, crop_encoder, is_stuff
            )
            if feat is not None:
                obj_feat_accum[k] += feat * n_pts
                obj_weight_sum[k] += n_pts
                obj_num_views[k]  += 1
                obj_num_points[k] += n_pts

        # (b) Patch-based vox_feats (existing CLIPPatchExtractor)
        feat_map, H_p, W_p = extractor.extract_dense_features(cam_path)
        uv_all, depths, valid = project_lidar_to_camera(nusc, sample_token, vox_xyz, cam_name)
        patch_px = 14
        for vi in np.where(valid)[0]:
            u, v = int(uv_all[vi, 0]), int(uv_all[vi, 1])
            gy = min(v // patch_px, H_p - 1)
            gx = min(u // patch_px, W_p - 1)
            voxel_feats_accum[vi] += feat_map[gy, gx]
            voxel_cam_count[vi]   += 1

    # 4. Finalize obj_feats
    for k in range(n_inst):
        if obj_weight_sum[k] > 0:
            feat_avg = obj_feat_accum[k] / obj_weight_sum[k]
            norm     = np.linalg.norm(feat_avg)
            if norm > 1e-8:
                obj_feats[k] = (feat_avg / norm).astype(np.float16)
                obj_valid[k] = True

    # 5. Finalize vox_feats (patch-pooled, same as v1)
    valid_voxels = voxel_cam_count > 0
    voxel_feats_accum[valid_voxels] /= voxel_cam_count[valid_voxels, None]
    norms = np.linalg.norm(voxel_feats_accum[valid_voxels], axis=1, keepdims=True)
    voxel_feats_accum[valid_voxels] /= np.maximum(norms, 1e-8)
    n_valid = int(valid_voxels.sum())

    # 6. Optional: compute obj_margin if protos provided
    obj_margin = np.zeros(n_inst, dtype=np.float16)
    if protos is not None:
        for k in range(n_inst):
            if not obj_valid[k]:
                continue
            feat = obj_feats[k].astype(np.float32)
            sim_all = protos @ feat      # (16,)
            cls_id  = int(obj_classes[k])
            sim_own = float(sim_all[cls_id])
            base_sims = [sim_all[b] for b in BASE_IDS if b != cls_id]
            sim_best_base = float(max(base_sims)) if base_sims else 0.0
            obj_margin[k] = np.float16(sim_own - sim_best_base)

    # 7. Sample 4096 valid voxels
    valid_indices = np.where(valid_voxels)[0]
    if len(valid_indices) >= N_VOX_SAMPLE:
        sampled = np.random.choice(valid_indices, N_VOX_SAMPLE, replace=False)
    elif len(valid_indices) > 0:
        sampled = np.random.choice(valid_indices, N_VOX_SAMPLE, replace=True)
    else:
        sampled = np.zeros(N_VOX_SAMPLE, dtype=np.int64)

    return {
        "obj_feats":      obj_feats,
        "obj_classes":    obj_classes,
        "obj_valid":      obj_valid,
        "obj_num_views":  obj_num_views.astype(np.int16),
        "obj_num_points": obj_num_points,
        "obj_margin":     obj_margin,
        "vox_feats":      voxel_feats_accum[sampled].astype(np.float16),
        "vox_indices":    sampled.astype(np.int32),
        "n_valid":        np.int32(n_valid),
        "n_voxels":       np.int32(V),
    }
```

#### New CLI args

```
--output_dir   default: /fs/scratch/.../clip_features_v2
--proto_path   default: "" (enables obj_margin if set)
--no_vox_feats flag: skip CLIPPatchExtractor (faster when L_V disabled)
```

### 3.2 Unit test on a single sample (interactive GPU)

```python
# Quick sanity check before running on all mini samples
python -c "
import numpy as np
import torch, sys
sys.path.insert(0, '.')
from panoptic.scripts.precompute_clip_voxel_features import *

# Load nuScenes mini
from nuscenes.nuscenes import NuScenes
nusc = NuScenes('v1.0-mini', '/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini', verbose=False)
token = nusc.sample[0]['token']

import clip
model, _ = clip.load('ViT-L/14', device='cuda', jit=False)
extractor   = CLIPPatchExtractor(device='cuda')
crop_encoder = CLIPCropEncoder(model, 'cuda')

result = process_sample(nusc, token, nusc.dataroot, extractor, crop_encoder, 'cuda')

print('Keys:', list(result.keys()))
print('obj_feats shape:', result['obj_feats'].shape)
print('obj_valid sum:', result['obj_valid'].sum(), '/', len(result['obj_valid']))
print('obj_num_views:', result['obj_num_views'])
print('vox_feats shape:', result['vox_feats'].shape)
print('n_valid:', result['n_valid'], '/', result['n_voxels'])

# Norm checks
norms = np.linalg.norm(result['obj_feats'][result['obj_valid']].astype(np.float32), axis=1)
print('obj_feats norms (valid only) — mean:', norms.mean(), 'min:', norms.min())
assert norms.min() > 0.9, 'Norms should be ~1.0'
print('PASS')
"
```

### GATE 3: Pass criteria
- [ ] All output keys present: `obj_feats`, `obj_valid`, `obj_num_views`, `obj_num_points`, `obj_margin`, `vox_feats`, `vox_indices`, `n_valid`, `n_voxels`
- [ ] `obj_feats` shape is `(N_gt, 768)` with N_gt > 0
- [ ] Valid obj_feats norms in [0.99, 1.01]
- [ ] `obj_valid.sum() > 0` (at least one instance visible)
- [ ] No NaN or Inf in any array
- [ ] `vox_feats.shape == (4096, 768)`

### GATE 3: Failure remediation
- NaN in obj_feats: check `encode_crop` return — add nan guard after `encode_image`
- `obj_valid.sum() == 0`: check projection — run `verify_camera_projection.py` on this sample
- Wrong shapes: check `build_gt_masks` output, ensure cls_id mapping is correct

---

## Phase 4: Mini Precompute + Script A Validation

### 4.1 Create `panoptic/scripts/precompute_v2_mini.sh`

```bash
#!/bin/bash
python -u panoptic/scripts/precompute_clip_voxel_features.py \
    --dataroot    /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --version     v1.0-mini \
    --output_dir  /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --proto_path  panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --device      cuda \
    --split       both
```

**Run interactively:**
```bash
bash panoptic/scripts/precompute_v2_mini.sh 2>&1 | tee panoptic/logs/precompute_v2_mini.log
```

Expected: ~323 files in 15–30 minutes on A100.

### 4.2 Run Script A on v2 mini features

```bash
python panoptic/scripts/diag_clip_coverage.py \
    --clip_feat_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --version       v1.0-mini \
    --proto_path    panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --out_dir       panoptic/diagnostics/v2 \
    --histogram
```

Compare against v1 baseline from `panoptic/diagnostics/coverage_v1.0-mini.json`.

### 4.3 Create `panoptic/scripts/verify_crop_quality.py` (for failure case)

If Script A gates fail, this script saves a visual contact sheet:

```python
"""Save 50 random crops per novel class with top-5 CLIP text predictions.

Usage:
    python panoptic/scripts/verify_crop_quality.py \
        --npz_dir /fs/scratch/.../clip_features_v2/v1.0-mini \
        --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
        --out_dir panoptic/diagnostics/crops
"""
# For each novel class, load 50 random instances with obj_valid=True
# Load original camera image, recompute 2D bbox projection, save crop
# Overlay top-5 text sims on image
# Saves: panoptic/diagnostics/crops/{class_name}_contact_sheet.png
```

### GATE 4: Pass criteria
- [ ] Novel valid coverage > 50% for all 4 novel classes
- [ ] Novel mean margin_to_base > 0 for all 4 novel classes
- [ ] Novel top-1 text accuracy > 50% for at least 3 of 4 novel classes
- [ ] Novel top-3 text accuracy > 70% for all 4 novel classes

### GATE 4: Failure remediation
- Margin still negative: run `verify_crop_quality.py`, inspect crops — if crops capture background, increase `MIN_PTS_THING` or expansion factor
- Low top-1 accuracy despite positive margin: prototype geometry issue — revise prompts (Phase 2)
- Low valid coverage: decrease `MIN_PTS_THING` from 5 to 3
- Do NOT proceed to Phase 5 until gate passes

---

## Phase 5: Full Trainval Precompute v2

### 5.1 Create `panoptic/scripts/precompute_v2_full.sh`

```bash
#!/bin/bash
#BSUB -J precompute-v2
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 32768
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_v2_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_v2_%J.err

module load conda && source activate openscene
module load cuda/11.0.2 cudnn/11.0_v8.0
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

python -u panoptic/scripts/precompute_clip_voxel_features.py \
    --dataroot   /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --version    v1.0-trainval \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --device cuda --split both --resume
```

**Submit:**
```bash
bsub < panoptic/scripts/precompute_v2_full.sh
```

Expected: ~8–12 hours, ~170GB output.

### 5.2 Spot-check after completion

```bash
python -c "
import numpy as np, os, glob
files = sorted(glob.glob('/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-trainval/*.npz'))
print(f'Total files: {len(files)}')
# Spot-check 10 random files
import random; random.shuffle(files)
for f in files[:10]:
    d = np.load(f)
    assert 'obj_valid' in d and 'n_voxels' in d, f'Missing fields in {f}'
    assert d['obj_feats'].shape[1] == 768
    valid_norms = np.linalg.norm(d['obj_feats'][d['obj_valid']].astype(np.float32), axis=1)
    assert not np.any(np.isnan(valid_norms)), f'NaN in {f}'
    print(f'OK: {os.path.basename(f)} — {d[\"obj_valid\"].sum()}/{len(d[\"obj_valid\"])} valid, n_valid={d[\"n_valid\"]}')
print('PASS')
"
```

### GATE 5: Pass criteria
- [ ] File count ≥ 34,000 (all trainval samples)
- [ ] All 10 spot-checked files have correct schema
- [ ] No NaN in spot-checked obj_feats (valid entries)
- [ ] `obj_valid.sum() > 0` in all spot-checked files

---

## Phase 6: Job1a-v2 — Mini Smoke Test

**Goal:** Confirm job1a trains correctly with v2 prototypes (new neg-bank, new prompts).

### 6.1 Create `panoptic/configs/job1a_v2.yaml`

Copy `job1a.yaml`, change:
```yaml
token:
  prototype_path: panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
  negative_bank:  panoptic/clip_prototypes/negative_bank_v2.pt
  multiprompt_path: panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt
  restore_from_warm_start: false

checkpoint:
  save_dir: panoptic/checkpoints_job1a_v2
  save_every: 1
```

### 6.2 Create `panoptic/configs/job1a_v2_mini.yaml`

Copy `job1a_mini.yaml`, same token changes, point to mini dataroot.

### 6.3 Run mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1a_v2_mini.yaml \
    --smoke_test 2>&1 | tee panoptic/logs/job1a_v2_mini.log
```

### GATE 6: Pass criteria
- [ ] No NaN in any loss term for all 3 smoke epochs
- [ ] `loss_token` starts < 6.0 and decreases across epochs
- [ ] `lora_gnorm` in [1e-4, 10] (after `scaler.unscale_()`)
- [ ] `temperature` < 3.0 at epoch 0 (starting at 0.5 — correctly initialized)
- [ ] No KeyError or shape mismatches in first 5 steps

### GATE 6: Failure remediation
- NaN in loss_token: check if any class prototype has zero norm; reload neg-bank
- Temperature wrong: check `restore_from_warm_start: false` is set in config
- Shape errors: likely neg-bank size change breaking proto indexing — check `n_total` print at startup

---

## Phase 7: Job1a-v2 — Full Training

### 7.1 Create `panoptic/scripts/job1a_v2_full.sh`

Based on `job1a_full.sh`, update:
- Config: `job1a_v2.yaml`
- Log files: `job1a_v2_%J.out/err`
- Wall time: 48 hours

```bash
bsub < panoptic/scripts/job1a_v2_full.sh
```

### 7.2 Monitoring checkpoints

```bash
# Check progress every few hours
ls -lh panoptic/checkpoints_job1a_v2/
# Latest epoch:
ls -t panoptic/checkpoints_job1a_v2/ | head -5
```

### GATE 7: Pass criteria (checked at epoch 10)
- [ ] PQ_base ≥ 0.35 at epoch 10
- [ ] PQ_novel = 0 (expected — no novel supervision in job1a)
- [ ] `loss_token` < 3.5 at epoch 10
- [ ] `temperature` in [0.07, 3.0] at epoch 10
- [ ] `best.pth` exists in checkpoint dir

### GATE 7: Failure remediation
- PQ_base < 0.30 at epoch 10: check mask losses are decreasing; increase `phase1_epochs` to 4
- NaN appearing after epoch 5: check `grad_clip` and `mixed_precision` settings
- Temperature saturated at 5.0: increase `temp_max` to 8.0 in config

---

## Phase 8: Job1b-v2 Ablation A — L_O Only (L_V = 0)

**Purpose:** Isolate whether crop-CLS L_O alone is sufficient to produce novel PQ > 0. No L_V to confound results.

### 8.1 Create `panoptic/configs/job1b_v2_lo_only.yaml`

Based on `job1b.yaml`:
```yaml
warm_start:
  checkpoint: panoptic/checkpoints_job1a_v2/best.pth

token:
  prototype_path:   panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
  negative_bank:    panoptic/clip_prototypes/negative_bank_v2.pt
  multiprompt_path: panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt

distillation:
  enabled: true
  clip_feat_dir: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2
  L_O:
    weight: 1.0
    warmup_epochs: 2
    loss_type: cosine
  L_V:
    weight: 0.0        # DISABLED — first ablation
    warmup_epochs: 3

checkpoint:
  save_dir: panoptic/checkpoints_job1b_v2_lo
  save_every: 1

training:
  epochs: 15
```

### 8.2 Create `panoptic/configs/job1b_v2_lo_only_mini.yaml`

Same, mini dataroot, 3 epochs.

### 8.3 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_lo_only_mini.yaml \
    --smoke_test 2>&1 | tee panoptic/logs/job1b_v2_lo_smoke.log
```

### GATE 8: Pass criteria (3-epoch smoke test)
- [ ] `loss_lo` appears in logs and starts decreasing from epoch 1 warmup completion
- [ ] `loss_lo` < 0.8 by epoch 3 (cosine loss 0→1; 0.8 = some alignment)
- [ ] No NaN in any loss
- [ ] PQ_novel > 0.0 at epoch 3 (at least one novel detection with correct class)
- [ ] PQ_base not collapsed (≥ 0.25 at epoch 3 — may drop slightly from warm-start distraction)

### GATE 8: Failure remediation
- `loss_lo` not appearing: check `distillation.enabled: true` and clip_feat_dir path
- `loss_lo` NaN: zero-norm obj_feats slipping through — check `norm > 0.1` filter in `distill_object_loss.py`
- PQ_novel still 0 after 3 epochs: may need more epochs — proceed to full job1b and check at epoch 5
- PQ_base collapsed (< 0.20): L_O weight too high — reduce to 0.5 and rerun smoke

---

## Phase 9: Job1b-v2 Ablation A — Full Training

### 9.1 Create `panoptic/scripts/job1b_v2_lo_full.sh`

```bash
#BSUB header: batch_a100, 48h, 1 GPU
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_lo_only.yaml
```

### GATE 9: Pass criteria (final epoch)
- [ ] novel PQ > 0.05 (at least motorcycle or bus getting recognized)
- [ ] PQ_base ≥ 0.38 (close to job1a level)
- [ ] `loss_lo` < 0.5 at final epoch
- [ ] `best.pth` saved in `checkpoints_job1b_v2_lo/`

### GATE 9: Failure remediation
- novel PQ = 0 after all 15 epochs: something fundamental is wrong — check that `indices_all` is being used (not `indices`) in the L_O block in `train_panoptic_job1.py`; log which queries get L_O supervision; check novel class obj_feats are being loaded correctly
- PQ_base degrades > 5 points from job1a: L_O is interfering with mask quality — add `stop_gradient` to mask path or reduce `warmup_epochs` for L_O to 3

---

## Phase 10: Job1b-v2 Ablation B — L_O + L_V Instance-Broadcast

**Purpose:** Test whether adding semantically meaningful L_V (instance-broadcast crop-CLS) improves novel PQ beyond ablation A.

### 10.1 Update `panoptic/losses/distill_voxel_loss.py`

Add `mode: broadcast` support:

```python
class VoxelDistillLoss(nn.Module):
    def __init__(self, loss_type="cosine", mode="patch"):
        super().__init__()
        self.loss_type = loss_type
        self.mode = mode   # "patch" = old v1, "broadcast" = instance-broadcast crop-CLS

    def forward(self, pred_embed, pred_masks, vox_clip_feats, vox_clip_indices,
                vox_instance_feats=None):
        """
        In "broadcast" mode: vox_instance_feats (B, 4096, 768) are the
        instance-broadcast crop-CLS features. Use these as targets instead of
        vox_clip_feats (patch-pooled).
        """
        if self.mode == "broadcast" and vox_instance_feats is not None:
            clip_target = vox_instance_feats
        else:
            clip_target = vox_clip_feats
        # rest of forward is unchanged
```

For instance-broadcast, the precompute script populates `vox_feats` differently: for each sampled voxel index, look up its GT instance and use `obj_feats[instance_of_voxel]` as the target. This is handled in the precompute script with a new `--lv_mode broadcast` flag.

### 10.2 Create `panoptic/configs/job1b_v2_lo_lv.yaml`

Same as `job1b_v2_lo_only.yaml` but:
```yaml
distillation:
  L_V:
    weight: 0.3
    warmup_epochs: 3
    mode: broadcast    # instance-broadcast crop-CLS targets

checkpoint:
  save_dir: panoptic/checkpoints_job1b_v2_lo_lv
```

### 10.3 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_lo_lv_mini.yaml \
    --smoke_test 2>&1 | tee panoptic/logs/job1b_v2_lo_lv_smoke.log
```

### GATE 10: Pass criteria
- [ ] `loss_lv` appears in logs and decreases from epoch 3+ (after L_V warmup)
- [ ] No NaN
- [ ] PQ_novel at epoch 3 ≥ ablation A epoch 3 result

---

## Phase 11: Job1b-v2 Ablation B — Full Training

### GATE 11: Pass criteria
- [ ] novel PQ at final epoch > ablation A final epoch result
- [ ] If not: L_V-B is not helping — proceed to job1c with ablation A checkpoint

---

## Phase 12: Job1c — L7 (Voxel Decomposition) + CASD

**Purpose:** Direct per-voxel backbone supervision (L7) to fix inter-class margin at backbone level. CASD for contextually ambiguous stuff classes.

### 12.1 Create `panoptic/losses/voxel_decomp_loss.py`

```python
"""L7: Voxel decomposition loss — direct per-voxel InfoNCE on backbone features.

Forces backbone features to have fine-grained CLIP-space margins.
Gradient path: L7 → backbone_feats → LoRA adapters (bypasses decoder entirely).
"""
class VoxelDecompLoss(nn.Module):
    def __init__(self, proto_path, neg_bank_path, n_sample=10000, temperature=0.07):
        # Load all 16 class protos + neg bank as target vocab
        # temp=0.07 fixed (CLIP native)
        pass

    def forward(self, backbone_feats, gt_sem, valid_mask):
        """
        backbone_feats: (V, 768) from backbone output (requires_grad=True via LoRA)
        gt_sem: (V,) GT semantic class (0-indexed, ignore=0)
        valid_mask: (V,) bool — valid voxels for supervision (excludes novel classes)

        Returns: per-voxel InfoNCE loss scalar
        """
        # Subsample up to n_sample valid voxels
        # sim = L2Norm(feats) @ all_protos.T / temperature
        # loss = cross_entropy(sim, gt_class_remapped)
        # Novel classes excluded from supervision (same as C1)
```

### 12.2 CASD: Context-Aware Stuff Distillation

CASD replaces L_V targets for ambiguous stuff classes with context-expanded crop CLS tokens (online, no precompute):

**Classes:** other_flat(11), sidewalk(12), terrain(13)  
**Expansion:** 3.0× / 2.5× / 2.0× respectively  

CASD is implemented as an online computation in the training loop:
1. Find voxels of CASD class in GT
2. Project to best camera (most visible) → compute bbox
3. Expand bbox by class factor → crop → CLIP encode_image → CLS
4. Broadcast to all voxels of that class → replace L_V target

**Addition in `train_panoptic_job1.py`:**
```python
if casd_enabled:
    casd_targets = compute_casd_targets(sample, backbone_feats, gt_sem, clip_model, casd_scales)
    # casd_targets shape: (V, 768), only non-zero for CASD class voxels
```

### 12.3 Create `panoptic/configs/job1c.yaml`

```yaml
warm_start:
  checkpoint: panoptic/checkpoints_job1b_v2_lo/best.pth  # or lo_lv, whichever is better

distillation:
  L_V:
    weight: 0.1
    mode: broadcast
  casd:
    enabled: true
    classes: [11, 12, 13]
    scales:  [3.0, 2.5, 2.0]

l7:
  enabled: true
  weight: 0.5
  warmup_epochs: 5
  n_sample_voxels: 10000
  temperature: 0.07

training:
  epochs: 10
  lr_decoder: 2.0e-5    # further reduced — mostly backbone fine-tuning
  lr_lora: 5.0e-5

checkpoint:
  save_dir: panoptic/checkpoints_job1c
```

### 12.4 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1c_mini.yaml \
    --smoke_test 2>&1 | tee panoptic/logs/job1c_smoke.log
```

### GATE 12: Pass criteria
- [ ] `loss_l7` appears in logs and decreases after warmup
- [ ] `lora_gnorm` increases slightly vs job1b (L7 provides stronger LoRA gradient)
- [ ] No NaN
- [ ] CASD targets non-zero for at least 1 sample per epoch

---

## Phase 13: Job1c — Full Training

### GATE 13: Pass criteria (final evaluation)
- [ ] novel PQ_things > 0.10 (bus, motorcycle, pedestrian)
- [ ] novel PQ_stuff > 0.05 (vegetation)
- [ ] voxel margin (from `eval_voxel_margin.py`) > 0.05 for all base classes
- [ ] PQ_base ≥ 0.38 (not degraded vs job1b)

---

## Execution Timeline

```
DAY 0 (today, no GPU):
  Phase 0  Cancel job1a, record checkpoint
  Phase 1  generate_negative_bank_v2.py → verify_neg_bank.py [GATE 1]
  Phase 2  generate_prototypes_v2.py → Script B [GATE 2]

DAY 0 (interactive GPU session):
  Phase 3  Rewrite precompute → unit test 1 sample [GATE 3]
  Phase 4  precompute_v2_mini.sh → Script A [GATE 4]

DAY 1 (submit batch jobs):
  Phase 5  precompute_v2_full.sh (8–12h) [GATE 5]
  Phase 6  job1a_v2 mini smoke test (interactive, 20min) [GATE 6]
  Phase 7  job1a_v2_full.sh (48h) — submit immediately after GATE 6 passes

DAY 3 (job1a finishes):
  Phase 8  job1b_v2_lo smoke test (interactive, 20min) [GATE 8]
  Phase 9  job1b_v2_lo_full.sh (24h) — submit after GATE 8

DAY 4 (job1b ablation A finishes):
  Evaluate novel PQ — if > 0.05, proceed to ablation B
  Phase 10 job1b_v2_lo_lv smoke test [GATE 10]
  Phase 11 job1b_v2_lo_lv_full.sh (24h)

DAY 5 (job1b ablation B finishes):
  Compare ablation A vs B
  Phase 12 job1c smoke test [GATE 12]
  Phase 13 job1c_full.sh (15h) — submit with best job1b checkpoint

DAY 6: Final evaluation, ablation table
```

---

## Ablation Table (Target Results)

| Experiment | PQ_base | PQ_novel_things | PQ_novel_stuff | Key change |
|---|---|---|---|---|
| Baseline (closed-set) | 0.476 | 0 | 0 | — |
| Job1a-v1 (old neg-bank) | 0.30+ | 0 | 0 | LoRA + C1 + margin |
| Job1a-v2 | ≥ 0.35 | 0 | 0 | + better neg-bank + prompts |
| Job1b ablation A (L_O only) | ≥ 0.35 | > 0.05 | > 0.02 | + crop-CLS L_O |
| Job1b ablation B (L_O + L_V) | ≥ 0.35 | > 0.08 | > 0.05 | + instance-broadcast L_V |
| Job1c (+ L7 + CASD) | ≥ 0.38 | > 0.12 | > 0.08 | + voxel decomp + CASD |

---

## Quick Reference: Gate Checklist

| Phase | Gate command | Pass condition |
|---|---|---|
| 1 | `python verify_neg_bank.py` | Zero violations, ≥100 entries |
| 2 | `python diag_prototype_similarity.py --proto ... --neg_bank ...` | All margins > 0.10, novel-neg < 0.80 |
| 3 | Python one-liner unit test | Correct shapes, norms ~1.0, no NaN |
| 4 | `python diag_clip_coverage.py` | Novel margin > 0, top-1 > 50% |
| 5 | Python spot-check 10 files | Correct schema, no NaN |
| 6 | Job1a mini smoke log | No NaN, T < 3.0, loss decreasing |
| 7 | WandB / checkpoint eval at epoch 10 | PQ_base ≥ 0.35 |
| 8 | Job1b smoke log | loss_lo decreasing, novel PQ > 0 at epoch 3 |
| 9 | Final eval checkpoint | novel PQ > 0.05 |
| 10 | Job1b-B smoke log | loss_lv decreasing, no regression |
| 11 | Final eval checkpoint | novel PQ ≥ ablation A |
| 12 | Job1c smoke log | loss_l7 decreasing, no NaN |
| 13 | Final eval checkpoint | novel PQ_things > 0.10 |

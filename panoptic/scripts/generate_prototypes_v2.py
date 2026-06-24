"""Generate discriminative multi-prompt CLIP text prototypes v2 for nuScenes 16-class.

PR-B strategy: per-class prompt sets engineered to maximize inter-class margin,
especially for fragile pairs:
  - motorcycle vs bicycle  (v1 margin=0.0875 — FRAGILE)
  - vegetation vs terrain  (v1 margin=0.1241)
  - bus vs truck           (v1 margin=0.1356)

Produces two files:
  text_prototypes_vitl14_224_v2.pt       — (16, 768) averaged prototype per class
  text_prototypes_vitl14_224_multiprompt_v2.pt — (16, T, 768) per-prompt for T4 sampling

Usage:
    python panoptic/scripts/generate_prototypes_v2.py \
        --output_dir panoptic/clip_prototypes \
        --device cuda
"""
import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

# nuScenes 16-class names (0-indexed)
NUSCENES_CLASS_NAMES = [
    "barrier",              # 0
    "bicycle",              # 1
    "bus",                  # 2
    "car",                  # 3
    "construction vehicle", # 4
    "motorcycle",           # 5
    "pedestrian",           # 6
    "traffic cone",         # 7
    "trailer",              # 8
    "truck",                # 9
    "driveable surface",    # 10
    "other flat",           # 11
    "sidewalk",             # 12
    "terrain",              # 13
    "manmade",              # 14
    "vegetation",           # 15
]

# ──────────────────────────────────────────────────────────────────────────────
# Discriminative multi-prompt templates.
# Goals:
#   - barrier:    distinctive from manmade/wall  → emphasize roadside placement
#   - bicycle:    distinctive from motorcycle    → emphasize pedals, chain, thin frame, no engine
#   - bus:        distinctive from truck         → emphasize many passengers, transit
#   - motorcycle: distinctive from bicycle       → emphasize engine block, exhaust, fuel tank, heavy chassis
#   - pedestrian: distinctive from sidewalk/env  → emphasize human figure, not scene context
#   - terrain:    distinctive from vegetation    → emphasize bare ground, no leaves
#   - vegetation: distinctive from terrain       → emphasize green, leafy, organic
# ──────────────────────────────────────────────────────────────────────────────
CLASS_PROMPTS = {
    "barrier": [
        "a roadside barrier in a driving scene",
        "a concrete barrier dividing traffic lanes",
        "a protective barrier along a road",
        "a road barrier in an urban driving scene",
        "a traffic separation barrier on a road",
        "a jersey barrier on a road",
        "a temporary concrete barrier on a street",
        "a road safety barrier seen from a vehicle",
        "a lane separation barrier in a city",
        "a median barrier on a highway",
    ],
    "bicycle": [
        "a bicycle with pedals and a chain drive",
        "a bicycle with thin frame, pedals, and no engine",
        "a pedal bicycle with chain and sprocket",
        "a lightweight bicycle with pedals and spokes",
        "a bicycle frame with handlebars and pedals",
        "a non-motorized bicycle with pedal cranks",
        "a bicycle with thin wheels and pedal drive",
        "a manual bicycle with no engine or exhaust",
        "a human-powered bicycle with chain and pedals",
        "a pedal-driven bicycle with thin spoked wheels",
    ],
    "bus": [
        "a bus in a driving scene",
        "a large passenger transit bus on a road",
        "a public transit bus in an urban scene",
        "a city bus carrying many passengers",
        "a large bus for public transportation",
        "a transit bus on a city street",
        "a passenger bus seen from a vehicle",
        "a large multi-passenger bus in traffic",
        "a public bus in a city driving scene",
        "an urban transit bus on a street",
    ],
    "car": [
        "a car in a driving scene",
        "a passenger car in a city",
        "a sedan or hatchback on a road",
        "an automobile in an urban environment",
        "a car in traffic",
        "a personal vehicle on a street",
        "a car parked or moving in a scene",
        "a passenger automobile in a driving scene",
        "a four-wheeled private car in traffic",
        "a car seen from another vehicle",
    ],
    "construction vehicle": [
        "a construction vehicle in a driving scene",
        "a heavy construction machine near a road",
        "a construction equipment vehicle in an urban scene",
        "a yellow construction machine on a worksite",
        "an excavator or bulldozer in a driving scene",
        "a construction site vehicle on a road",
        "a large construction machine near a street",
        "a construction equipment near a driving scene",
        "a heavy machinery vehicle at a construction site",
        "a worksite vehicle in an urban driving scene",
    ],
    "motorcycle": [
        "a motorcycle with an engine block and exhaust pipe",
        "a motorcycle with a fuel tank and heavy chassis",
        "a motorcycle with engine, exhaust, and kickstand",
        "a gas-powered motorcycle with visible engine components",
        "a motorcycle with a large engine and exhaust system",
        "a motorcycle engine and fuel tank on a road",
        "a heavy motorized motorcycle with exhaust pipes",
        "a motorcycle with engine block seen from a vehicle",
        "a powered motorcycle with fuel tank in traffic",
        "a motorcycle with mechanical engine parts in a scene",
    ],
    "pedestrian": [
        "a pedestrian in a driving scene",
        "a human figure walking in an urban environment",
        "a person on foot in city traffic",
        "a walking human seen from a vehicle",
        "a person standing upright in a driving scene",
        "a human body walking near a road",
        "a person in motion on a street",
        "a human pedestrian figure in traffic",
        "a person walking independently in a scene",
        "a human figure on foot in an urban area",
    ],
    "traffic cone": [
        "a traffic cone in a driving scene",
        "an orange traffic cone on a road",
        "a traffic cone marking a road hazard",
        "a traffic cone in an urban driving scene",
        "a road traffic cone seen from a vehicle",
        "an orange cone used for traffic management",
        "a traffic cone on a street",
        "a reflective traffic cone on a road",
        "a traffic delineation cone in a driving scene",
        "a traffic cone at a construction zone",
    ],
    "trailer": [
        "a trailer in a driving scene",
        "a vehicle trailer being towed on a road",
        "a towed trailer behind a truck or car",
        "a cargo trailer in a driving scene",
        "a trailer attached to a towing vehicle",
        "a towed vehicle trailer on a street",
        "a trailer unit in traffic",
        "a detached trailer on a road",
        "a semi-trailer or cargo trailer seen from a vehicle",
        "a towed trailer in an urban driving scene",
    ],
    "truck": [
        "a truck in a driving scene",
        "a cargo delivery truck on a road",
        "a large goods vehicle in an urban scene",
        "a freight truck in traffic",
        "a delivery truck on a city street",
        "a heavy goods truck in a driving scene",
        "a truck carrying cargo in a city",
        "a large truck for goods transport in traffic",
        "a haulage truck on a road",
        "a freight-carrying truck in an urban environment",
    ],
    "driveable surface": [
        "a driveable road surface in a driving scene",
        "the road that vehicles drive on",
        "a paved road surface seen from a vehicle",
        "a drivable asphalt or concrete road",
        "the road beneath a vehicle in a driving scene",
        "a vehicle road surface in an urban environment",
        "a paved surface used for driving",
        "a road pavement in a driving scene",
        "the drivable road in a city environment",
        "an asphalt road surface in a driving scene",
    ],
    "other flat": [
        "a flat non-road surface in a driving scene",
        "a flat ground area beside a road",
        "an unclassified flat surface in a driving scene",
        "a flat outdoor surface not meant for vehicles",
        "a flat terrain area in an urban scene",
        "a flat open ground in a driving environment",
        "a flat surface in an urban outdoor area",
        "a flat ground beside a road in a driving scene",
        "an open flat area in an urban driving scene",
        "a flat outdoor ground surface near a road",
    ],
    "sidewalk": [
        "a sidewalk in a driving scene",
        "a pedestrian footpath beside a road",
        "a paved sidewalk for walking in a city",
        "a footway beside a road in an urban scene",
        "a pedestrian pavement in a driving scene",
        "a walkway beside a street in a city",
        "a paved pedestrian path in an urban area",
        "a sidewalk or pavement for pedestrians",
        "a footpath beside a road in a driving scene",
        "an urban sidewalk along a city street",
    ],
    "terrain": [
        "terrain in a driving scene",
        "rough unpaved ground in an outdoor scene",
        "bare earth or dirt terrain near a road",
        "natural ground without pavement or vegetation",
        "rough outdoor terrain near a driving scene",
        "bare soil or gravel ground in an urban scene",
        "unpaved natural ground beside a road",
        "dirt or gravel terrain in a driving environment",
        "rough ground surface without vegetation or pavement",
        "natural bare terrain near a road in a scene",
    ],
    "manmade": [
        "a manmade structure in a driving scene",
        "a building or constructed structure near a road",
        "a man-made object in an urban driving environment",
        "a constructed building or wall in a scene",
        "an artificial structure like a building or fence",
        "a building facade in an urban driving scene",
        "a manmade structure visible from a vehicle",
        "an urban constructed structure in a driving scene",
        "a building or manmade surface near a road",
        "a human-built structure in an urban scene",
    ],
    "vegetation": [
        "vegetation in a driving scene",
        "green trees and leafy plants along a road",
        "a green leafy tree or bush near a street",
        "natural vegetation with leaves and branches",
        "green foliage and plants in an urban scene",
        "trees and shrubs in a driving environment",
        "leafy green vegetation beside a road",
        "green trees with canopy in a driving scene",
        "natural plant life with green leaves near a road",
        "trees and bushes with green foliage in a scene",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="panoptic/clip_prototypes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="ViT-L/14")
    args = parser.parse_args()

    import clip

    print(f"Loading CLIP {args.model_name}...")
    model, _ = clip.load(args.model_name, device=args.device, jit=False)
    model.eval()

    # Verify all 16 classes have prompts
    for name in NUSCENES_CLASS_NAMES:
        assert name in CLASS_PROMPTS, f"Missing prompts for class: {name}"
    n_prompts_per_class = [len(CLASS_PROMPTS[n]) for n in NUSCENES_CLASS_NAMES]
    print(f"  Prompts per class: {n_prompts_per_class}")
    assert all(n == 10 for n in n_prompts_per_class), "Expected exactly 10 prompts per class"

    all_class_embs = []      # (16, 768) — averaged
    all_prompt_embs = []     # (16, 10, 768) — per-prompt

    for cls_idx, class_name in enumerate(NUSCENES_CLASS_NAMES):
        prompts = CLASS_PROMPTS[class_name]
        tokens = clip.tokenize(prompts).to(args.device)
        with torch.no_grad():
            emb = model.encode_text(tokens)            # (10, 768)
            emb = F.normalize(emb.float(), dim=-1)     # per-prompt L2
        emb = emb.cpu()

        # Average and re-normalize
        avg_emb = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1).squeeze(0)  # (768,)

        all_class_embs.append(avg_emb)
        all_prompt_embs.append(emb)
        print(f"  [{cls_idx:2d}] {class_name:<25} norm={avg_emb.norm():.4f}")

    class_embs = torch.stack(all_class_embs, dim=0)    # (16, 768)
    prompt_embs = torch.stack(all_prompt_embs, dim=0)  # (16, 10, 768)

    print(f"\nClass embeddings shape: {class_embs.shape}")
    print(f"Per-prompt embeddings shape: {prompt_embs.shape}")

    # Print inter-class similarity matrix for fragile pairs
    print("\nInter-class cosine similarity (fragile pairs):")
    fragile_pairs = [
        (1, 5, "bicycle vs motorcycle"),
        (13, 15, "terrain vs vegetation"),
        (2, 9, "bus vs truck"),
        (6, 5, "pedestrian vs motorcycle"),
    ]
    for a, b, label in fragile_pairs:
        sim = (class_embs[a] @ class_embs[b]).item()
        margin = 1.0 - sim
        flag = "  ← FRAGILE" if margin < 0.09 else ""
        print(f"  {label:<30}: sim={sim:.4f}  margin={margin:.4f}{flag}")

    # Save averaged prototypes
    import os
    out1 = os.path.join(args.output_dir, "text_prototypes_vitl14_224_v2.pt")
    torch.save({
        "embeddings": class_embs,
        "class_names": NUSCENES_CLASS_NAMES,
        "model_name": args.model_name,
        "n_prompts_per_class": 10,
        "strategy": "PR-B discriminative multi-prompt, averaged + re-normalized",
    }, out1)
    print(f"\nSaved averaged prototypes to: {out1}")

    # Save per-prompt prototypes (for T4 multi-prompt sampling during training)
    out2 = os.path.join(args.output_dir, "text_prototypes_vitl14_224_multiprompt_v2.pt")
    torch.save({
        "embeddings": prompt_embs,      # (16, 10, 768)
        "class_names": NUSCENES_CLASS_NAMES,
        "model_name": args.model_name,
        "n_prompts": 10,
        "strategy": "PR-B discriminative multi-prompt, per-prompt for random sampling",
    }, out2)
    print(f"Saved per-prompt prototypes to: {out2}")


if __name__ == "__main__":
    main()

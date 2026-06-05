"""Generate 200+ negative text embedding bank for C1 token loss.

Uses OpenAI CLIP ViT-L/14@224 (same as training text prototypes).
Concepts include driving confusors, infrastructure, generic objects.

Output: panoptic/clip_prototypes/negative_bank_200.pt
  - embeddings: (N, 768) L2-normalized
  - concepts: list of concept strings
  - model_name: "ViT-L/14"

Usage:
    python panoptic/scripts/generate_negative_bank.py [--device cuda]
"""
import argparse
import sys
import torch

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

NEGATIVE_CONCEPTS = [
    # Driving confusors (similar to base classes)
    "a van in a scene", "an SUV in a scene", "a minivan in a scene",
    "a pickup truck in a scene", "a delivery van in a scene",
    "a scooter in a scene", "a moped in a scene", "a segway in a scene",
    "a child walking in a scene", "a jogger in a scene", "a skateboarder in a scene",
    "a cyclist in a scene", "an electric bicycle in a scene",
    "a school bus in a scene", "a coach bus in a scene", "a double decker bus in a scene",
    "a fire truck in a scene", "an ambulance in a scene", "a police car in a scene",
    "a tow truck in a scene", "a garbage truck in a scene", "a cement truck in a scene",
    # Infrastructure / novel driving objects
    "a street lamp in a scene", "a lamp post in a scene", "a utility pole in a scene",
    "a traffic light in a scene", "a traffic sign in a scene", "a stop sign in a scene",
    "a fire hydrant in a scene", "a bollard in a scene", "a parking meter in a scene",
    "a bench in a scene", "a trash bin in a scene", "a mailbox in a scene",
    "a fence in a scene", "a guard rail in a scene", "a curb in a scene",
    "a speed bump in a scene", "a pothole in a scene", "a crosswalk in a scene",
    "a tree trunk in a scene", "a bush in a scene", "a shrub in a scene",
    "a wall in a scene", "a pillar in a scene", "a column in a scene",
    # Ground surface variants
    "an asphalt road in a scene", "a concrete road in a scene",
    "a gravel road in a scene", "a dirt road in a scene",
    "a bike lane in a scene", "a parking lot in a scene",
    "a driveway in a scene", "an intersection in a scene",
    # Vehicles not in base classes
    "a motorcycle with sidecar in a scene", "a tricycle in a scene",
    "an ATV in a scene", "a golf cart in a scene",
    "a forklift in a scene", "a bulldozer in a scene",
    "an excavator in a scene", "a crane in a scene",
    "a road roller in a scene", "a tractor in a scene",
    # General objects (regularisation)
    "a dog in a scene", "a cat in a scene", "a bird in a scene",
    "a person sitting in a scene", "a person standing in a scene",
    "a group of people in a scene",
    "a building in a scene", "a house in a scene", "a tower in a scene",
    "a bridge in a scene", "a tunnel in a scene", "an underpass in a scene",
    # Materials/surfaces
    "metal in a scene", "concrete in a scene", "glass in a scene",
    "grass in a scene", "sand in a scene", "gravel in a scene",
    "snow in a scene", "puddles in a scene", "shadows in a scene",
    # Ambiguous stuff-like objects
    "a parking barrier in a scene", "a road divider in a scene",
    "a median strip in a scene", "a raised kerb in a scene",
    "a drainage grate in a scene", "a manhole cover in a scene",
    "road markings in a scene", "lane markings in a scene",
    # Night/weather variants
    "a car at night in a scene", "a person in rain in a scene",
    "a wet road in a scene", "a road with reflections in a scene",
    # Additional confusors for per-class margin
    "a sedan in a scene", "a coupe in a scene", "a hatchback in a scene",
    "a sports car in a scene", "a limousine in a scene",
    "a cargo van in a scene", "a food truck in a scene",
    "a dump truck in a scene", "a tanker truck in a scene",
    "a flatbed truck in a scene", "a box truck in a scene",
    "a motorcycle rider in a scene", "a bicycle rider in a scene",
    "a wheelchair in a scene", "a stroller in a scene",
    "a shopping cart in a scene", "a pallet in a scene",
    "a dumpster in a scene", "a portable toilet in a scene",
    "a construction barrel in a scene", "a construction sign in a scene",
    "a jersey barrier in a scene", "a chain link fence in a scene",
    "a wooden fence in a scene", "a stone wall in a scene",
    "a retaining wall in a scene", "a sound barrier in a scene",
    "a highway overpass in a scene", "a pedestrian bridge in a scene",
    "a railroad crossing in a scene", "a roundabout in a scene",
    "a toll booth in a scene", "a gas station in a scene",
    "a car wash in a scene", "a loading dock in a scene",
    "a bus stop in a scene", "a train station in a scene",
    "a subway entrance in a scene", "a parking garage in a scene",
    "a rooftop in a scene", "a balcony in a scene",
    "a garden in a scene", "a playground in a scene",
    "a sports field in a scene", "a swimming pool in a scene",
    "a fountain in a scene", "a statue in a scene",
    "a monument in a scene", "a billboard in a scene",
    "an antenna in a scene", "a satellite dish in a scene",
    "a solar panel in a scene", "a wind turbine in a scene",
    "a power line in a scene", "a transformer box in a scene",
    "a fire escape in a scene", "a scaffolding in a scene",
    "a crane boom in a scene", "a conveyor belt in a scene",
    "a shipping container in a scene", "a cargo crate in a scene",
    "a tire in a scene", "a wheel in a scene",
    "a hubcap in a scene", "a license plate in a scene",
    "a side mirror in a scene", "a bumper in a scene",
    "a hood in a scene", "a trunk in a scene",
    "a windshield in a scene", "a headlight in a scene",
    "a taillight in a scene", "an exhaust pipe in a scene",
    "a roof rack in a scene", "a spoiler in a scene",
    "a mud flap in a scene", "a running board in a scene",
    "a tow hitch in a scene", "a bike rack in a scene",
    "a ski rack in a scene", "a canoe on car in a scene",
    "an ice cream truck in a scene", "a mail truck in a scene",
    "a moving truck in a scene", "a camper van in a scene",
    "a motorhome in a scene", "a horse trailer in a scene",
    "a boat trailer in a scene", "a utility trailer in a scene",
    "a car carrier in a scene", "a logging truck in a scene",
]

# Deduplicate and limit
NEGATIVE_CONCEPTS = list(dict.fromkeys(NEGATIVE_CONCEPTS))[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="panoptic/clip_prototypes/negative_bank_200.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="ViT-L/14")
    args = parser.parse_args()

    import clip
    print(f"Loading CLIP {args.model_name}...")
    model, _ = clip.load(args.model_name, device=args.device, jit=False)
    model.eval()

    print(f"Encoding {len(NEGATIVE_CONCEPTS)} negative concepts...")
    embeddings = []
    batch_size = 32
    for i in range(0, len(NEGATIVE_CONCEPTS), batch_size):
        batch = NEGATIVE_CONCEPTS[i:i + batch_size]
        tokens = clip.tokenize(batch).to(args.device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.float().cpu())

    negative_bank = torch.cat(embeddings, dim=0)
    print(f"Negative bank shape: {negative_bank.shape}")
    print(f"Norm check: {negative_bank.norm(dim=-1).mean():.4f}")

    output = {
        "embeddings": negative_bank,
        "concepts": NEGATIVE_CONCEPTS,
        "model_name": args.model_name,
    }
    torch.save(output, args.output)
    print(f"Saved to {args.output}")

    # Also save concepts list as txt
    txt_path = args.output.replace(".pt", "_concepts.txt")
    with open(txt_path, "w") as f:
        for c in NEGATIVE_CONCEPTS:
            f.write(c + "\n")
    print(f"Concepts list saved to {txt_path}")


if __name__ == "__main__":
    main()

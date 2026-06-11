"""Generate filtered negative embedding bank v2 for C1 token loss.

NB-C strategy: automated filtering removes any concept with cosine sim > 0.75
to any of the 16 class prototypes, and any concept with sim > 0.70 to an
already-accepted entry (diversity filter).

This fixes the critical contamination in v1:
  - "school bus" → sim(bus)=0.926 → FILTERED OUT
  - "motorcycle rider" → sim(motorcycle)=0.914 → FILTERED OUT
  - "shrub", "bush", "grass" → sim(vegetation)>0.85 → FILTERED OUT
  - "person standing" → sim(pedestrian)=0.872 → FILTERED OUT

Output: panoptic/clip_prototypes/negative_bank_v2.pt
  - embeddings: (N, 768) L2-normalized  where all N pass both filters
  - concepts:   list of accepted concept strings
  - model_name: "ViT-L/14"
  - sim_to_nearest_class: (N,) max cosine sim to any class proto
  - nearest_class_idx:    (N,) argmax of above

Usage:
    python panoptic/scripts/generate_negative_bank_v2.py \
        --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
        --output panoptic/clip_prototypes/negative_bank_v2.pt \
        --device cuda
"""
import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

# ──────────────────────────────────────────────────────────────────────────────
# Candidate concept pool (~400 entries).
# Designed to be genuinely different from the 16 nuScenes classes.
# The sim-threshold filter is the authoritative gate — this list is deliberately
# broad and relies on filtering to remove near-duplicates.
# ──────────────────────────────────────────────────────────────────────────────
CANDIDATE_CONCEPTS = [
    # Street furniture
    "a fire hydrant in a driving scene",
    "a parking meter in a driving scene",
    "a bench in a driving scene",
    "a bus stop shelter in a driving scene",
    "a newspaper stand in a driving scene",
    "an ATM kiosk in a driving scene",
    "a vending machine in a driving scene",
    "a phone booth in a driving scene",
    "a post box in a driving scene",
    "a bike rack in a driving scene",
    "a public toilet in a driving scene",
    "a ticket machine in a driving scene",
    "a water fountain in a driving scene",
    "a recycling bin in a driving scene",
    "a waste bin in a driving scene",
    "a storage cabinet in a driving scene",
    "a kiosk in a driving scene",
    "a flower pot in a driving scene",
    "a planters box in a driving scene",
    "a street art installation in a driving scene",
    # Signs and signals
    "a traffic light in a driving scene",
    "a stop sign in a driving scene",
    "a yield sign in a driving scene",
    "a speed limit sign in a driving scene",
    "a no parking sign in a driving scene",
    "a street name sign in a driving scene",
    "a direction sign in a driving scene",
    "a construction warning sign in a driving scene",
    "a detour sign in a driving scene",
    "an information board in a driving scene",
    "a billboard in a driving scene",
    "an advertising hoarding in a driving scene",
    "a digital display board in a driving scene",
    "a road information panel in a driving scene",
    # Poles and infrastructure
    "a street lamp post in a driving scene",
    "a utility pole in a driving scene",
    "a power line tower in a driving scene",
    "a telephone pole in a driving scene",
    "a flag pole in a driving scene",
    "a lighting column in a driving scene",
    "a CCTV camera pole in a driving scene",
    "a parking sensor pole in a driving scene",
    "an antenna mast in a driving scene",
    "a communications tower in a driving scene",
    "a satellite dish in a driving scene",
    "a solar panel array in a driving scene",
    "a wind turbine in a driving scene",
    "a transformer box in a driving scene",
    "a junction box in a driving scene",
    # Buildings and structures
    "a building facade in a driving scene",
    "a storefront in a driving scene",
    "a house exterior in a driving scene",
    "a garage door in a driving scene",
    "a warehouse in a driving scene",
    "a factory building in a driving scene",
    "a parking structure in a driving scene",
    "a pedestrian overpass in a driving scene",
    "a highway overpass in a driving scene",
    "a bridge support in a driving scene",
    "a tunnel entrance in a driving scene",
    "a retaining wall in a driving scene",
    "a highway sound barrier in a driving scene",
    "a chain link fence in a driving scene",
    "a metal railing in a driving scene",
    "a wooden fence in a driving scene",
    "a stone wall in a driving scene",
    "a brick wall in a driving scene",
    "a glass facade in a driving scene",
    "a corrugated metal wall in a driving scene",
    "scaffolding on a building in a driving scene",
    "a fire escape ladder in a driving scene",
    "an awning in a driving scene",
    "a canopy in a driving scene",
    "a loading dock in a driving scene",
    # Animals
    "a dog in a driving scene",
    "a cat in a driving scene",
    "a bird in a driving scene",
    "a crow in a driving scene",
    "a pigeon in a driving scene",
    "a horse in a driving scene",
    "a deer in a driving scene",
    "a squirrel in a driving scene",
    "a rabbit in a driving scene",
    # Luggage, bags, carts
    "a shopping cart in a driving scene",
    "a luggage trolley in a driving scene",
    "a suitcase in a driving scene",
    "a backpack in a driving scene",
    "a delivery box in a driving scene",
    "a pallet in a driving scene",
    "a hand truck in a driving scene",
    "a cargo dolly in a driving scene",
    "a stroller in a driving scene",
    "a pram in a driving scene",
    "a wheelchair in a driving scene",
    "a mobility scooter in a driving scene",
    # Construction equipment (not in base)
    "a forklift in a driving scene",
    "a bulldozer in a driving scene",
    "an excavator in a driving scene",
    "a crane in a driving scene",
    "a road roller in a driving scene",
    "a tractor in a driving scene",
    "a street sweeper vehicle in a driving scene",
    "a snow plow vehicle in a driving scene",
    "an airport baggage cart in a driving scene",
    # Outdoor recreation
    "a golf cart in a driving scene",
    "a go-kart in a driving scene",
    "a riding lawn mower in a driving scene",
    "a dune buggy in a driving scene",
    "an ATV vehicle in a driving scene",
    "a snowmobile in a driving scene",
    # Specialty vehicles (not in 16 classes)
    "a horse-drawn carriage in a driving scene",
    "a rickshaw in a driving scene",
    "a food cart in a driving scene",
    "an ice cream cart in a driving scene",
    "a hot dog stand in a driving scene",
    "a market stall in a driving scene",
    # Shipping and logistics
    "a shipping container in a driving scene",
    "a cargo crate in a driving scene",
    "a dumpster in a driving scene",
    "a skip bin in a driving scene",
    "a portable storage unit in a driving scene",
    "a modular site office in a driving scene",
    "a porta-cabin in a driving scene",
    # Road features (not classification targets)
    "a manhole cover in a driving scene",
    "a drainage grate in a driving scene",
    "road lane markings in a driving scene",
    "a speed bump in a driving scene",
    "a pothole in a driving scene",
    "a road stud in a driving scene",
    "a raised pavement marker in a driving scene",
    "a rubber speed cushion in a driving scene",
    "a raised crosswalk in a driving scene",
    "a painted arrow in a driving scene",
    "a zebra crossing marking in a driving scene",
    # Sports and recreation areas
    "a basketball hoop in a driving scene",
    "a tennis court in a driving scene",
    "a football goal post in a driving scene",
    "a playground slide in a driving scene",
    "a swing set in a driving scene",
    "a park gazebo in a driving scene",
    "a picnic table in a driving scene",
    "an outdoor gym equipment in a driving scene",
    # Night and weather phenomena
    "a street light glow at night in a driving scene",
    "rain reflections on pavement in a driving scene",
    "a shadow cast on a road in a driving scene",
    "snow covering the ground in a driving scene",
    "a puddle on a road in a driving scene",
    "fog over a road in a driving scene",
    # Materials and textures
    "gravel in a driving scene",
    "sand in a driving scene",
    "cobblestones in a driving scene",
    "a tiled surface in a driving scene",
    "a wooden deck in a driving scene",
    "rubber matting in a driving scene",
    "a metal grate in a driving scene",
    "painted concrete in a driving scene",
    # Specific architectural elements
    "a doorway in a driving scene",
    "a window in a driving scene",
    "a staircase in a driving scene",
    "a ramp in a driving scene",
    "a balcony in a driving scene",
    "a rooftop in a driving scene",
    "a chimney in a driving scene",
    "a ventilation duct in a driving scene",
    "a downpipe in a driving scene",
    "a drain pipe in a driving scene",
    # Utility infrastructure
    "a water tower in a driving scene",
    "a gas pipe in a driving scene",
    "a water main cover in a driving scene",
    "an electrical substation in a driving scene",
    "a cable tray in a driving scene",
    "a conduit pipe in a driving scene",
    # Garden and landscaping
    "a garden bed in a driving scene",
    "a flower bed in a driving scene",
    "a topiary in a driving scene",
    "a hedge maze in a driving scene",
    "an ornamental pond in a driving scene",
    "a stone fountain in a driving scene",
    "a garden statue in a driving scene",
    "a garden gnome in a driving scene",
    # Food and beverage
    "an outdoor cafe table in a driving scene",
    "an umbrella over a cafe table in a driving scene",
    "a food kiosk in a driving scene",
    "a juice stand in a driving scene",
    # Retail and services
    "a shop window display in a driving scene",
    "a checkout counter in a driving scene",
    "a grocery bag in a driving scene",
    "a shopping bag in a driving scene",
    "a crate of produce in a driving scene",
    # Emergency equipment
    "a fire extinguisher in a driving scene",
    "a defibrillator cabinet in a driving scene",
    "a first aid kit box in a driving scene",
    "a safety cone in a driving scene",  # Note: diff from traffic_cone class
    "an emergency exit sign in a driving scene",
    "a warning light in a driving scene",
    "a reflective safety vest in a driving scene",
    "a hard hat in a driving scene",
    # Parking and vehicle-adjacent
    "a parking barrier arm in a driving scene",
    "a toll booth gate in a driving scene",
    "a car park ticket machine in a driving scene",
    "an EV charging station in a driving scene",
    "a petrol pump in a driving scene",
    "a car wash brush in a driving scene",
    "a squeegee bucket in a driving scene",
    "a windshield wiper blade in a driving scene",
    # Advertising and media
    "a poster in a driving scene",
    "a flyer in a driving scene",
    "a banner in a driving scene",
    "a street mural in a driving scene",
    "a neon sign in a driving scene",
    "a LED sign in a driving scene",
    # Office and workspace
    "a desk in a driving scene",
    "a chair in a driving scene",
    "a filing cabinet in a driving scene",
    "a whiteboard in a driving scene",
    "a projector screen in a driving scene",
    # Sports equipment in public spaces
    "a bicycle helmet in a driving scene",
    "a skateboard in a driving scene",
    "a scooter helmet in a driving scene",
    "roller skates in a driving scene",
    # Miscellaneous urban
    "a monument in a driving scene",
    "a public sculpture in a driving scene",
    "a sundial in a driving scene",
    "a clock tower in a driving scene",
    "a bell tower in a driving scene",
    "a fountain in a driving scene",
    "a pond in a driving scene",
    "a canal in a driving scene",
    "a river in a driving scene",
    "a lake in a driving scene",
    "a sky in a driving scene",
    "clouds in a driving scene",
    "a mountain in a driving scene",
    "a hill in a driving scene",
    "a field in a driving scene",
    # Medical
    "a stretcher in a driving scene",
    "a hospital gurney in a driving scene",
    # Cargo / industry
    "a pallet jack in a driving scene",
    "a hand cart in a driving scene",
    "a barrel in a driving scene",
    "a drum container in a driving scene",
    "a gas cylinder in a driving scene",
    "a propane tank in a driving scene",
    "a fire hose reel in a driving scene",
    "a ladder in a driving scene",
    "a scaffolding plank in a driving scene",
    "a workbench in a driving scene",
    "a tool box in a driving scene",
    "a generator in a driving scene",
    "a compressor in a driving scene",
    "a pump in a driving scene",
    # Boats and marine (unlikely near-class)
    "a boat in a driving scene",
    "a kayak in a driving scene",
    "a canoe in a driving scene",
    "a yacht in a driving scene",
    "a jet ski in a driving scene",
    # Air
    "an airplane in a driving scene",
    "a helicopter in a driving scene",
    "a drone in a driving scene",
    "a hot air balloon in a driving scene",
    # Farm
    "a tractor trailer in a driving scene",  # generic trailer-tractor combo, not truck
    "a combine harvester in a driving scene",
    "a hay bale in a driving scene",
    "a scarecrow in a driving scene",
    "a greenhouse in a driving scene",
    # Education and culture
    "a school building in a driving scene",
    "a library in a driving scene",
    "a museum in a driving scene",
    "a cinema in a driving scene",
    "a theater in a driving scene",
    "a church in a driving scene",
    "a mosque in a driving scene",
    "a temple in a driving scene",
    "a synagogue in a driving scene",
    # Tech
    "a server rack in a driving scene",
    "a CCTV camera in a driving scene",
    "a radar unit in a driving scene",
    "a LiDAR sensor in a driving scene",
    "a traffic camera in a driving scene",
    "an induction loop sensor in a driving scene",
    "a weather station in a driving scene",
    # Children's items
    "a baby stroller in a driving scene",
    "a toy car in a driving scene",
    "a bicycle trailer in a driving scene",
    # Food establishments
    "a restaurant in a driving scene",
    "a cafe in a driving scene",
    "a bakery in a driving scene",
    "a market in a driving scene",
    # Health
    "a pharmacy in a driving scene",
    "a hospital in a driving scene",
    "a clinic in a driving scene",
    # Services
    "a bank in a driving scene",
    "a post office in a driving scene",
    "a fire station in a driving scene",
    "a police station in a driving scene",
    "a hotel in a driving scene",
    "a motel in a driving scene",
    # Landscaping
    "a planted median in a driving scene",
    "a roundabout island in a driving scene",
    "an ornamental tree in a driving scene",
    "a palm tree in a driving scene",
    "a cactus in a driving scene",
    "a bamboo cluster in a driving scene",
    "a flower in a driving scene",
    "a wildflower patch in a driving scene",
    # Waste management
    "a recycling center in a driving scene",
    "a composting bin in a driving scene",
    "a waste truck in a driving scene",  # Note: different from garbage truck (similar to truck)
    # Sports
    "a baseball diamond in a driving scene",
    "a running track in a driving scene",
    "a soccer field in a driving scene",
    # Other
    "an outdoor market stall in a driving scene",
    "a street performer in a driving scene",
    "a busker in a driving scene",
    "a mime in a driving scene",
]

# Deduplicate preserving order
CANDIDATE_CONCEPTS = list(dict.fromkeys(CANDIDATE_CONCEPTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto_path", default="panoptic/clip_prototypes/text_prototypes_vitl14_224.pt",
                        help="Path to 16-class prototypes for filtering")
    parser.add_argument("--output", default="panoptic/clip_prototypes/negative_bank_v2.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="ViT-L/14")
    parser.add_argument("--class_sim_thresh", type=float, default=0.75,
                        help="Reject if sim to any class proto exceeds this")
    parser.add_argument("--diversity_thresh", type=float, default=0.70,
                        help="Reject if sim to any already-accepted entry exceeds this")
    args = parser.parse_args()

    import clip

    # Load class prototypes
    print(f"Loading class prototypes from {args.proto_path}...")
    proto_data = torch.load(args.proto_path, map_location="cpu")
    if isinstance(proto_data, dict):
        protos = proto_data["embeddings"].float()
        class_names = proto_data.get("class_names", [str(i) for i in range(16)])
    else:
        protos = proto_data.float()
        class_names = [str(i) for i in range(16)]
    protos = F.normalize(protos, dim=-1)  # (16, 768)
    print(f"  Loaded {protos.shape[0]} class prototypes")

    # Load CLIP
    print(f"Loading CLIP {args.model_name}...")
    model, _ = clip.load(args.model_name, device=args.device, jit=False)
    model.eval()

    # Encode all candidate concepts
    print(f"Encoding {len(CANDIDATE_CONCEPTS)} candidate concepts...")
    all_embs = []
    batch_size = 64
    for i in range(0, len(CANDIDATE_CONCEPTS), batch_size):
        batch = CANDIDATE_CONCEPTS[i:i + batch_size]
        tokens = clip.tokenize(batch).to(args.device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb.float(), dim=-1)
        all_embs.append(emb.cpu())
    all_embs = torch.cat(all_embs, dim=0)  # (N_cand, 768)
    print(f"  Encoded {all_embs.shape[0]} candidates")

    protos = protos.to("cpu")

    # Filter 1: class similarity
    sim_to_class = all_embs @ protos.T  # (N_cand, 16)
    max_sim_to_class, nearest_class = sim_to_class.max(dim=1)  # (N_cand,)

    rejected_class = (max_sim_to_class > args.class_sim_thresh).sum().item()
    print(f"\nFilter 1 (sim>{args.class_sim_thresh} to any class proto):")
    print(f"  Rejected: {rejected_class} / {len(CANDIDATE_CONCEPTS)}")

    # Print top-10 most similar to class protos (so we can see what's filtered)
    sorted_idx = max_sim_to_class.argsort(descending=True)
    print("  Top-10 most similar to class protos (filtered out first):")
    for ii in sorted_idx[:10]:
        cn = class_names[nearest_class[ii].item()] if nearest_class[ii].item() < len(class_names) else str(nearest_class[ii].item())
        print(f"    [{ii:3d}] sim={max_sim_to_class[ii]:.4f} → {cn:20s}  '{CANDIDATE_CONCEPTS[ii]}'")

    mask1 = max_sim_to_class <= args.class_sim_thresh
    candidates_pass1 = all_embs[mask1]
    concepts_pass1 = [CANDIDATE_CONCEPTS[i] for i in range(len(CANDIDATE_CONCEPTS)) if mask1[i]]
    sims_pass1 = max_sim_to_class[mask1]
    nearest_pass1 = nearest_class[mask1]
    print(f"  Remaining after filter 1: {len(concepts_pass1)}")

    # Filter 2: diversity (greedy)
    print(f"\nFilter 2 (diversity sim>{args.diversity_thresh}):")
    accepted_embs = []
    accepted_concepts = []
    accepted_sims = []
    accepted_nearest = []
    rejected_diversity = 0

    for i in range(len(concepts_pass1)):
        emb = candidates_pass1[i]
        if len(accepted_embs) == 0:
            accepted_embs.append(emb)
            accepted_concepts.append(concepts_pass1[i])
            accepted_sims.append(sims_pass1[i].item())
            accepted_nearest.append(nearest_pass1[i].item())
            continue
        bank = torch.stack(accepted_embs, dim=0)  # (k, 768)
        sim_to_bank = (bank @ emb).max().item()
        if sim_to_bank > args.diversity_thresh:
            rejected_diversity += 1
        else:
            accepted_embs.append(emb)
            accepted_concepts.append(concepts_pass1[i])
            accepted_sims.append(sims_pass1[i].item())
            accepted_nearest.append(nearest_pass1[i].item())

    print(f"  Rejected: {rejected_diversity}")
    print(f"  Final bank size: {len(accepted_concepts)}")

    if len(accepted_concepts) < 100:
        print(f"\n  WARNING: bank size {len(accepted_concepts)} < 100. Consider relaxing thresholds "
              f"or adding more candidate concepts.")

    # Build output tensors
    neg_bank_embs = torch.stack(accepted_embs, dim=0)  # (N, 768)
    sim_to_nearest = torch.tensor(accepted_sims)
    nearest_cls_idx = torch.tensor(accepted_nearest)

    print(f"\nFinal bank: {neg_bank_embs.shape[0]} entries")
    print(f"  Max sim to any class: {sim_to_nearest.max():.4f}")
    print(f"  Mean sim to nearest class: {sim_to_nearest.mean():.4f}")
    print(f"  Norm check: {neg_bank_embs.norm(dim=-1).mean():.4f}")

    # Print class distribution of nearest class
    print("\n  Nearest-class distribution (closest class, even though all < thresh):")
    for cls_i, cname in enumerate(class_names):
        count = (nearest_cls_idx == cls_i).sum().item()
        if count > 0:
            print(f"    [{cls_i:2d}] {cname:<25}: {count} entries")

    output = {
        "embeddings": neg_bank_embs,
        "concepts": accepted_concepts,
        "model_name": args.model_name,
        "sim_to_nearest_class": sim_to_nearest,
        "nearest_class_idx": nearest_cls_idx,
        "filter_thresholds": {
            "class_sim_thresh": args.class_sim_thresh,
            "diversity_thresh": args.diversity_thresh,
        },
    }
    torch.save(output, args.output)
    print(f"\nSaved to {args.output}")

    # Save concepts list as txt
    txt_path = args.output.replace(".pt", "_concepts.txt")
    with open(txt_path, "w") as f:
        for concept, sim, nidx in zip(accepted_concepts, accepted_sims, accepted_nearest):
            cname = class_names[nidx] if nidx < len(class_names) else str(nidx)
            f.write(f"{sim:.4f}  {cname:<25}  {concept}\n")
    print(f"Concepts list saved to {txt_path}")


if __name__ == "__main__":
    main()

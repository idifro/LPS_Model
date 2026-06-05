"""Generate all CLIP text prototypes needed for Job 1a (ViT-L/14 @224).

Generates:
  1. text_prototypes_vitl14_224.pt (16, 768) — single prompt
  2. text_prototypes_vitl14_224_multiprompt.pt (16, 10, 768) — 10 templates per class

Usage:
    python panoptic/scripts/generate_prototypes_224.py [--device cuda]
"""
import sys
import torch

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction vehicle",
    "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
    "driveable surface", "other flat", "sidewalk", "terrain", "manmade", "vegetation",
]

# 10 diverse prompt templates for T4 multi-prompt augmentation
TEMPLATES = [
    "a {} in a scene",
    "a photo of a {}",
    "a {} in an urban driving scene",
    "there is a {} in the scene",
    "a {} on the road",
    "a {} in a city",
    "this is a {} in a driving scenario",
    "a {} visible from a car",
    "an outdoor photo showing a {}",
    "a {} captured by a vehicle camera",
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import clip
    print("Loading CLIP ViT-L/14 ...", flush=True)
    model, _ = clip.load("ViT-L/14", device=args.device, jit=False)
    model.eval()

    out_dir = "panoptic/clip_prototypes"

    # 1. Single-prompt prototypes
    prompts = [TEMPLATES[0].format(name) for name in NUSCENES_CLASS_NAMES]
    tokens = clip.tokenize(prompts).to(args.device)
    with torch.no_grad():
        emb = model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu()
    torch.save({
        "embeddings": emb,
        "class_names": NUSCENES_CLASS_NAMES,
        "prompt_template": TEMPLATES[0],
        "model_name": "ViT-L/14",
    }, f"{out_dir}/text_prototypes_vitl14_224.pt")
    print(f"Saved single-prompt prototypes: {emb.shape}", flush=True)

    # 2. Multi-prompt prototypes (16, 10, 768)
    all_embs = []
    for template in TEMPLATES:
        prompts_t = [template.format(name) for name in NUSCENES_CLASS_NAMES]
        tokens_t = clip.tokenize(prompts_t).to(args.device)
        with torch.no_grad():
            emb_t = model.encode_text(tokens_t).float()
            emb_t = emb_t / emb_t.norm(dim=-1, keepdim=True)
        all_embs.append(emb_t.cpu())

    mp_emb = torch.stack(all_embs, dim=1)  # (16, 10, 768)
    torch.save({
        "embeddings": mp_emb,
        "class_names": NUSCENES_CLASS_NAMES,
        "templates": TEMPLATES,
        "model_name": "ViT-L/14",
    }, f"{out_dir}/text_prototypes_vitl14_224_multiprompt.pt")
    print(f"Saved multi-prompt prototypes: {mp_emb.shape}", flush=True)


if __name__ == "__main__":
    main()

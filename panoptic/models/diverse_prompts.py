"""Diverse text prompt generation for open-vocabulary training.

The key insight for open-vocab: if you train against ONLY "a car in a scene",
the model only learns to match that exact phrase. For open-vocab to work,
the model must learn the SEMANTIC CONCEPT of "car" from many diverse descriptions.

This module generates varied text prompts including:
1. Standard templates ("a photo of a {class}")  
2. Attribute-augmented ("a large red {class}")
3. Compositional ("a {class} near a building", "a {class} on the road")
4. Fine-grained descriptions ("an SUV", "a sedan", "a pickup truck" for "car")

At training time, we randomly sample prompts per class per batch → forces the
region features to align with the general CONCEPT, not a specific phrase.
"""
import random
from typing import Dict, List

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Standard prompt templates (used for all classes)
# ═══════════════════════════════════════════════════════════════════════════════

STANDARD_TEMPLATES = [
    "a {} in a scene",
    "a photo of a {}",
    "a {} in a driving scene",
    "there is a {} in this point cloud",
    "a {} on the road",
    "a 3D point cloud of a {}",
    "this is a {}",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Compositional context prompts (class + spatial relation + context)
# ═══════════════════════════════════════════════════════════════════════════════

SPATIAL_CONTEXTS = [
    "near a building",
    "on the road",
    "next to the sidewalk",
    "in a parking area",
    "at an intersection",
    "beside a tree",
    "in front of a building",
    "on the left side of the road",
    "that is partially occluded",
    "in the distance",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Fine-grained class expansions (class → list of more specific descriptions)
# These teach the model that "car" includes SUVs, sedans, etc.
# ═══════════════════════════════════════════════════════════════════════════════

FINE_GRAINED_EXPANSIONS: Dict[str, List[str]] = {
    "car": ["car", "sedan", "SUV", "hatchback", "sports car", "electric vehicle",
            "parked car", "moving car", "white car", "black car", "red car"],
    "truck": ["truck", "pickup truck", "delivery truck", "box truck", "flatbed truck",
              "large truck", "parked truck"],
    "bus": ["bus", "city bus", "school bus", "double-decker bus", "transit bus",
            "large bus", "parked bus"],
    "pedestrian": ["pedestrian", "person walking", "person standing", "person crossing",
                   "adult", "child", "person with backpack", "jogger"],
    "bicycle": ["bicycle", "bike", "mountain bike", "road bike", "parked bicycle",
                "bicycle rider"],
    "motorcycle": ["motorcycle", "motorbike", "scooter", "moped", "parked motorcycle"],
    "barrier": ["barrier", "road barrier", "concrete barrier", "metal barrier",
                "jersey barrier", "guardrail"],
    "traffic_cone": ["traffic cone", "orange cone", "construction cone", "safety cone"],
    "construction_vehicle": ["construction vehicle", "excavator", "bulldozer", "crane",
                             "forklift", "construction equipment"],
    "trailer": ["trailer", "cargo trailer", "semi-trailer", "attached trailer",
                "parked trailer"],
    "driveable_surface": ["driveable surface", "road", "road surface", "asphalt",
                          "paved road", "driving lane"],
    "sidewalk": ["sidewalk", "pavement", "walkway", "pedestrian path", "footpath"],
    "terrain": ["terrain", "grass", "dirt", "unpaved ground", "gravel", "soil"],
    "manmade": ["manmade structure", "building", "wall", "fence", "pole",
                "sign", "bridge structure"],
    "vegetation": ["vegetation", "tree", "bush", "shrub", "plants", "greenery",
                   "tall tree", "hedge"],
    "other_flat": ["other flat surface", "platform", "ground marking", "median"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# Novel / unseen class prompts (for open-vocab testing)
# These are classes NOT in the 16 training classes but may exist in the scene
# ═══════════════════════════════════════════════════════════════════════════════

NOVEL_CLASS_PROMPTS = {
    "pole": ["pole", "street light pole", "utility pole", "lamp post"],
    "bench": ["bench", "park bench", "bus stop bench", "seating"],
    "trash_can": ["trash can", "garbage bin", "recycling bin", "waste container"],
    "fire_hydrant": ["fire hydrant", "red fire hydrant", "hydrant"],
    "tree_trunk": ["tree trunk", "tree stem", "bare tree trunk"],
    "curb": ["curb", "road curb", "curb edge", "street curb"],
    "mailbox": ["mailbox", "post box", "letter box"],
    "bollard": ["bollard", "post", "street bollard"],
    "speed_bump": ["speed bump", "road hump", "traffic calming device"],
}


class DiversePromptGenerator:
    """Generates diverse text prompts for open-vocabulary CLIP supervision.
    
    Instead of training against 1 fixed prompt per class, this generates
    many variations. At each training step, a random prompt is sampled.
    This prevents the model from overfitting to specific phrase patterns
    and forces it to learn the underlying semantic concept.
    """
    
    def __init__(self, class_names, num_templates=7, compositional=True,
                 fine_grained=True):
        """
        Args:
            class_names: List of class names (16 nuScenes classes)
            num_templates: Number of standard templates to use
            compositional: Whether to include spatial context prompts
            fine_grained: Whether to include fine-grained class expansions
        """
        self.class_names = class_names
        self.num_templates = num_templates
        self.compositional = compositional
        self.fine_grained = fine_grained
        
        # Build prompt pool per class
        self.prompt_pools = self._build_prompt_pools()
    
    def _build_prompt_pools(self) -> Dict[int, List[str]]:
        """Build a pool of diverse prompts for each class."""
        pools = {}
        
        for idx, cls_name in enumerate(self.class_names):
            prompts = []
            
            # Standard templates
            for tmpl in STANDARD_TEMPLATES[:self.num_templates]:
                prompts.append(tmpl.format(cls_name))
            
            # Compositional (class + context)
            if self.compositional:
                for ctx in SPATIAL_CONTEXTS:
                    prompts.append(f"a {cls_name} {ctx}")
            
            # Fine-grained expansions
            if self.fine_grained and cls_name in FINE_GRAINED_EXPANSIONS:
                for variant in FINE_GRAINED_EXPANSIONS[cls_name]:
                    prompts.append(f"a {variant} in a scene")
                    prompts.append(f"a photo of a {variant}")
            
            pools[idx] = prompts
        
        return pools
    
    def sample_prompts(self, n_per_class=1) -> List[List[str]]:
        """Sample random prompts for each class.
        
        Args:
            n_per_class: Number of prompts to sample per class
        
        Returns:
            List of lists: prompts[class_idx] = [prompt1, prompt2, ...]
        """
        sampled = []
        for idx in range(len(self.class_names)):
            pool = self.prompt_pools[idx]
            chosen = random.sample(pool, min(n_per_class, len(pool)))
            sampled.append(chosen)
        return sampled
    
    def get_all_prompts(self) -> Dict[int, List[str]]:
        """Get the full prompt pool for all classes."""
        return self.prompt_pools
    
    def get_novel_prompts(self) -> Dict[str, List[str]]:
        """Get prompts for novel/unseen classes (for eval)."""
        return NOVEL_CLASS_PROMPTS


def encode_diverse_prototypes(clip_model, tokenizer, class_names, device,
                              num_templates=7, compositional=True, fine_grained=True):
    """Encode all diverse prompts for all classes using CLIP text encoder.
    
    Returns averaged prototypes (one per class) from diverse prompts.
    This gives MUCH better CLIP prototypes than single "a {class} in a scene".
    
    Args:
        clip_model: CLIP model (e.g., from open_clip or openai/CLIP)
        tokenizer: CLIP tokenizer
        class_names: List of 16 class names
        device: torch device
    
    Returns:
        prototypes: (num_classes, clip_dim) averaged and L2-normalized
    """
    generator = DiversePromptGenerator(
        class_names, num_templates=num_templates,
        compositional=compositional, fine_grained=fine_grained,
    )
    
    all_prototypes = []
    
    with torch.no_grad():
        for idx in range(len(class_names)):
            prompts = generator.prompt_pools[idx]
            # Encode all prompts for this class
            tokens = tokenizer(prompts).to(device)
            features = clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            # Average all prompt embeddings → single prototype
            avg_feature = features.mean(dim=0)
            avg_feature = avg_feature / avg_feature.norm()
            all_prototypes.append(avg_feature)
    
    prototypes = torch.stack(all_prototypes)  # (num_classes, clip_dim)
    return prototypes

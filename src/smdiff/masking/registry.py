from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MaskingSpec:
    id: str
    description: str
    available: bool = True
    notes: Optional[str] = None


MASKING_REGISTRY: Dict[str, MaskingSpec] = {
    # Note: In diffusion training we still apply time gating (t/T). The descriptions below
    # describe the *structural* masking unit; the final mask is gated by t/T.
    "random": MaskingSpec(
        id="random",
        description="Token-level masking: mask whole token (all channels) at random positions",
    ),
    "mixed": MaskingSpec(
        id="mixed",
        description="Randomly choose one masking strategy per batch (includes 'random')",
    ),
    "bar_all": MaskingSpec(
        id="bar_all",
        description="Dynamic bar-level masking: masks K bars where K is proportional to t/T",
        notes="Implementation masks attributes {pitch,duration,velocity,tempo}. K scales linearly with timestep.",
    ),
    "bar_attribute": MaskingSpec(
        id="bar_attribute",
        description="Dynamic attribute-level masking: masks K (bar, attribute) pairs where K is proportional to t/T",
        notes="Attribute is chosen from {pitch,duration,velocity,tempo}. K scales linearly with timestep.",
    ),
    
}


def resolve_masking_id(masking_id: str) -> MaskingSpec:
    key = masking_id.strip().lower()
    if key not in MASKING_REGISTRY:
        known = ", ".join(sorted(MASKING_REGISTRY.keys()))
        raise ValueError(f"Unknown masking id '{masking_id}'. Known: {known}")
    spec = MASKING_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Masking '{masking_id}' not available: {spec.notes or 'N/A'}")
    return spec

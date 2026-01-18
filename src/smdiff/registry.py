from dataclasses import dataclass
from typing import Callable, Dict, Optional

@dataclass(frozen=True)
class ModelSpec:
    id: str
    internal_model: str
    description: str
    factory: Optional[Callable] = None
    available: bool = True
    notes: Optional[str] = None


def _create_schmu_conv(H):
    """Factory for Conv_Transformer model."""
    from .models import ConVormer, AbsorbingDiffusion
    from torch.nn import DataParallel
    denoise_fn = ConVormer(H)
    denoise_fn = DataParallel(denoise_fn)
    return AbsorbingDiffusion(H, denoise_fn, H.codebook_size)


def _create_schmu_tx(H):
    """Factory for Transformer model."""
    from .models import Transformer, AbsorbingDiffusion
    from torch.nn import DataParallel
    denoise_fn = Transformer(H)
    denoise_fn = DataParallel(denoise_fn)
    return AbsorbingDiffusion(H, denoise_fn, H.codebook_size)


def _create_octuple_ddpm(H):
    """Factory for Octuple DDPM model."""
    from .models import Transformer, AbsorbingDiffusion
    from torch.nn import DataParallel
    denoise_fn = Transformer(H)
    denoise_fn = DataParallel(denoise_fn)
    return AbsorbingDiffusion(H, denoise_fn, H.codebook_size)


def _create_octuple_mask_ddpm(H):
    """Factory for Octuple DDPM with masking."""
    from .models import Transformer, AbsorbingDiffusion
    from torch.nn import DataParallel
    denoise_fn = Transformer(H)
    denoise_fn = DataParallel(denoise_fn)
    return AbsorbingDiffusion(H, denoise_fn, H.codebook_size)


def _create_musicbert_ddpm(H):
    """Factory for MusicBERT DDPM model."""
    from .models import MusicBERTDiffusion, AbsorbingDiffusion
    from torch.nn import DataParallel
    denoise_fn = MusicBERTDiffusion(H)
    denoise_fn = DataParallel(denoise_fn)
    return AbsorbingDiffusion(H, denoise_fn, H.codebook_size)


# Canonical model IDs and mapping to current internal model strings
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # SchmuBERT variants
    "schmu_conv": ModelSpec(
        id="schmu_conv",
        internal_model="conv_transformer",
        description="Conv_Transformer (SchmuBERT)",
        factory=_create_schmu_conv,
    ),
    "schmu_tx": ModelSpec(
        id="schmu_tx",
        internal_model="transformer",
        description="Transformer (SchmuBERT)",
        factory=_create_schmu_tx,
    ),

    # Octuple discrete diffusion
    # Using an 'octuple' prefix ensures existing code paths activate Octuple hparams
    "octuple_ddpm": ModelSpec(
        id="octuple_ddpm",
        internal_model="octuple_ddpm",
        description="Octuple MIDI + Transformer (discrete diffusion)",
        factory=_create_octuple_ddpm,
    ),
    "octuple_mask_ddpm": ModelSpec(
        id="octuple_mask_ddpm",
        internal_model="octuple_mask_ddpm",
        description="Octuple MIDI + Partial Masking + Transformer (discrete diffusion)",
        factory=_create_octuple_mask_ddpm,
    ),

    # MusicBERT + Transformer
    "musicbert_ddpm": ModelSpec(
        id="musicbert_ddpm",
        internal_model="musicbert_ddpm",
        description="MusicBERT + Transformer (discrete diffusion)",
        factory=_create_musicbert_ddpm,
    ),
}


def resolve_model_id(model_id: str) -> ModelSpec:
    key = model_id.strip().lower()
    if key not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model id '{model_id}'. Known: {known}")
    spec = MODEL_REGISTRY[key]
    if not spec.available:
        raise ValueError(
            f"Model '{model_id}' is not available yet. Notes: {spec.notes or 'N/A'}"
        )
    return spec

from dataclasses import dataclass
from typing import Callable, Dict, Optional

@dataclass(frozen=True)
class TokenizerSpec:
    id: str
    description: str
    factory: Optional[Callable] = None
    available: bool = True
    notes: Optional[str] = None


def _create_melody():
    from ..preprocessing.data import OneHotMelodyConverter
    return OneHotMelodyConverter()

def _create_trio():
    from ..preprocessing.data import POP909TrioConverter
    return POP909TrioConverter()

def _create_melody_octuple():
    from ..preprocessing.data import POP909OctupleMelodyConverter
    return POP909OctupleMelodyConverter()

def _create_trio_octuple():
    from ..preprocessing.data import POP909OctupleTrioConverter
    return POP909OctupleTrioConverter()


TOKENIZER_REGISTRY: Dict[str, TokenizerSpec] = {
    "melody": TokenizerSpec(
        id="melody",
        description="melody converter (1 track)",
        factory=_create_melody
    ),
    "trio": TokenizerSpec(
        id="trio",
        description="trio converter (3 tracks)",
        factory=_create_trio
    ),
    "melody_octuple": TokenizerSpec(
        id="melody_octuple",
        description="Octuple melody converter (1 track, 8-tuple tokens)",
        factory=_create_melody_octuple
    ),
    "trio_octuple": TokenizerSpec(
        id="trio_octuple",
        description="Octuple trio converter (3 tracks, 8-tuple tokens with program encoding)",
        factory=_create_trio_octuple
    )
}


def resolve_tokenizer_id(tokenizer_id: str) -> TokenizerSpec:
    key = tokenizer_id.strip().lower()
    if key not in TOKENIZER_REGISTRY:
        known = ", ".join(sorted(TOKENIZER_REGISTRY.keys()))
        raise ValueError(f"Unknown tokenizer id '{tokenizer_id}'. Known: {known}")
    spec = TOKENIZER_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Tokenizer '{tokenizer_id}' not available: {spec.notes or 'N/A'}")
    return spec

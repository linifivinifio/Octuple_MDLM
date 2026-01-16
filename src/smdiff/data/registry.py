from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    description: str
    dataset_path: str
    tracks: str
    bars: int
    notes: int
    tokenizer_id: str
    available: bool = True
    notes_txt: Optional[str] = None


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    # OneHot encoded datasets
    "pop909_melody": DatasetSpec(
        id="pop909_melody",
        description="POP909 melody combined .npy",
        dataset_path="data/POP909_melody.npy",
        tracks="melody",
        bars=64,
        notes=1024,
        tokenizer_id="melody",
    ),
    "pop909_trio": DatasetSpec(
        id="pop909_trio",
        description="POP909 trio combined .npy",
        dataset_path="data/POP909_trio.npy",
        tracks="trio",
        bars=64,
        notes=1024,
        tokenizer_id="trio",
        available=True,
        notes_txt="Generate via: python -m smdiff.cli.prepare_data --tokenizer_id trio",
    ),
    
    # Octuple encoded datasets
    "pop909_melody_octuple": DatasetSpec(
        id="pop909_melody_octuple",
        description="POP909 melody combined .npy (Octuple encoding)",
        dataset_path="data/POP909_melody_octuple.npy",
        tracks="melody_octuple",
        bars=64,
        notes=1024,
        tokenizer_id="melody_octuple",
        available=True,
        notes_txt="Generate via: python -m smdiff.cli.prepare_data --tokenizer_id melody_octuple",
    ),
    "pop909_trio_octuple": DatasetSpec(
        id="pop909_trio_octuple",
        description="POP909 trio combined .npy (Octuple encoding)",
        dataset_path="data/POP909_trio_octuple.npy",
        tracks="trio_octuple",
        bars=64,
        notes=1024,
        tokenizer_id="trio_octuple",
        available=True,
        notes_txt="Generate via: python -m smdiff.cli.prepare_data --tokenizer_id trio_octuple",
    ),
}


def resolve_dataset_id(dataset_id: str) -> DatasetSpec:
    key = dataset_id.strip().lower()
    if key not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset id '{dataset_id}'. Known: {known}")
    spec = DATASET_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Dataset '{dataset_id}' not available: {spec.notes_txt or 'N/A'}")
    return spec


def apply_dataset_to_config(cfg: Dict, dataset_id: str) -> Dict:
    spec = resolve_dataset_id(dataset_id)
    updated = dict(cfg)
    updated.update({
        "dataset_path": spec.dataset_path,
        "tracks": spec.tracks,
        "bars": spec.bars,
        "NOTES": spec.notes,
        "tokenizer_id": spec.tokenizer_id,
    })
    return updated

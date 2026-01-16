from typing import Dict, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    id: str
    description: str
    available: bool = True
    notes: Optional[str] = None

TASK_REGISTRY: Dict[str, TaskSpec] = {
    "uncond": TaskSpec(id="uncond", description="Unconditional generation"),
    "infill": TaskSpec(id="infill", description="Infilling / masked completion"),
}


def resolve_task_id(task_id: str) -> TaskSpec:
    key = task_id.strip().lower()
    if key not in TASK_REGISTRY:
        known = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task id '{task_id}'. Known: {known}")
    spec = TASK_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Task '{task_id}' not available: {spec.notes or 'N/A'}")
    return spec

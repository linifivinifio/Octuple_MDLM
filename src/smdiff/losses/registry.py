from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LossSpec:
    id: str
    description: str
    available: bool = True
    notes: Optional[str] = None


LOSS_REGISTRY: Dict[str, LossSpec] = {
    "mmd_fmd_loss": LossSpec(
        id="mmd_fmd_loss",
        description="Cross entropy + multi-kernel MMD + minibatch Frechet-style regularization",
        notes="Default loss. Regularizers are weak by default.",
    ),
    "mmd_loss": LossSpec(
        id="mmd_loss",
        description="Pure multi-kernel MMD objective (raw MMD only; CE/FMD excluded from optimization loss)",
        notes="Uses the same feature construction as mmd_fmd_loss but optimizes only raw mmd_loss.",
    ),
    "strict_fmd": LossSpec(
        id="strict_fmd",
        description="Pure strict Fr\u00e9chet objective with fixed training-data reference mean/covariance in token-proxy feature space",
        notes="Optimizes only strict Fr\u00e9chet distance (no CE/MMD terms).",
    ),
    "plain_ce_loss": LossSpec(
        id="plain_ce_loss",
        description="Legacy plain CE objective (same behavior as historical reweighted_elbo path)",
    ),
    "elbo": LossSpec(
        id="elbo",
        description="Variational bound objective",
    ),
    "mlm": LossSpec(
        id="mlm",
        description="Masked language model objective",
    ),
    "reweighted_elbo": LossSpec(
        id="reweighted_elbo",
        description="Legacy id kept for backward compatibility",
        notes="Canonicalizes to plain_ce_loss behavior.",
    ),
}


LOSS_ALIASES: Dict[str, str] = {
    "plain_CE_loss": "plain_ce_loss",
    "plain_ce": "plain_ce_loss",
    "plain_celoss": "plain_ce_loss",
    "mmd": "mmd_loss",
    "default": "mmd_fmd_loss",
}


def resolve_loss_id(loss_id: str) -> LossSpec:
    key = loss_id.strip()
    if key in LOSS_ALIASES:
        key = LOSS_ALIASES[key]
    else:
        key = LOSS_ALIASES.get(key.lower(), key.lower())

    if key not in LOSS_REGISTRY:
        known = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss id '{loss_id}'. Known: {known}")
    spec = LOSS_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Loss '{loss_id}' not available: {spec.notes or 'N/A'}")
    return spec

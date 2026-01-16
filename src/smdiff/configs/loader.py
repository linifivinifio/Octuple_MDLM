import os
from copy import deepcopy
from typing import Dict, Iterable, Optional

import yaml

from src.smdiff.registry import resolve_model_id


def _deep_update(base: Dict, overrides: Dict) -> Dict:
    out = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_set_overrides(pairs: Iterable[str]) -> Dict:
    overrides: Dict = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid override '{p}'. Use key=value")
        key, val = p.split("=", 1)
        # Try to parse scalars/lists/dicts via YAML
        overrides[key] = yaml.safe_load(val)
    return overrides


def load_config(model_id: str,
                config_path: Optional[str] = None,
                set_overrides: Optional[Iterable[str]] = None,
                base_path: Optional[str] = None,
                models_path: Optional[str] = None) -> Dict:
    """
    Merge configs in this order (later wins):
    1) base.yaml
    2) models.yaml entry for model_id
    3) user config file (optional)
    4) --set key=value overrides (optional)
    """
    spec = resolve_model_id(model_id)

    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "base.yaml")
    if models_path is None:
        models_path = os.path.join(os.path.dirname(__file__), "models.yaml")

    base_cfg = _load_yaml(base_path)
    models_cfg = _load_yaml(models_path)
    model_cfg = models_cfg.get(spec.id, {}) if isinstance(models_cfg, dict) else {}

    cfg = _deep_update(base_cfg, model_cfg)

    user_cfg = _load_yaml(config_path) if config_path else {}
    cfg = _deep_update(cfg, user_cfg)

    if set_overrides:
        cfg = _deep_update(cfg, _parse_set_overrides(set_overrides))

    return cfg

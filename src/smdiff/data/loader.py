import os
from typing import Dict
import numpy as np

from .base import SimpleNpyDataset
from .registry import resolve_dataset_id

def load_dataset(cfg: Dict):
    """Load dataset based on dataset_id or dataset_path in cfg."""
    dataset_id = cfg.get("dataset_id")
    tokenizer_id = None

    # 1. Resolve ID and Path from Registry
    if dataset_id:
        spec = resolve_dataset_id(dataset_id)
        dataset_path = spec.dataset_path
        seq_len = spec.notes
        tokenizer_id = spec.tokenizer_id  # <--- Extract ID here
    else:
        # Fallback for manual paths
        dataset_path = cfg.get("dataset_path")
        seq_len = cfg.get("NOTES")
        # Allow passing tokenizer_id manually in config if not using registry
        tokenizer_id = cfg.get("tokenizer_id", "")

    if dataset_path is None:
        raise ValueError("dataset_path or dataset_id must be provided")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading dataset from {dataset_path} (Tokenizer: {tokenizer_id})...")
    data = np.load(dataset_path, allow_pickle=True)
    
    # 2. Pass tokenizer_id to the dataset
    dataset = SimpleNpyDataset(data, seq_len, tokenizer_id=tokenizer_id)
    
    return dataset
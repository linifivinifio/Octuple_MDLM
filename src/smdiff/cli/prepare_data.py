"""Prepare POP909 datasets into NumPy caches (OneHot and Octuple formats)."""
import argparse
import os
import sys
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from note_seq import midi_to_note_sequence
from tqdm import tqdm

from ..preprocessing import (
    OneHotMelodyConverter, 
    POP909TrioConverter,
    POP909OctupleMelodyConverter,
    POP909OctupleTrioConverter
)

# Ensure repository root is on sys.path so top-level packages like 'hparams' resolve
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also ensure 'src' is on sys.path so 'smdiff' package resolves when running by path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _make_converter(tokenizer_id: str, bars: int, max_t_per_ns: int, strict_tempo: bool = False):

    # Create converter with proper parameters based on tokenizer_id
    if tokenizer_id == "melody":
        return OneHotMelodyConverter(
            slice_bars=bars,
            max_tensors_per_notesequence=max_t_per_ns,
            gap_bars=None, # type: ignore
            presplit_on_time_changes=False,
            strict_tempo=strict_tempo,
            instrument=0,  # Filter to instrument 0 (MELODY track)
        )
    elif tokenizer_id == "trio":
        return POP909TrioConverter(
            slice_bars=bars,
            max_tensors_per_notesequence=max_t_per_ns,
            gap_bars=None,
            presplit_on_time_changes=False,
            strict_tempo=strict_tempo,
        )
    elif tokenizer_id == "melody_octuple":
        return POP909OctupleMelodyConverter(
            slice_bars=bars,
            max_tensors_per_notesequence=max_t_per_ns,
            gap_bars=None,
            presplit_on_time_changes=False,
            strict_tempo=strict_tempo,
        )
    elif tokenizer_id == "trio_octuple":
        return POP909OctupleTrioConverter(
            slice_bars=bars,
            max_tensors_per_notesequence=max_t_per_ns,
            gap_bars=None,
            presplit_on_time_changes=False,
            strict_tempo=strict_tempo,
        )
    else:
        raise ValueError(f"Tokenizer '{tokenizer_id}' not supported for MIDI extraction.")


def process_midi_file(args):
    """Worker function for processing a single MIDI file."""
    midi_path, tokenizer_id, bars, max_t_per_ns, strict_tempo = args
    converter = _make_converter(tokenizer_id, bars, max_t_per_ns, strict_tempo)

    result = []
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with open(midi_path, "rb") as f:
                content = f.read()
            ns = midi_to_note_sequence(content)
            tensors = converter.to_tensors(ns).outputs
            result = list(tensors)
            if not result:
                print(f"No sequences extracted from {midi_path}")
            # Log chosen tempo when sanitizer is active (trio converter)
            chosen = getattr(converter, "last_chosen_tempo", None)
            if chosen is not None:
                print(f"Tempo chosen for {midi_path}: {chosen:.2f} qpm")
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        pass
    return result


def chunk_sequences(tensors, block_size=1024):
    """
    Splits long sequences into chunks of size `block_size`.
    Shorter leftovers are kept as-is (to be padded at runtime).
    """
    chunked_data = []
    
    for x in tensors:
        length = x.shape[0]
        
        # Case 1: Sequence fits in one block
        if length <= block_size:
            chunked_data.append(x)
            continue
            
        # Case 2: Sequence needs splitting
        # We use a non-overlapping stride equal to block_size
        num_chunks = int(np.ceil(length / block_size))
        
        for i in range(num_chunks):
            start = i * block_size
            end = min(start + block_size, length)
            
            chunk = x[start:end]
            
            # Safety check for empty chunks
            if chunk.shape[0] > 256: # threshold to have meaningful learning samples
                chunked_data.append(chunk)
                
    return chunked_data


def load_dataset(root_dir: str,
                 tokenizer_id: str = "melody",
                 bars: int = 64,
                 max_tensors_per_ns: int = 5,
                 cache_path: str | None = None,
                 limit: int = 0,
                 num_workers: int | None = None,
                 strict_tempo: bool = False):
    """
    Load and process a dataset of MIDI files into a NumPy cache.

    Args:
        root_dir: Directory containing MIDI files (searched recursively).
        tokenizer_id: 'melody' or 'trio'.
        bars: Number of bars per slice.
        max_tensors_per_ns: Max segments extracted per MIDI file.
        cache_path: Optional .npy cache path for saving/loading.
        limit: Limit number of files processed (0 = no limit).
        num_workers: Worker processes (defaults to min(40, cpu_count)).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        return np.load(cache_path, allow_pickle=True)

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    exclude_dirs = {"versions"}
    all_midis = [
        m for m in sorted(root_path.rglob("*.mid"))
        if not any(part.lower() in exclude_dirs for part in m.parts)
    ]
    if limit > 0:
        all_midis = all_midis[:limit]

    print(f"Processing {len(all_midis)} MIDI files from {root_dir} with tokenizer={tokenizer_id}, bars={bars}")

    worker_args = [(str(m), tokenizer_id, bars, max_tensors_per_ns, strict_tempo) for m in all_midis]
    if num_workers is None:
        num_workers = min(40, os.cpu_count() or 4)

    result: list[np.ndarray] = []
    with Pool(num_workers) as pool:
        for file_res in tqdm(pool.imap(process_midi_file, worker_args), total=len(worker_args)):
            result.extend(file_res)

    print(f"Extracted {len(result)} sequences.")
    
    # We enforce a maximum length of 1024 here.
    # Longer sequences are split. Shorter sequences are left alone (padded at runtime).
    # This ensures Bar 0 is always the start of the first chunk.
    print("Chunking sequences to max length 1024...")
    result = chunk_sequences(result, block_size=1024)
    
    print(f"Final dataset size: {len(result)} chunks.")
    # ------------------------------

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving to {cache_path}...")
        np.save(cache_path, np.array(result, dtype=object))

    return np.array(result, dtype=object)


def main():
    parser = argparse.ArgumentParser(description="Prepare POP909 datasets (Standard and Octuple formats)")
    parser.add_argument("--root_dir", type=str, default="data/train/POP909", help="Root directory of the dataset")
    parser.add_argument("--tokenizer_id", type=str, default="melody",
                        choices=["melody", "trio", "melody_octuple", "trio_octuple"],
                        help="Tokenizer to use: OneHot (melody/trio) or Octuple (melody/trio)")
    parser.add_argument("--target", type=str, default=None, help="Output .npy file (defaults per tokenizer)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process")
    parser.add_argument("--bars", type=int, default=64, help="Sequence length in bars")
    parser.add_argument("--max_tensors_per_ns", type=int, default=5, help="Max tensors extracted per MIDI")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--strict_tempo", action="store_true", help="Do not sanitize tempo; respect original tempo curve")

    args = parser.parse_args()

    # Sensible defaults for targets
    if args.target is None:
        if args.tokenizer_id == "trio":
            args.target = "data/POP909_trio.npy"
        elif args.tokenizer_id == "trio_octuple":
            args.target = "data/POP909_trio_octuple.npy"
        elif args.tokenizer_id == "melody_octuple":
            args.target = "data/POP909_melody_octuple.npy"
        else:  # melody
            args.target = "data/POP909_melody.npy"

    load_dataset(
        root_dir=args.root_dir,
        tokenizer_id=args.tokenizer_id,
        bars=args.bars,
        max_tensors_per_ns=args.max_tensors_per_ns,
        cache_path=args.target,
        limit=args.limit,
        num_workers=args.workers,
        strict_tempo=args.strict_tempo,
    )


if __name__ == "__main__":
    main()

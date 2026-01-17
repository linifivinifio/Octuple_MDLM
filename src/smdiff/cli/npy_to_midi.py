import os
import sys
import glob
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from note_seq import note_sequence_to_midi_file

# Ensure repository root is on sys.path so top-level packages like 'hparams' resolve
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also ensure 'src' is on sys.path so 'smdiff' package resolves when running by path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# Import your existing utilities
from src.smdiff.utils.log_utils import samples_2_noteseq
from src.smdiff.tokenizers.registry import resolve_tokenizer_id


def load_tokenizer_id(run_dir):
    """Try to infer tokenizer_id from hparams.yaml or config.yaml"""
    # Check standard locations
    paths = [
        os.path.join(run_dir, "configs", "config.yaml")
    ]
    
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check for standard keys
                if 'tokenizer_id' in config:
                    return config['tokenizer_id']
                if 'tracks' in config:
                    # 'tracks' is usually the tokenizer id (e.g. 'trio', 'melody')
                    return config['tracks']
            except Exception as e:
                print(f"Warning: Failed to parse {p}: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Convert .npy sample files to MIDI")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory (e.g. runs/my_model_trio)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer ID (e.g. trio, melody, trio_octuple)")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of steps to process")
    args = parser.parse_args()

    samples_dir = os.path.join(args.run_dir, "samples")
    if not os.path.exists(samples_dir):
        print(f"Error: No samples folder found at {samples_dir}")
        return

    # 1. Determine Tokenizer
    tokenizer_id = args.tokenizer
    if tokenizer_id is None:
        tokenizer_id = load_tokenizer_id(args.run_dir)
        if tokenizer_id is None:
            # Fallback guessing based on folder name
            if 'trio' in args.run_dir: tokenizer_id = 'trio'
            elif 'melody' in args.run_dir: tokenizer_id = 'melody'
            else:
                print("Error: Could not infer tokenizer. Please use --tokenizer [trio|melody]")
                return
    
    if "_onehot" in tokenizer_id:
        tokenizer_id = tokenizer_id.split("_onehot")[0]
    print(f"Using Tokenizer: {tokenizer_id}")

    # 2. Setup Output Directory
    midi_dir = os.path.join(samples_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)

    # 3. Find files
    # Matches samples_100.npy, samples_200.npy etc.
    npy_files = sorted(glob.glob(os.path.join(samples_dir, "*.npy")))
    
    if args.max_files:
        npy_files = npy_files[-args.max_files:]

    print(f"Found {len(npy_files)} sample files.")

    # 4. Process
    for npy_path in tqdm(npy_files, desc="Converting"):
        filename = os.path.basename(npy_path)
        # Extract step number (e.g., samples_200.npy -> 200)
        step_str = filename.replace("samples_", "").replace(".npy", "")
        
        try:
            # Load Samples
            samples = np.load(npy_path)
            
            print("Shape: ", samples.shape)
            
            # Convert to NoteSequences
            # samples_2_noteseq handles the shape and registry lookup
            note_seqs = samples_2_noteseq(samples, tokenizer_id)
            
            for i, ns in enumerate(note_seqs):
                # We skip empty sequences to save space
                if len(ns.notes) == 0:
                    continue
                
                out_name = f"step_{step_str}_sample_{i}.mid"
                out_path = os.path.join(midi_dir, out_name)
                note_sequence_to_midi_file(ns, out_path)
                
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

    print(f"Done! MIDI files saved to: {midi_dir}")

if __name__ == "__main__":
    main()
import argparse
import os
import json
import numpy as np
import torch
import sys
from tqdm import tqdm

# Ensure repository root is on sys.path so top-level packages like 'hparams' resolve
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also ensure 'src' is on sys.path so 'smdiff' package resolves when running by path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from hparams.set_up_hparams import get_sampler_hparams
from smdiff.utils.sampler_utils import get_sampler, save_generated_samples, ns_to_np
from smdiff.utils.log_utils import load_model, config_log, log
from smdiff.metrics.unconditional import evaluate_unconditional
from smdiff.metrics.infilling import evaluate_infilling
from smdiff.preprocessing.data import POP909OctupleTrioConverter
from smdiff.configs.loader import load_config
from smdiff.masking import resolve_masking_id
from note_seq import midi_file_to_note_sequence
from smdiff.cluster import get_scratch_dir


def clean_sample(s, eos_token=None):
    """Remove padding and truncate at first EOS row when provided."""
    # Keep only rows where there is no subtoken that is -1
    cleaned = s[~(s == -1).any(axis=1)]

    if eos_token is not None and cleaned.ndim == 2 and cleaned.shape[1] == len(eos_token):
        eos_rows = np.where(np.all(cleaned == eos_token, axis=1))[0]
        if len(eos_rows) > 0:
            cleaned = cleaned[:eos_rows[0]]

    return cleaned


def load_octuple_dataset(path):
    log(f"Loading dataset from {path}...")
    try:
        data = np.load(path, allow_pickle=True)
        # Convert to list of arrays if it's an object array
        if data.dtype == object or data.dtype.type is np.str_:
            loaded_data = []
            for x in data:
                # If the item is a string/path, load the actual chunk
                if isinstance(x, (str, np.str_)):
                    x = np.load(x, allow_pickle=True)
                loaded_data.append(x)
            return loaded_data
        return data
    except Exception as e:
        log(f"Error loading dataset: {e}")
        return []

def resolve_model_path(load_dir, model_arg, h_model_id):
    """Helper to determine model ID/path"""
    # If model arg is provided, use it
    if model_arg:
        return model_arg
    # If not, try to use the H params model_id
    if h_model_id:
        return h_model_id
    # If neither, infer from load_dir name (fragile but possible)
    return os.path.basename(os.path.normpath(load_dir))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Octuple Models")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--task", type=str, required=True, choices=["uncond", "infill"], help="Task to evaluate")
    parser.add_argument("--model", type=str, required=True, default=None, help="Model ID (e.g. musicbert_ddpm_trio_octuple)")
    parser.add_argument("--input_midi_dir", type=str, help="Directory of input MIDIs for infill task")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples (uncond)")
    parser.add_argument("--n_midis", type=int, default=None, help="Limit number of MIDI files for infilling")
    parser.add_argument("--load_step", type=int, default=0, help="Checkpoint step to load (0 for best/latest)")
    parser.add_argument("--strategy", type=str, default=None, help="Optional masking strategy for infill mask construction")
    parser.add_argument("--mask_token_start", type=int, default=256, help="Start token index for masking")
    parser.add_argument("--mask_token_end", type=int, default=512, help="End token index for masking")
    parser.add_argument("--preserve_structure", action="store_true", 
                        help="If set, leaves Bar (0) and Position (1) tokens unmasked in the target range, masking only musical content.")
    parser.add_argument("--compute_fmd", action="store_true", help="Skip generation and evaluate FMD on existing samples")
    parser.add_argument("--eos", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable EOS token handling (--eos or --no-eos)")
    args = parser.parse_args()

    if args.strategy:
        resolve_masking_id(args.strategy)
    
    
    # 1. Prepare Output Directories
    metrics_dir = os.path.join(args.load_dir, "metrics")
    samples_dir = os.path.join(metrics_dir, f"{args.task}_{args.load_step if args.load_step != 0 else 'best'}")
    os.makedirs(samples_dir, exist_ok=True)
    
    # configure metrics logging
    config_log(metrics_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    if args.compute_fmd:
        log("FMD evaluation mode enabled. Skipping model loading and generation.")
        # Perform FMD computation directly
        try:
            from frechet_music_distance import FrechetMusicDistance
        except ImportError:
            log("frechet-music-distance not installed. Please install it first.")
            return

        # Prepare reference directory (temporary flat dir of symlinks for test POS909)
        import tempfile
        import shutil
        test_dir = os.path.join(_REPO_ROOT, "data", "test", "POP909")
        temp_ref_dir = tempfile.mkdtemp(prefix="fmd_ref_")
        
        # Creating flat directory with symlinks to reference MIDI files
        log(f"Creating temporary reference directory at {temp_ref_dir}...")
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith('.mid') or f.lower().endswith('.midi'):
                    src = os.path.join(root, f)
                    # Use unique name in case of overlaps
                    dst = os.path.join(temp_ref_dir, f"{os.path.basename(root)}_{f}")
                    os.symlink(src, dst)
        
        # Determine path to existing metrics JSON
        metrics_path = os.path.join(metrics_dir, f"metrics_{args.task}_{args.load_step if args.load_step != 0 else 'best'}.json")
        if not os.path.exists(metrics_path):
            log(f"Metrics file {metrics_path} does not exist. Please run standard evaluation first.")
            shutil.rmtree(temp_ref_dir)
            return
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        log("Initializing FrechetMusicDistance (CLaMP2)...")
        fmd_metric = FrechetMusicDistance(feature_extractor='clamp2', gaussian_estimator='mle', verbose=True)

        log(f"Computing FMD score against {samples_dir}...")
        try:
            score = fmd_metric.score(
                reference_path=temp_ref_dir,
                test_path=samples_dir
            )
            log(f"Computed FMD: {score}")
            metrics["frechet_music_distance"] = float(score)
            
            # Save updated metrics
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            log(f"Updated {metrics_path} with Frechet Music Distance.")
        except Exception as e:
            log(f"FMD computation failed: {e}")
        finally:
            shutil.rmtree(temp_ref_dir)
            log("Cleaned up temporary reference directory.")

        return

    model_id = args.model
    log(f"Using Model ID: {model_id}")

    # Set up H
    # We need to construct argv for get_sampler_hparams to work correctly 
    # as it parses command line args internally
    prev_argv = sys.argv
    sys.argv = [
        sys.argv[0],
        "--model", model_id,
        "--load_dir", args.load_dir,
        "--bars", "64",
        "--batch_size", str(args.batch_size),
        "--tracks", "trio_octuple"
    ]

    if args.strategy:
        sys.argv += ["--masking_strategy", args.strategy]

    if args.eos is not None:
        sys.argv += ["--eos" if args.eos else "--no-eos"]
    
    try:
        H = get_sampler_hparams('sample')
    except Exception as e:
        log(f"Error setting up hparams: {e}")
        raise
    finally:
        sys.argv = prev_argv

    # Override H params from args/context if needed
    H.load_dir = args.load_dir 
    H.masking_strategy = args.strategy
    model_cfg = load_config(model_id)
    H.hierarchical_masking = load_config(model_id).get("hierarchical_masking", {})
    H.octuple_mdlm = model_cfg.get("octuple_mdlm", {})
    eos_token = np.array(H.codebook_size, dtype=np.int64) if getattr(H, 'eos', False) else None
    # Ensure Octuple masking strategy if relevant (usually loaded from H)
    
    # 2. Load Model
    log("Loading model...")
    sampler = get_sampler(H).to(device)
    
    # Check fallback paths in scratch
    fallback_dirs = []
    # If using a relative load_dir (e.g. runs/my_model), check scratch/user/runs/my_model
    scratch_dir = get_scratch_dir()
    if scratch_dir and os.path.exists(scratch_dir):
        fallback_dirs.append(os.path.join(scratch_dir, args.load_dir))
    
    # Load weights (EMA default)
    try:
        # load_model(sampler, "ema", 0, args.load_dir) 
        # Note: 0 usually means "best" or implicitly handled if load_step is 0
        load_model(sampler, "ema", args.load_step, args.load_dir, fallback_dirs=fallback_dirs, strict=False) 
    except Exception as e:
        log(f"Failed to load EMA, trying standard model: {e}")
        load_model(sampler, "model", args.load_step, args.load_dir, fallback_dirs=fallback_dirs, strict=False)

    sampler.eval()
    

    
    # 4. Load Ground Truth Data (for metrics)
    train_data_path = os.path.join(_REPO_ROOT, "data", "POP909_trio_octuple.npy")
    train_samples = load_octuple_dataset(train_data_path)
    
    generated_samples = []
    original_samples_for_metrics = [] # Corresponding GT for infilling
    
    # 5. Execute Task
    if args.task == "uncond":
        log(f"Generating {args.n_samples} unconditional samples...")
        
        n_batches = int(np.ceil(args.n_samples / args.batch_size))
        all_samples = []
        
        for _ in tqdm(range(n_batches), desc="Sampling"):
            curr_batch = min(args.batch_size, args.n_samples - len(all_samples))
            if curr_batch <= 0: break
            
            # Unconditional sampling: pass x_T=None (will be all MASK)
            # Sample returns (B, T, 8) tensor or numpy
            samples = sampler.sample(sample_steps=H.sample_steps, B=curr_batch)
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()
            
            # remove -1 tokens
            cleaned_batch = [clean_sample(s, eos_token=eos_token) for s in samples]
            
            all_samples.extend(cleaned_batch)
            
        generated_samples = all_samples[:args.n_samples]
        
        # Save Samples to MIDI
        log("Saving samples...")
        save_generated_samples(generated_samples, "trio_octuple", samples_dir, prefix="uncond")
        
        # Calculate Metrics
        log("Calculating metrics...")
        metrics = evaluate_unconditional(generated_samples, train_samples, is_octuple=True)
        
    elif args.task == "infill":
        if not args.input_midi_dir:
            raise ValueError("--input_midi_dir required for infilling")
            
        log(f"Infilling from MIDIs in {args.input_midi_dir}...")
        
        midi_files = []
        for root, dirs, files in os.walk(args.input_midi_dir):
            for f in files:
                if f.lower().endswith('.mid') or f.lower().endswith('.midi'):
                    midi_files.append(os.path.join(root, f))
        
        # Sort to ensure deterministic order
        midi_files.sort()
        
        if args.n_midis is not None:
            midi_files = midi_files[:args.n_midis]
            
        log(f"Found {len(midi_files)} MIDI files (limit: {args.n_midis}).")
        
        converter = POP909OctupleTrioConverter(slice_bars=64, presplit_on_time_changes=False,
            strict_tempo=False, gap_bars=None) # Ensure max length covers needed range
        
        # Token-based masking
        mask_token_start = args.mask_token_start
        mask_token_end = args.mask_token_end
        log(f"Masking Tokens: {mask_token_start} - {mask_token_end} (Range: {mask_token_end - mask_token_start})")
        log(f"Structure Preservation: {'ENABLED' if args.preserve_structure else 'DISABLED'}")
        
        # Mask ID for Octuple is usually a vector (one per channel)
        if hasattr(sampler, 'mask_id'):
            mask_id = sampler.mask_id
            if isinstance(mask_id, torch.Tensor):
                mask_id = mask_id.cpu().numpy()
            mask_token_id = mask_id
        else:
            mask_token_id = H.codebook_size

        count = 0
        for midi_path in tqdm(midi_files, desc="Infilling"):
            try:
                ns = midi_file_to_note_sequence(midi_path)
                tensors = converter.to_tensors(ns)
                if not tensors.outputs:
                    log("Error converting to npy during conversion to octuple format")
                    continue
                
                original_tokens = tensors.outputs[0] # Take first slice
                
                # Check minimum length for structure
                if original_tokens.ndim != 2 or original_tokens.shape[1] < 8:
                    continue
                
                # Truncate to model block size (1024) to avoid "model block size exhausted"
                if len(original_tokens) > H.NOTES:
                     original_tokens = original_tokens[:H.NOTES]

                # VALIDATION: Check if sequence is long enough to contain the mask region
                if len(original_tokens) <= mask_token_end:
                    # Sequence too short to evaluate this mask range
                    log(f"Skipping {midi_path}: length {len(original_tokens)} < mask_end {mask_token_end}")
                    continue
                    
                # Prepare Masked Input
                # Copy original
                masked_input = original_tokens.copy()
                
                if args.strategy == "hierarchical" and hasattr(sampler, "build_hierarchical_mask"):
                    constrain_window = True
                    if hasattr(sampler, "_get_hierarchical_config"):
                        cfg = sampler._get_hierarchical_config(masked_input.shape[1])
                        constrain_window = cfg.get("eval_constrain_to_window", True)

                    x_ref = torch.tensor(masked_input[np.newaxis, :, :], dtype=torch.long, device=device)
                    t_eval = torch.full((1,), sampler.num_timesteps, dtype=torch.long, device=device)
                    hmask = sampler.build_hierarchical_mask(
                        x_0=x_ref,
                        t=t_eval,
                        window_start=mask_token_start if constrain_window else None,
                        window_end=mask_token_end if constrain_window else None,
                        preserve_structure=args.preserve_structure,
                    )[0].cpu().numpy()

                    for ch in range(masked_input.shape[1]):
                        masked_input[hmask[:, ch], ch] = mask_token_id[ch]
                elif args.preserve_structure:
                    # Keep Bar (0) and Pos (1) unmasked.
                    # Mask columns 2 through 7 (Instrument, Pitch, Dur, Vel, TimeSig, Tempo)
                    masked_input[mask_token_start:mask_token_end, 2:] = mask_token_id[2:]
                else:
                    # Apply mask to token range
                    masked_input[mask_token_start:mask_token_end] = mask_token_id
                
                # Repeat for batch (2 samples per midi)
                batch_size = 2
                x_T = np.tile(masked_input[np.newaxis, :, :], (batch_size, 1, 1))
                x_T_torch = torch.tensor(x_T, dtype=torch.long).to(device)
                
                # Sample              
                samples = sampler.sample(sample_steps=H.sample_steps, x_T=x_T_torch, B=batch_size)
                if isinstance(samples, torch.Tensor):
                    samples = samples.cpu().numpy()
                
                # Store
                cleaned_samples = [clean_sample(s, eos_token=eos_token) for s in samples]
                generated_samples.extend(cleaned_samples)
                original_samples_for_metrics.extend([original_tokens] * batch_size)
                
                # Just save individual batch here (convenience)
                mid_name = os.path.splitext(os.path.basename(midi_path))[0]
                save_generated_samples(cleaned_samples, "trio_octuple", samples_dir, prefix=f"infill_{mid_name}")
                
                count += 1
                
            except Exception as e:
                log(f"Skipping {midi_path}: {e}")
                continue
        
        log(f"Generated {len(generated_samples)} samples from {count} files.")
        
        # Calculate Metrics
        if generated_samples:
            log("Calculating infilling metrics...")
            
            metrics = evaluate_infilling(
                generated_samples, 
                original_samples_for_metrics,
                mask_start_step=mask_token_start,
                mask_end_step=mask_token_end
            )
        else:
            log("No samples generated, skipping metrics.")
            metrics = {}

    # Save Metrics
    metrics_path = os.path.join(metrics_dir, f"metrics_{args.task}_{args.load_step if args.load_step != 0 else 'best'}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f"Metrics saved to {metrics_path}")
    log("Done.")

if __name__ == "__main__":
    main()

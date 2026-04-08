import os
import re
import torch
import numpy as np
import logging
from ..preprocessing.data import POP909TrioConverter, OneHotMelodyConverter
from note_seq import note_sequence_to_midi_file


def log(output):
    """Log message to both file and console."""
    logging.info(output)
    print(output)


def resolve_unique_log_dir(target_dir, max_tries=1000):
    """
    Return a unique directory path by appending a numeric suffix if needed.

    Examples:
      /runs/exp      -> /runs/exp      (if missing)
      /runs/exp      -> /runs/exp_1    (if /runs/exp exists)
      /runs/exp      -> /runs/exp_2    (if _1 also exists)

    Returns:
      (unique_path, suffix_index_or_none)
    """
    if not os.path.exists(target_dir):
        return target_dir, None

    parent, name = os.path.split(target_dir)
    for idx in range(1, max_tries + 1):
        candidate = os.path.join(parent, f"{name}_{idx}")
        if not os.path.exists(candidate):
            return candidate, idx

    raise RuntimeError(
        f"Unable to find a unique run directory for '{target_dir}' after {max_tries} attempts."
    )


def _iter_run_dir_variants(base_dir):
    """Yield existing run directories matching base_dir or base_dir_<N>."""
    base_dir = os.path.abspath(base_dir)
    parent, name = os.path.split(base_dir)
    if not parent:
        return
    if not os.path.isdir(parent):
        return

    pattern = re.compile(rf"^{re.escape(name)}(?:_(\d+))?$")
    for entry in os.listdir(parent):
        m = pattern.match(entry)
        if not m:
            continue
        candidate = os.path.join(parent, entry)
        if os.path.isdir(candidate):
            yield candidate


def find_latest_checkpoint_step(run_dir):
    """Return the highest model checkpoint step in run_dir/checkpoints, or None."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    pattern = re.compile(r"^model_(\d+)\.th$")
    max_step = None
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if not m:
            continue
        step = int(m.group(1))
        if max_step is None or step > max_step:
            max_step = step
    return max_step


def resolve_resume_run_dir(base_dirs):
    """
    Find the best run directory to resume from across one or more base dirs.

    Returns:
      (run_dir, step) where step is the latest numeric model checkpoint step.
      Returns (None, None) if no checkpointed run is found.
    """
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    seen = set()
    best_dir, best_step, best_mtime = None, None, None

    for base in base_dirs:
        if not base:
            continue
        for candidate in _iter_run_dir_variants(base):
            if candidate in seen:
                continue
            seen.add(candidate)

            step = find_latest_checkpoint_step(candidate)
            if step is None:
                continue

            ckpt_dir = os.path.join(candidate, "checkpoints")
            mtime = os.path.getmtime(ckpt_dir)
            if (
                best_step is None
                or step > best_step
                or (step == best_step and (best_mtime is None or mtime > best_mtime))
            ):
                best_dir = candidate
                best_step = step
                best_mtime = mtime

    return best_dir, best_step


def config_log(log_dir, filename="log.txt"):
    """
    Configure logging to write to log_dir/logs/filename.
    
    Args:
        log_dir: Base directory for logs (e.g., runs/model_id/)
        filename: Name of log file (default: log.txt)
    """
    logs_dir = os.path.join(log_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, filename),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )


def start_training_log(hparams):
    """Log all hyperparameters at training start."""
    log("Using following hparams:")
    param_keys = list(hparams)
    param_keys.sort()
    for key in param_keys:
        log(f"> {key}: {hparams[key]}")


def _normalize_ckpt_name(name: str) -> str:
    # Map legacy absorbing names to clearer identifiers
    if name.endswith("_optim"):
        return "optim"
    if name.endswith("_ema"):
        return "ema"
    if name in ("absorbing", "sampler", "model"):
        return "model"
    return name


def save_model(model, model_save_name, step, log_dir):
    """
    Save model checkpoint to log_dir/checkpoints/.
    
    Args:
        model: PyTorch model to save
        model_save_name: Name identifier (e.g., "model", "ema", "optim")
        step: Training step number
        log_dir: Base directory (e.g., runs/model_id/)
    """
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    friendly_name = _normalize_ckpt_name(model_save_name)
    model_name = f"{friendly_name}_{step}.th"
    print(f"Saving {model_save_name} as {model_name}")
    save_path = os.path.join(ckpt_dir, model_name)
    torch.save(model.state_dict(), save_path)


def load_model(model, model_load_name, step, log_dir, fallback_dirs=None, strict=True):
    """
    Load model checkpoint from log_dir/checkpoints/.
    
    Args:
        model: PyTorch model to load weights into
        model_load_name: Name identifier (e.g., "model", "ema", "optim")
        step: Training step number
        log_dir: Base directory (e.g., runs/model_id/)
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        model: Model with loaded weights
    """
    ending = "th"
    if step == 0:
        step = "best"
        ending = "pt"
        
    ckpt_dir = os.path.join(log_dir, "checkpoints")

    friendly_name = _normalize_ckpt_name(model_load_name)
    candidates = [f"{friendly_name}_{step}.{ending}"]
    if friendly_name != model_load_name:
        candidates.append(f"{model_load_name}_{step}.{ending}")
        
    last_error = None
    search_dirs = [ckpt_dir]
    if fallback_dirs is not None:
        search_dirs_fallback = [os.path.join(base_dir, "checkpoints") for base_dir in fallback_dirs]
        search_dirs.extend(search_dirs_fallback)
        
    for base in search_dirs:
        for fname in candidates:
            path = os.path.join(base, fname)
            if not os.path.exists(path):
                continue
            print(f"Loading {fname} from {base}")
            try:
                state = torch.load(path)
                model.load_state_dict(state, strict=strict)
                return model
            except TypeError:
                model.load_state_dict(torch.load(path))
                return model
            except Exception as e:
                last_error = e
                continue

    if last_error:
        raise last_error
    raise FileNotFoundError(f"No checkpoint found for names {candidates} in {search_dirs}")


def save_samples(np_samples, step, log_dir):
    """
    Save generated samples to log_dir/samples/.
    
    Args:
        np_samples: NumPy array of generated samples
        step: Training step number
        log_dir: Base directory (e.g., runs/model_id/)
    """
    samples_dir = os.path.join(log_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    save_path = os.path.join(samples_dir, f'samples_{step}.npy')
    np.save(save_path, np_samples, allow_pickle=True)


def save_stats(H, stats, step):
    """
    Save training statistics to log_dir/stats/.
    
    Args:
        H: Hyperparameters object with log_dir
        stats: Dictionary of training statistics
        step: Training step number
    """
    base_dir = H.log_dir if os.path.isabs(H.log_dir) else H.log_dir
    stats_dir = os.path.join(base_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    save_path = os.path.join(stats_dir, f"stats_{step}.pt")
    log(f"Saving stats to {save_path}")
    torch.save(stats, save_path)


def load_stats(H, step):
    """
    Load training statistics from log_dir/stats/.
    
    Args:
        H: Hyperparameters object with log_dir
        step: Training step number
        
    Returns:
        dict: Dictionary of training statistics
    """

    stats_dir = [os.path.join(base_dir, "stats", f"stats_{step}.pt") for base_dir in [H.load_dir, H.log_dir]]
    
    for candidate_dir in stats_dir:
        if not os.path.exists(candidate_dir):
            raise FileNotFoundError(f"Stats file not found: {candidate_dir}")
    
        log(f"Loading stats from {candidate_dir}")
        return torch.load(candidate_dir)


def log_stats(step, stats):
    """
    Log training statistics to console and file.
    
    Args:
        step: Training step number
        stats: Dictionary of statistics to log
    """
    msg_parts = [f"Step {step}"]
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            msg_parts.append(f"{key}: {value:.6f}")
        elif hasattr(value, 'item'):  # torch.Tensor
            msg_parts.append(f"{key}: {value.item():.6f}")
    log(" | ".join(msg_parts))


def save_noteseqs(ns, prefix='pre_adv'):
    for i, n in enumerate(ns):
        note_sequence_to_midi_file(n, prefix + f'_{i}.mid')


def samples_2_noteseq(np_samples, tokenizer_id=None):
    """
    Convert numpy samples to note_seq objects using tokenizer registry.
    Handles fixing out-of-range tokens from early training.
    """
    
    # expected shape for melody encoder is (samples, Time,)
    if tokenizer_id == "melody" and np_samples.ndim == 3 and np_samples.shape[-1] == 1:
        samples = np_samples.squeeze(-1)
    
    if tokenizer_id:
        from ..tokenizers.registry import TOKENIZER_REGISTRY
        spec = TOKENIZER_REGISTRY.get(tokenizer_id)
        
        if spec and spec.factory:
            converter = spec.factory()
                    
            is_octuple = 'octuple' in tokenizer_id
            
            # --- SAFETY CLAMP: Fix for "Event out of range" ---
            if not is_octuple:
                max_val = None
                
                # 1. Explicitly defined sizes for known converters
                if tokenizer_id in ['melody', 'trio']:
                    max_val = 108 # HIGHEST MIDI TON IN MAGENTA PIPELINES FOR PIANO
                
                # 3. Apply Clamp
                if max_val is not None:
                    mask = np_samples > max_val
                    if np.any(mask):
                        # Clamp to 0
                        np_samples[mask] = 0
                
                return converter.from_tensors(np_samples)
            
            if is_octuple:
                # Octuple Structure: [Bar, Pos, Inst, Pitch, Dur, Vel, TimeSig, Tempo]
                
                #remove invalid pad tokens (-1)
                # Convert 3D batch to a list of valid 2D sequences
                cleaned_samples = []
                for i in range(len(np_samples)):
                    sample = np_samples[i]  # Shape: (Time, 8)
                    
                    # Create mask: True only for rows where NO subtoken is -1
                    valid_rows = ~(sample == -1).any(axis=1)
                    
                    sample = sample[valid_rows]
                    
                    if 'melody' in tokenizer_id:
                        # For Melody: Force Instrument to 0 (Grand Piano)
                        # This cleans up noise where the model hallucinates other instruments
                        sample[:, 2] = 0
                        
                    elif 'trio' in tokenizer_id:
                        # For Trio: We expect Inst IDs 0, 1, 2 (Melody, Bridge, Piano)
                        # The model might predict 5, 99, etc. 
                        # We simply Modulo 3 to force them back into valid track IDs
                        sample[:, 2] = sample[:, 2] % 3
                    
                    cleaned_samples.append(sample)

                return converter.from_tensors(cleaned_samples)
                
    return []
    



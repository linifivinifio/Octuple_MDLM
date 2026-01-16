import os
import torch
import numpy as np
import logging
from ..preprocessing.data import POP909TrioConverter, OneHotMelodyConverter
from note_seq import note_sequence_to_midi_file


def log(output):
    """Log message to both file and console."""
    logging.info(output)
    print(output)


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
                
                # 2. Dynamic check for other converters (like octuple)
                elif hasattr(converter, 'input_depth'): 
                    max_val = converter.input_depth - 1
                elif hasattr(converter, '_vocab_size'):
                    max_val = converter._vocab_size - 1
                
                # 3. Apply Clamp
                if max_val is not None:
                    # Identify out-of-bounds indices
                    mask = np_samples > max_val
                    if np.any(mask):
                        # Clamp to max_val (usually 'Silence'/'No Event')
                        np_samples[mask] = 0
                
                return converter.from_tensors(np_samples)
            
            if is_octuple:
                # Octuple Structure: [Bar, Pos, Inst, Pitch, Dur, Vel, Tempo, TimeSig]
                
                #remove invalid pad tokens (-1)
                # Convert 3D batch to a list of valid 2D sequences
                cleaned_samples = []
                for i in range(len(np_samples)):
                    sample = np_samples[i]  # Shape: (Time, 8)
                    
                    # Create mask: True only for rows where NO subtoken is -1
                    # (sample == -1).any(axis=1) finds rows with at least one -1
                    # ~ inverts it to find rows with NO -1s
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
                        # OR clamp them. Modulo preserves variance better usually.
                        sample[:, 2] = sample[:, 2] % 3
                    
                    # Append the clean, variable-length sample
                    cleaned_samples.append(sample)

                return converter.from_tensors(cleaned_samples)
                
    return []
    



import numpy as np
import torch
import torch.distributions as dists
from torch.nn import DataParallel
import os

from ..models import Transformer, AbsorbingDiffusion, ConVormer, HierarchTransformer, UTransformer
from ..registry import resolve_model_id
from note_seq import note_sequence_to_midi_file
from .log_utils import samples_2_noteseq


def get_sampler(H):
    """
    Factory function to create a sampler based on model type.
    
    Uses the model registry to resolve model IDs to internal implementations.
    The registry entry includes a factory function that handles model instantiation.
    
    Args:
        H: Hyperparameters object with model configuration
        
    Returns:
        AbsorbingDiffusion sampler with appropriate denoising model
    """
    # Resolve model_id to get ModelSpec with factory
    # Prefer canonical model_id over internal model name
    model_id = getattr(H, 'model_id', None) or H.model
    try:
        model_spec = resolve_model_id(model_id)
    except (ValueError, AttributeError):
        raise ValueError(f"Unknown model id '{model_id}'")
    
    # Use the factory function from the registry
    if model_spec.factory is None:
        raise ValueError(
            f"Model '{H.model}' has no factory registered"
        )
    
    return model_spec.factory(H)


@torch.no_grad()
def get_samples(sampler, sample_steps, x_T=None, temp=1.0, b=None, progress_handler=None):
    sampler.eval()

    if x_T is not None and not torch.is_tensor(x_T):
        x_T = torch.tensor(x_T).to(next(sampler.parameters()).device)

    result = sampler.sample(sample_steps=sample_steps, x_T=x_T, temp=temp, B=b, progress_handler=progress_handler)
    return result.cpu().numpy()


def save_generated_samples(samples, tokenizer_id, output_dir, prefix="sample"):
    """
    Converts raw tokens -> NoteSequences -> MIDI files.
    Includes robustness fixes for Octuple and Integers.
    """
    # 1. Ensure Integer Type (Critical fix)
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    samples = samples.astype(np.int64)

    # 2. Shape Fix for Melody OneHot
    if tokenizer_id == 'melody' and samples.ndim == 3 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)

    # 3. Convert to NoteSequence (using the fixed log_utils logic)
    # This handles the Instrument=0 enforcement automatically now.
    note_seqs = samples_2_noteseq(samples, tokenizer_id)

    # 4. Save Files
    saved_count = 0
    for i, ns in enumerate(note_seqs):
        if len(ns.notes) == 0:
            continue # Skip empty/noise outputs
        
        filename = f"{prefix}_{i}.mid"
        path = os.path.join(output_dir, filename)
        try:
            note_sequence_to_midi_file(ns, path)
            saved_count += 1
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    print(f"Saved {saved_count} MIDI files to {output_dir}")

def ns_to_np(ns, bars, tokenizer_id='melody'):
    """Helper to convert input MIDI to tokens for Infilling."""
    from ..preprocessing import (
        OneHotMelodyConverter, POP909TrioConverter, 
        POP909OctupleMelodyConverter, POP909OctupleTrioConverter
    )
    
    if 'octuple' in tokenizer_id:
        if 'trio' in tokenizer_id:
            converter = POP909OctupleTrioConverter(slice_bars=bars)
        else:
            converter = POP909OctupleMelodyConverter(slice_bars=bars)
    elif tokenizer_id == 'trio':
        converter = POP909TrioConverter(slice_bars=bars)
    else:
        converter = OneHotMelodyConverter(slice_bars=bars)

    tensors = converter.to_tensors(ns)
    if tensors.outputs and len(tensors.outputs) > 0:
        return tensors.outputs[0] # Return first slice
    raise ValueError("Input MIDI resulted in empty tokens.")

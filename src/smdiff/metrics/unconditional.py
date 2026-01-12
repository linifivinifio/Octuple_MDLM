"""Metrics for unconditional generation evaluation."""
import numpy as np
from .common import (
    kl_divergence,
    pitch_class_histogram,
    duration_histogram,
    velocity_histogram,
    note_density_per_bar,
    compute_self_similarity,
    compute_pitch_range,
    compute_sample_diversity,
    is_valid_octuple_sample,
    extract_trio_durations,
    compute_trio_self_similarity,
    compute_trio_sample_diversity,
    is_valid_trio_sample
)


def evaluate_unconditional(generated_samples, train_samples, is_octuple=True):
    """
    Evaluate unconditional generation against training data.
    
    Args:
        generated_samples: List of (T, C) generated token arrays (variable length)
        train_samples: List of (T, C) training token arrays (variable length)
        is_octuple: Whether samples are Octuple encoded (Defaults to True for this script)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Octuple indices: Bar=0, Pos=1, Prog=2, Pitch=3, Dur=4, Vel=5
    # Enforce Octuple defaults strictly
    pitch_idx = 3
    duration_idx = 4
    velocity_idx = 5
    bar_idx = 0

    # Compute distributions on lists (functions handle variable-length sequences)
    # Ensure inputs are lists of arrays
    if not isinstance(generated_samples, list):
         generated_samples = [s for s in generated_samples]
    if not isinstance(train_samples, list):
         train_samples = [s for s in train_samples]

    if not is_octuple:
        # TRIO / Melody Encoding Logic (Pitch Only)
        # Tokens are 0 (No Event), 1 (Note Off), 2+ (Pitch + 21 - 2)
        MIN_PITCH = 21

        def decode_pitches(samples):
            decoded_list = []
            for s in samples:
                # s: (T, 3) or (T, 1) or flat
                flat = np.asarray(s).flatten()
                # Filter valid Note Ons (>= 2)
                valid = flat[flat >= 2]
                pitches = valid - 2 + MIN_PITCH
                decoded_list.append(pitches)
            return decoded_list

        gen_pitches = decode_pitches(generated_samples)
        train_pitches = decode_pitches(train_samples)

        # Pitch Class Histogram
        # Note: pitch_class_histogram handles 1D arrays by flattening, so pitch_idx is ignored
        gen_pch = pitch_class_histogram(gen_pitches, pitch_idx=0)
        train_pch = pitch_class_histogram(train_pitches, pitch_idx=0)
        metrics['pch_kl'] = kl_divergence(train_pch, gen_pch)

        # Pitch Range (Mean/Std)
        # compute_pitch_range expects (T, C) and slices [:, pitch_idx]
        # We can implement simpler version here or trick it
        p_ranges = []
        for p_arr in gen_pitches:
            if len(p_arr) > 0:
                p_ranges.append(p_arr.max() - p_arr.min())
            else:
                p_ranges.append(0)
        
        metrics['pitch_range_mean'] = np.mean(p_ranges) if p_ranges else 0.0
        metrics['pitch_range_std'] = np.std(p_ranges) if p_ranges else 0.0

        # Note Density (Approximate: Note On events per 16 steps)
        # Assume 1024 steps, 64 bars -> 16 steps/bar
        def compute_density(samples_raw):
            densities = []
            steps_per_bar = 16
            for s in samples_raw:
                # s: (1024, 3)
                # Reshape to (Bars, Steps, Tracks) -> (64, 16, 3)
                # Count tokens >= 2
                if len(s) == 0: continue
                # Handle varying lengths if any
                n_bars = len(s) // steps_per_bar
                if n_bars == 0: continue
                
                s_trunc = s[:n_bars*steps_per_bar]
                s_reshaped = s_trunc.reshape(n_bars, steps_per_bar, -1)
                
                # Count >= 2 per bar
                notes_per_bar = np.sum(s_reshaped >= 2, axis=(1, 2))
                densities.extend(notes_per_bar)
            return np.array(densities)

        gen_density = compute_density(generated_samples)
        train_density = compute_density(train_samples)
        
        max_d = int(max(gen_density.max(), train_density.max()) + 1) if (len(gen_density) > 0 and len(train_density) > 0) else 1
        gen_d_hist = np.bincount(gen_density.astype(int), minlength=max_d)
        train_d_hist = np.bincount(train_density.astype(int), minlength=max_d)
        metrics['note_density_kl'] = kl_divergence(train_d_hist, gen_d_hist)

        # Duration KL (Explicitly extracted from grid)
        gen_durs = extract_trio_durations(generated_samples)
        train_durs = extract_trio_durations(train_samples)
        
        # Clip to reasonable max bins (e.g. 128 = 8 bars) for histogram
        MAX_DUR = 128
        gen_durs = gen_durs[gen_durs < MAX_DUR]
        train_durs = train_durs[train_durs < MAX_DUR]
        
        if len(gen_durs) > 0 and len(train_durs) > 0:
             # Make histograms
             d_max = max(gen_durs.max(), train_durs.max()) + 1
             hist_g = np.bincount(gen_durs, minlength=d_max)
             hist_t = np.bincount(train_durs, minlength=d_max)
             metrics['duration_kl'] = kl_divergence(hist_t, hist_g)
        else:
             metrics['duration_kl'] = None

        # Duration Stats
        if len(gen_durs) > 0:
            metrics['duration_mean'] = np.mean(gen_durs)
            metrics['duration_std'] = np.std(gen_durs)
        else:
            metrics['duration_mean'] = None
            metrics['duration_std'] = None

        # Diversity
        metrics['sample_diversity'] = compute_trio_sample_diversity(generated_samples)
        
        # Validity
        v_count = sum([is_valid_trio_sample(s) for s in generated_samples])
        metrics['valid_samples_pct'] = 100.0 * v_count / len(generated_samples) if len(generated_samples) > 0 else None

        # Self Sim
        ss = [compute_trio_self_similarity(s) for s in generated_samples]
        metrics['self_similarity'] = np.mean(ss) if ss else None
        metrics['self_similarity_std'] = np.std(ss) if ss else None
        
        return metrics

    # OCTUPLE LOGIC (Original)
    gen_pch = pitch_class_histogram(generated_samples, pitch_idx=pitch_idx)
    train_pch = pitch_class_histogram(train_samples, pitch_idx=pitch_idx)
    metrics['pch_kl'] = kl_divergence(train_pch, gen_pch)
    
    # Duration KL
    # Increase max_bins for Octuple (which has typically >32 duration tokens)
    gen_dur = duration_histogram(generated_samples, duration_idx=duration_idx, max_bins=128)
    train_dur = duration_histogram(train_samples, duration_idx=duration_idx, max_bins=128)
    metrics['duration_kl'] = kl_divergence(train_dur, gen_dur)
    
    # Velocity KL
    gen_vel = velocity_histogram(generated_samples, velocity_idx=velocity_idx)
    train_vel = velocity_histogram(train_samples, velocity_idx=velocity_idx)
    metrics['velocity_kl'] = kl_divergence(train_vel, gen_vel)
    
    # Note Density KL
    gen_density = note_density_per_bar(generated_samples, bar_idx=bar_idx)
    train_density = note_density_per_bar(train_samples, bar_idx=bar_idx)
    
    # Create histograms for note density
    # Handle empty arrays if any
    max_d_gen = gen_density.max() if len(gen_density) > 0 else 0
    max_d_train = train_density.max() if len(train_density) > 0 else 0
    max_density = int(max(max_d_gen, max_d_train) + 1)
    
    gen_density_hist = np.bincount(gen_density.astype(int), minlength=max_density)
    train_density_hist = np.bincount(train_density.astype(int), minlength=max_density)
    metrics['note_density_kl'] = kl_divergence(train_density_hist, gen_density_hist)
    
    # Musical coherence metrics (per-sample averages)
    self_sims = []
    pitch_ranges = []
    
    for sample in generated_samples:
        if len(sample) == 0:
            continue
        self_sims.append(compute_self_similarity(sample, pitch_idx=pitch_idx, duration_idx=duration_idx))
        pitch_ranges.append(compute_pitch_range(sample, pitch_idx=pitch_idx))
    
    metrics['self_similarity'] = np.mean(self_sims) if self_sims else None
    metrics['self_similarity_std'] = np.std(self_sims) if self_sims else None
    
    metrics['pitch_range_mean'] = np.mean(pitch_ranges) if pitch_ranges else None
    metrics['pitch_range_std'] = np.std(pitch_ranges) if pitch_ranges else None
    
    # Diversity metric
    metrics['sample_diversity'] = compute_sample_diversity(generated_samples, 
                                                         pitch_idx=pitch_idx, 
                                                         duration_idx=duration_idx)
    
    # Validity metric
    valid_count = sum([is_valid_octuple_sample(s, 
                                      pitch_idx=pitch_idx, 
                                      duration_idx=duration_idx) for s in generated_samples])
    metrics['valid_samples_pct'] = 100.0 * valid_count / len(generated_samples) if len(generated_samples) > 0 else None
    
    return metrics

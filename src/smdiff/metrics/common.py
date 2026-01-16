"""Common utilities for metrics computation."""
import numpy as np
from scipy.stats import entropy


def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence between two distributions.
    
    Args:
        p: True distribution (numpy array)
        q: Predicted distribution (numpy array)
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize to sum to 1
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)
    
    # Add epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon
    
    return entropy(p, q)


def extract_trio_durations(samples):
    """
    Extract note durations from Trio time-grid samples.
    Args:
        samples: List of (T, 3) or (T,) arrays. or (B, T, 3)
    Returns:
        List of duration values (integers).
    """
    all_durations = []
    
    # Standardize input to list of arrays
    if not isinstance(samples, list):
         if hasattr(samples, 'ndim') and samples.ndim == 3:
             samples = [s for s in samples]
         else:
             samples = [samples]

    for s in samples:
        s = np.asarray(s)
        if s.ndim == 1:
            # Handle single track
            tracks = [s]
        else:
            # Handle 3 tracks (or N tracks)
            tracks = [s[:, i] for i in range(s.shape[1])]
            
        for track in tracks:
            # Iterate through time steps
            N = len(track)
            i = 0
            while i < N:
                token = track[i]
                if token >= 2: # Note On
                    dur = 1
                    j = i + 1
                    while j < N:
                        if track[j] == 0: # Sustain
                            dur += 1
                            j += 1
                        else: # Note Off (1) or New Note (>=2)
                            break
                    all_durations.append(dur)
                i += 1
    return np.array(all_durations)



def pitch_class_histogram(tokens, pitch_idx=2):
    """
    Compute pitch class histogram (C, C#, D, ..., B).
    
    Args:
        tokens: List of (T, C) arrays OR single (T, C) array OR (N, T, C) array
        pitch_idx: Index of pitch in the token tuple
        
    Returns:
        12-element array with pitch class counts
    """
    # Handle list of variable-length arrays
    if isinstance(tokens, list):
        pitches = []
        for sample in tokens:
            sample = np.asarray(sample)
            if sample.ndim == 2:
                pitches.extend(sample[:, pitch_idx])
            else:
                pitches.extend(sample.flatten())
        pitches = np.array(pitches)
    else:
        tokens = np.asarray(tokens)
        if tokens.ndim == 2:
            pitches = tokens[:, pitch_idx]
        else:  # (N, T, C)
            pitches = tokens[:, :, pitch_idx].flatten()
    
    # Remove invalid pitches (e.g., > 127 or special tokens)
    # Ignore 0 (padding/silence)
    pitches = pitches[(pitches > 0) & (pitches < 128)]
    
    # Map to pitch classes (0-11)
    pitch_classes = pitches % 12
    
    # Count occurrences
    hist = np.bincount(pitch_classes.astype(int), minlength=12)
    return hist


def duration_histogram(tokens, duration_idx=3, max_bins=32):
    """
    Compute duration histogram.
    
    Args:
        tokens: List of (T, C) arrays OR single (T, C) array OR (N, T, C) array
        duration_idx: Index of duration in the token tuple
        max_bins: Maximum duration bins to consider
        
    Returns:
        Histogram of duration values
    """
    # Handle list of variable-length arrays
    if isinstance(tokens, list):
        durations = []
        for sample in tokens:
            sample = np.asarray(sample)
            if sample.ndim == 2:
                durations.extend(sample[:, duration_idx])
            else:
                durations.extend(sample.flatten())
        durations = np.array(durations)
    else:
        tokens = np.asarray(tokens)
        if tokens.ndim == 2:
            durations = tokens[:, duration_idx]
        else:  # (N, T, C)
            durations = tokens[:, :, duration_idx].flatten()
    
    # Filter valid durations
    durations = durations[(durations >= 0) & (durations < max_bins)]
    
    hist = np.bincount(durations.astype(int), minlength=max_bins)
    return hist


def velocity_histogram(tokens, velocity_idx=4, max_bins=128):
    """
    Compute velocity histogram.
    
    Args:
        tokens: List of (T, C) arrays OR single (T, C) array OR (N, T, C) array
        velocity_idx: Index of velocity in the token tuple
        max_bins: Maximum velocity bins (typically 128)
        
    Returns:
        Histogram of velocity values
    """
    # Handle list of variable-length arrays
    if isinstance(tokens, list):
        velocities = []
        for sample in tokens:
            sample = np.asarray(sample)
            if sample.ndim == 2:
                velocities.extend(sample[:, velocity_idx])
            else:
                velocities.extend(sample.flatten())
        velocities = np.array(velocities)
    else:
        tokens = np.asarray(tokens)
        if tokens.ndim == 2:
            velocities = tokens[:, velocity_idx]
        else:  # (N, T, C)
            velocities = tokens[:, :, velocity_idx].flatten()
    
    # Filter valid velocities
    velocities = velocities[(velocities >= 0) & (velocities < max_bins)]
    
    hist = np.bincount(velocities.astype(int), minlength=max_bins)
    return hist


def note_density_per_bar(tokens, bar_idx=0, steps_per_bar=16):
    """
    Compute notes per bar distribution.
    
    Args:
        tokens: List of (T, C) arrays OR (N, T, C) array
        bar_idx: Index of bar number inside token (Octuple) OR None (Grid/Implicit)
        steps_per_bar: Steps per bar (typically 16 for 16th notes) - used for Grid
        
    Returns:
        Array of notes per bar for each sample
    """
    densities = []
    
    # Handle list of variable-length arrays
    if isinstance(tokens, list):
        samples = tokens
    else:
        tokens = np.asarray(tokens)
        if tokens.ndim == 2:
            samples = [tokens]
        else:  # (N, T, C)
            samples = [tokens[i] for i in range(tokens.shape[0])]
    
    for sample in samples:
        sample = np.asarray(sample)
        
        # --- Octuple Case (Bar Index exists) ---
        if bar_idx is not None and sample.ndim == 2 and sample.shape[1] > bar_idx:
            bars = sample[:, bar_idx]
            unique_bars = np.unique(bars[bars >= 0])
            
            notes_per_bar = []
            for bar in unique_bars:
                mask = bars == bar
                notes_per_bar.append(mask.sum())
            if notes_per_bar:
                densities.extend(notes_per_bar)

        # --- Grid Case (Implicit Bars by Time) ---
        else:
            # Assume constant time grid
            # sample shape is (T,) or (T, C)
            # pitches are non-zero/non-padding
            
            total_steps = len(sample)
            num_bars = total_steps // steps_per_bar
            
            # Identify active notes (pitch > 0)
            # If (T, C), any active channel counts as a note? Or sum of notes?
            # Typically density = note onsets. 
            # Simplified: Count > 0 entries per bar window
            
            if sample.ndim == 2:
                # Sum active notes across tracks (e.g. Trio)
                is_note = (sample > 0).sum(axis=1) # (T,) -> count of notes at each step
            else:
                 is_note = (sample > 0).astype(int) # (T,)
            
            for b in range(num_bars):
                start = b * steps_per_bar
                end = start + steps_per_bar
                # extract window
                window = is_note[start:end]
                # Count note occurrences (non-zero steps? or discrete notes?)
                # For grid, this approximates density
                densities.append(np.sum(window))

    return np.array(densities)


def compute_self_similarity(tokens, pitch_idx=2, duration_idx=3, window_bars=4, steps_per_bar=16):
    """
    Compute self-similarity by comparing consecutive windows.
    
    Args:
        tokens: (T, C) single sample
        pitch_idx: Index of pitch
        duration_idx: Index of duration
        window_bars: Window size in bars
        steps_per_bar: Steps per bar
        
    Returns:
        Average cosine similarity between consecutive windows
    """
    window_size = window_bars * steps_per_bar
    num_windows = len(tokens) // window_size
    
    if num_windows < 2:
        return 0.0
    
    similarities = []
    for i in range(num_windows - 1):
        start1 = i * window_size
        end1 = start1 + window_size
        start2 = (i + 1) * window_size
        end2 = start2 + window_size
        
        window1 = tokens[start1:end1]
        window2 = tokens[start2:end2]
        
        # Create feature vector (pitch + duration)
        feat1 = np.concatenate([window1[:, pitch_idx], window1[:, duration_idx]])
        feat2 = np.concatenate([window2[:, pitch_idx], window2[:, duration_idx]])
        
        # Cosine similarity
        norm1 = np.linalg.norm(feat1) + 1e-10
        norm2 = np.linalg.norm(feat2) + 1e-10
        sim = np.dot(feat1, feat2) / (norm1 * norm2)
        similarities.append(sim)
    
    return np.mean(similarities) if similarities else 0.0


def compute_pitch_range(tokens, pitch_idx=2):
    """
    Compute pitch range (span between min and max pitch).
    
    Args:
        tokens: (T, C) single sample
        pitch_idx: Index of pitch
        
    Returns:
        Pitch range in semitones
    """
    pitches = tokens[:, pitch_idx]
    # Ignore 0 (padding/silence)
    valid_pitches = pitches[(pitches > 0) & (pitches < 128)]
    
    if len(valid_pitches) == 0:
        return 0
    
    return valid_pitches.max() - valid_pitches.min()


def compute_sample_diversity(tokens_list, pitch_idx=2, duration_idx=3):
    """
    Compute average pairwise distance between samples.
    
    Args:
        tokens_list: List of (T, C) arrays (variable length)
        pitch_idx: Index of pitch
        duration_idx: Index of duration
        
    Returns:
        Average pairwise L2 distance
    """
    if len(tokens_list) < 2:
        return 0.0
    
    features = []
    for tokens in tokens_list:
        tokens = np.asarray(tokens)
        # Create feature vector
        pitches = tokens[:, pitch_idx]
        # Ignore 0 (padding/silence)
        valid_pitches = pitches[(pitches > 0) & (pitches < 128)]
        pch = np.bincount(valid_pitches % 12, minlength=12)
        
        durations = tokens[:, duration_idx]
        valid_durations = durations[(durations >= 0) & (durations < 32)]
        dur = np.bincount(valid_durations.astype(int), minlength=32)
        
        feat = np.concatenate([pch, dur])
        features.append(feat)
    
    features = np.array(features)
    
    # Compute pairwise distances
    distances = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(features[i] - features[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0


def is_valid_trio_sample(s, max_token=89):
    # s is (T, 3) or (T,)
    s = np.asarray(s)
    # Check bounds
    if np.any(s < 0) or np.any(s > max_token):
        return False
    return True

def compute_trio_sample_diversity(samples):
    if len(samples) < 2:
        return 0.0
    
    features = []
    for s in samples:
        s = np.asarray(s)
        # Flatten and filter for Note Ons (>=2)
        flat = s.flatten()
        valid = flat[flat >= 2]
        
        # Mapping to pitch class (0-11)
        # Token 2 -> Pitch 21. 21 % 12 = 9.
        # Token n -> Pitch n - 2 + 21.
        # Pitch Class = (n - 2 + 21) % 12 = (n + 19) % 12 = (n + 7) % 12
        pitch_classes = (valid + 7) % 12
        
        # Feature: Normalized PCH
        pch = np.bincount(pitch_classes.astype(int), minlength=12)
        # Normalize
        norm = np.linalg.norm(pch) 
        if norm > 0:
            pch = pch / norm
            
        features.append(pch)
    
    features = np.array(features)
    
    # Calculate pairwise distances
    distances = []
    n = len(features)
    # Subsampling if too many for n^2? 
    # If n > 100, maybe sample? common.py doesn't, so we won't.
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(features[i] - features[j])
            distances.append(dist)
            
    return np.mean(distances) if distances else 0.0

def compute_trio_self_similarity(sample, window_steps=64):
    # sample: (T, 3) or (T,)
    s = np.asarray(sample)
    if s.ndim == 1:
        # Handle 1D melody
        pass
    elif s.ndim != 2: 
        # Should be (T, C) or (T,)
        return 0.0
        
    n_steps = s.shape[0]
    num_windows = n_steps // window_steps
    
    if num_windows < 2:
        return 0.0
        
    windows = []
    for i in range(num_windows):
        start = i * window_steps
        end = start + window_steps
        win_data = s[start:end] # (64, 3)
        
        flat = win_data.flatten()
        valid = flat[flat >= 2]
        pitch_classes = (valid + 7) % 12
        pch = np.bincount(pitch_classes.astype(int), minlength=12)
        
        windows.append(pch)
        
    similarities = []
    for i in range(len(windows) - 1):
        v1 = windows[i]
        v2 = windows[i+1]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(v1, v2) / (norm1 * norm2)
            similarities.append(sim)
        else:
             # if one window is empty (silence), sim is 0? or 1 if both empty?
             # If both empty, they are similar (silence==silence).
             if norm1 == 0 and norm2 == 0:
                 similarities.append(1.0)
             else:
                 similarities.append(0.0)
                 
    return np.mean(similarities) if similarities else 0.0


def is_valid_octuple_sample(tokens, max_pitch=127, max_duration=255, pitch_idx=2, duration_idx=3):
    """
    Check if a sample is valid (decodable to MIDI).
    
    Args:
        tokens: (T, C) array
        max_pitch: Maximum valid pitch
        max_duration: Maximum valid duration
        pitch_idx: Index of pitch token
        duration_idx: Index of duration token
        
    Returns:
        Boolean indicating validity
    """
    # Check if pitches are in valid range
    if tokens.ndim < 2 or tokens.shape[1] <= max(pitch_idx, duration_idx):
        return False

    pitches = tokens[:, pitch_idx]
    if np.any(pitches > max_pitch) or np.any(pitches < 0):
        return False
    
    # Check if durations are reasonable
    durations = tokens[:, duration_idx]
    if np.any(durations > max_duration) or np.any(durations < 0):
        return False
    
    # Check if there's at least one note
    if len(pitches) == 0:
        return False
    
    return True

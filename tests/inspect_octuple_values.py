import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.smdiff.metrics.common import duration_histogram, velocity_histogram

try:
    # path = r"runs/samples_35000.npy"
    path = "data/POP909_trio_octuple.npy"
    data = np.load(path, allow_pickle=True)
    
    print(f"Loaded {path}, shape {data.shape}")
    
    # Take first 100 samples
    samples = data

    
    # Inspect Sample 0
    s0 = samples[0]
    print(f"Sample 0 shape: {s0.shape}")
    print("Sample 0 first 5 rows:")
    print(s0[:5])
    
    # Check Pitch(3), Duration(4), Velocity(5) stats
    # Handle object array if necessary
    all_s = []
    for s in samples:
        if isinstance(s, np.ndarray):
            all_s.append(s)
            
    pitches = np.concatenate([s[:, 3] for s in all_s])
    durations = np.concatenate([s[:, 4] for s in all_s])
    velocities = np.concatenate([s[:, 5] for s in all_s])
    bars = np.concatenate([s[:, 0] for s in all_s])
    positions = np.concatenate([s[:, 1] for s in all_s])
    
    print("\n--- STATISTICS (First 100 samples) ---")
    print(f"Pitch (idx 3): min={pitches.min()}, max={pitches.max()}, mean={pitches.mean():.2f}")
    print(f"Duration (idx 4): min={durations.min()}, max={durations.max()}, mean={durations.mean():.2f}")
    print(f"Velocity (idx 5): min={velocities.min()}, max={velocities.max()}, mean={velocities.mean():.2f}")
    print(f"Bars (idx 0): min={bars.min()}, max={bars.max()}, mean={bars.mean():.2f}")
    print(f"Position (idx 1): min={positions.min()}, max={positions.max()}, mean={positions.mean():.2f}")
    
    print(f"Unique Durations: {np.unique(durations)}")
    print(f"Unique Velocities: {np.unique(velocities)}")
    
    # Check histograms
    d_hist = duration_histogram(all_s, duration_idx=4)
    v_hist = velocity_histogram(all_s, velocity_idx=5)
    b_hist = duration_histogram(all_s, duration_idx=0, max_bins=64)
    pos_hist = duration_histogram(all_s, duration_idx=1, max_bins=64)

    print(f"\nDuration Hist: {d_hist}")
    print(f"Velocity Hist: {v_hist}")
    print(f"Bin hist: ", b_hist)
    print(f"Pos hist: ", pos_hist)
    
    # Plot bars and positions for a single sample
    import matplotlib.pyplot as plt
    
    # Use s0 (single sample)
    single_bars = s0[:, 0]
    single_pos = s0[:, 1]

    plt.figure(figsize=(12, 6))
    plt.plot(single_bars, label='Bar', marker='o', markersize=2)
    plt.plot(single_pos, label='Position', marker='x', markersize=2, alpha=0.7)
    plt.title("Sequential Bar and Position Values (Single Sample)")
    plt.xlabel("Token Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/single_sample_structure_no_normalization.png")
    print("Saved plots/single_sample_structure.png")
    
except Exception as e:
    print(f"Error: {e}")

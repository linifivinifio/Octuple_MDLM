import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
import sys

# Ensure repository root is on sys.path so top-level packages like 'hparams' resolve
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also ensure 'src' is on sys.path so 'smdiff' package resolves when running by path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect generated samples vs Reference Data.")
    parser.add_argument('--models', nargs='+', required=True, 
                        help="List of model names (folder names under runs/).")
    parser.add_argument('--mode', type=str, required=True, choices=['hist', 'seq'],
                        help="Mode: 'hist' for distribution histograms, 'seq' for single-sample bar structure.")
    parser.add_argument('--step', type=str, default='latest',
                        help="Step number to load (int) or 'latest' to auto-detect highest number.")
    parser.add_argument('--project_dir', type=str, default='.',
                        help="Root directory containing 'runs' and 'data'.")
    parser.add_argument('--ref_path', type=str, default='data/POP909_trio_octuple.npy',
                        help="Path to the reference .npy file.")
    parser.add_argument('--num_samples', type=int, default=10,
                        help="Number of samples to overlay in sequential plots (default: 10).")
    return parser.parse_args()

def find_samples_file(model_dir, step_arg):
    """
    Finds the samples file based on step (specific int or 'latest').
    """
    samples_dir = os.path.join(model_dir, 'samples')
    if not os.path.exists(samples_dir):
        return None

    if step_arg == 'latest':
        pattern = re.compile(r"samples_(\d+)\.npy")
        matches = []
        for fname in os.listdir(samples_dir):
            match = pattern.match(fname)
            if match:
                matches.append((fname, int(match.group(1))))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        if matches:
            return os.path.join(samples_dir, matches[0][0])
    else:
        fname = f"samples_{step_arg}.npy"
        fpath = os.path.join(samples_dir, fname)
        if os.path.exists(fpath):
            return fpath
            
    return None

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    try:
        data = np.load(path, allow_pickle=True)
        valid_samples = []
        # Handle case where data might be object array or structured differently
        for s in data:
            if isinstance(s, np.ndarray):
                valid_samples.append(s)
        return valid_samples
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def plot_histograms(model_data, ref_data, output_dir):
    """
    Generates 4 figures. Each subplot compares Model (Blue) vs Reference (Orange).
    """
    features = {
        'Bars': {'idx': 0, 'bins': 32},
        'Velocity': {'idx': 5, 'bins': 50},
        'Pitch': {'idx': 3, 'bins': 88},
        'Duration': {'idx': 4, 'bins': 50},
        'Position': {'idx': 1, 'bins': 32}
    }

    # Pre-calculate reference values
    ref_values = {}
    if ref_data:
        for name, config in features.items():
            ref_values[name] = np.concatenate([s[:, config['idx']] for s in ref_data])

    num_models = len(model_data)
    
    for feature_name, config in features.items():
        fig, axes = plt.subplots(num_models, 1, figsize=(10, 4 * num_models), constrained_layout=True)
        if num_models == 1: axes = [axes]
        
        idx = config['idx']
        
        for ax, (model_name, samples) in zip(axes, model_data.items()):
            # Plot Reference Data first (Background, Orange)
            if ref_data:
                ax.hist(ref_values[feature_name], bins=config['bins'], density=True, 
                        alpha=0.4, color='orange', label='Reference', edgecolor='none')

            # Plot Model Data (Foreground, Blue)
            if samples:
                values = np.concatenate([s[:, idx] for s in samples])
                ax.hist(values, bins=config['bins'], density=True, 
                        alpha=0.6, color='steelblue', label=model_name, edgecolor='black')
                
                stats_txt = f"Mean: {values.mean():.2f}"
                ax.set_xlabel(stats_txt)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center')

            ax.set_title(f"{feature_name}: {model_name} vs Reference")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, f"hist_{feature_name.lower()}.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close(fig)

def plot_sequences(model_data, ref_data, output_dir, num_samples_limit):
    """
    Generates sequence plots for BOTH Bars (idx 0) and Positions (idx 1).
    Overlays 'num_samples_limit' samples.
    """
    num_models = len(model_data)
    if num_models == 0:
        return

    # Define features to plot sequentially: (Name, Index)
    seq_features = [('Bars', 0), ('Position', 1)]

    for feat_name, feat_idx in seq_features:
        fig, axes = plt.subplots(num_models, 1, figsize=(12, 5 * num_models), constrained_layout=True)
        if num_models == 1: axes = [axes]
        
        # Pre-fetch Reference Sequence (Sample 0)
        ref_seq = None
        if ref_data and len(ref_data) > 0:
            ref_seq = ref_data[0][:, feat_idx]

        for ax, (model_name, samples) in zip(axes, model_data.items()):
            # 1. Plot Reference (Background, Bold Orange)
            if ref_seq is not None:
                ax.plot(ref_seq, marker='', linestyle='--', linewidth=2.5, 
                        color='orange', alpha=0.8, label='Reference (Ground Truth)')
            
            # 2. Plot First N Model Samples (Foreground, Thin Blue Spaghetti)
            if samples and len(samples) > 0:
                limit = num_samples_limit
                actual_plotted = 0
                
                for i in range(min(limit, len(samples))):
                    s = samples[i]
                    model_seq = s[:, feat_idx]
                    
                    lbl = model_name if i == 0 else "_nolegend_"
                    
                    ax.plot(model_seq, marker='', linestyle='-', linewidth=1.0, 
                            color='steelblue', alpha=0.6, label=lbl)
                    actual_plotted += 1
                
                ax.set_title(f"{model_name} - {feat_name} Structure (First {actual_plotted} Samples)")
            else:
                ax.text(0.5, 0.5, "No Data", ha='center')
                ax.set_title(f"{model_name} - {feat_name}")

            ax.set_ylabel(f"{feat_name} Index")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Token Index")
        
        save_path = os.path.join(output_dir, f"seq_{feat_name.lower()}_structure.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close(fig)

def main():
    args = parse_args()
    runs_dir = os.path.join(args.project_dir, 'runs')
    
    # 1. Load Reference Data
    ref_path = os.path.join(args.project_dir, args.ref_path)
    print(f"Loading Reference: {ref_path}")
    ref_data = load_data(ref_path)
    
    # 2. Load Model Data
    collected_data = {} 
    
    for model_name in args.models:
        model_path = os.path.join(runs_dir, model_name)
        file_path = find_samples_file(model_path, args.step)
        
        if file_path:
            print(f"Loading Model {model_name}: {file_path}")
            samples = load_data(file_path)
            collected_data[model_name] = samples if samples else []
        else:
            print(f"Warning: No samples found for {model_name}")
            collected_data[model_name] = []

    if not collected_data and not ref_data:
        print("No data loaded at all. Exiting.")
        return

    # 3. Plotting
    results_dir = os.path.join(args.project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    if args.mode == 'hist':
        plot_histograms(collected_data, ref_data, results_dir)
    elif args.mode == 'seq':
        plot_sequences(collected_data, ref_data, results_dir, args.num_samples)

if __name__ == "__main__":
    main()
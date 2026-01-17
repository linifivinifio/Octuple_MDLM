import os
import json
import argparse
import re
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also ensure 'src' is on sys.path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse metrics JSON files and export to Transposed Markdown table.")
    parser.add_argument('--load_step', type=str, required=True, choices=['best', 'latest'],
                        help="The load step: 'best' (looks for _best.json) or 'latest' (looks for any _<number>.json).")
    parser.add_argument('--task', type=str, required=True, choices=['infill', 'uncond'],
                        help="The task name (infill or uncond).")
    parser.add_argument('--project_dir', type=str, default='.',
                        help="Root project directory containing 'runs'. Defaults to current dir.")
    return parser.parse_args()

def format_value(val):
    """Formats floats to 4 decimal places, handles None."""
    if val is None:
        return "None"
    if isinstance(val, (float, int)):
        return f"{val:.4f}"
    return str(val)

def find_metrics_file(metrics_dir, task, mode):
    """
    Finds the specific metrics file based on mode ('best' or 'latest').
    Returns tuple: (full_path, step_identifier) or (None, None)
    """
    if not os.path.exists(metrics_dir):
        return None, None

    if mode == 'best':
        # Look specifically for metrics_<task>_best.json
        filename = f"metrics_{task}_best.json"
        full_path = os.path.join(metrics_dir, filename)
        if os.path.exists(full_path):
            return full_path, "best"
    
    elif mode == 'latest':
        # Regex to capture the integer step number
        pattern = re.compile(rf"metrics_{task}_(\d+)\.json")
        matches = []
        
        # 1. Collect all valid matches
        for fname in os.listdir(metrics_dir):
            match = pattern.match(fname)
            if match:
                step_num = int(match.group(1)) # Convert regex group to int for correct sorting
                matches.append((fname, step_num))
        
        # 2. Sort descending by step number (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Return the first result if any exist
        if matches:
            highest_file, highest_step = matches[0]
            return os.path.join(metrics_dir, highest_file), str(highest_step)

    return None, None

def main():
    args = parse_args()
    
    runs_dir = os.path.join(args.project_dir, 'runs')
    results_dir = os.path.join(args.project_dir, 'results')
    
    print(f"Scanning '{runs_dir}' for task='{args.task}' with load_step='{args.load_step}'...")

    if not os.path.exists(runs_dir):
        print(f"Error: Directory '{runs_dir}' does not exist.")
        return

    # Data structure: 
    # models_data = { 'model_name': { 'metric_name': value, ... }, ... }
    models_data = {}
    
    # Track all unique metrics found across all models
    all_metric_keys = set()
    
    # 1. Collect Data
    # Get immediate subdirectories of runs/
    model_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]

    for model_name in model_dirs:
        metrics_dir = os.path.join(runs_dir, model_name, 'metrics')
        
        file_path, step_found = find_metrics_file(metrics_dir, args.task, args.load_step)
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Filter out standard deviation metrics
                filtered_data = {k: v for k, v in data.items() if not k.endswith('_std')}
                
                # Store data
                models_data[model_name] = filtered_data
                all_metric_keys.update(filtered_data.keys())
                
                # Optional: print which step was actually used
                print(f"  [{model_name}] Found step: {step_found}")
                
            except json.JSONDecodeError:
                print(f"  [{model_name}] Warning: JSON decode error in {file_path}")
        else:
            # Model exists but no matching file found
            pass

    if not models_data:
        print("No matching metrics files found.")
        return

    # 2. Prepare Table Data (Transposing)
    # Columns = Models (sorted alphabetically)
    # Rows = Metrics (sorted alphabetically)
    
    sorted_models = sorted(list(models_data.keys()))
    sorted_metrics = sorted(list(all_metric_keys))

    md_lines = []
    
    # Header Row: Empty first cell, then Model names
    header_row = "| Metric | " + " | ".join(sorted_models) + " |"
    md_lines.append(header_row)
    
    # Separator Row
    separator_row = "| " + " | ".join(["---"] * (len(sorted_models) + 1)) + " |"
    md_lines.append(separator_row)
    
    # Data Rows: Loop through metrics
    for metric in sorted_metrics:
        row_str = f"| {metric} |"
        for model in sorted_models:
            # Get value for this model and metric, default to None
            val = models_data[model].get(metric, None)
            row_str += f" {format_value(val)} |"
        md_lines.append(row_str)

    # 3. Write Output
    os.makedirs(results_dir, exist_ok=True)
    output_filename = f"results_metrics_{args.task}_{args.load_step}.md"
    output_path = os.path.join(results_dir, output_filename)
    
    with open(output_path, 'w') as f:
        f.write("\n".join(md_lines))
        
    print(f"\nSuccessfully wrote transposed results to:\n{output_path}")

if __name__ == "__main__":
    main()
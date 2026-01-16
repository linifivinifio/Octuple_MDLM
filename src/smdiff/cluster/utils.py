"""
Cluster utilities for ETH HPC environment.
Handles path resolution between Scratch (fast, temporary) and Home (permanent).
"""
import os
import platform
import getpass
import shutil
import subprocess
import re

def is_cluster():
    """
    Detect if running on the ETH cluster.
    """
    try:
        # Check /etc/os-release which is standard on modern Linux
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                content = f.read()
                if "Ubuntu" in content or "ID=ubuntu" in content:
                    return True
        # Fallback check
        if "Ubuntu" in platform.version():
            return True
    except Exception:
        pass
    return False

def get_current_username():
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'

def get_scratch_dir(username=None):
    """Returns /work/scratch/{username} or raises error if not on cluster logic."""
    if not username:
        username = get_current_username()
    return f"/work/scratch/{username}"

def copy_final_model_to_home(scratch_run_dir, project_run_dir):
    """
    Smart Sync:
    1. FULL COPY: samples/, logs/, configs/ (History is important here)
    2. LATEST ONLY: checkpoints/, stats/ (Files are cumulative or heavy)
    """
    if scratch_run_dir == project_run_dir:
        return

    print(f"Syncing artifacts from Scratch ({scratch_run_dir}) to Home ({project_run_dir})...")
    
    # Ensure destination exists
    os.makedirs(project_run_dir, exist_ok=True)

    # --- 1. FULL COPY (Lightweight / History needed) ---
    folders_to_sync = ['samples', 'logs', 'configs']
    
    for folder in folders_to_sync:
        src = os.path.join(scratch_run_dir, folder)
        dst = os.path.join(project_run_dir, folder)
        
        if os.path.exists(src):
            try:
                # dirs_exist_ok=True allows updating existing folder
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"  [OK] Synced {folder}/ (Full history)")
            except Exception as e:
                print(f"  [ERR] Failed to sync {folder}/: {e}")

    # --- 2. LATEST ONLY (Heavy / Cumulative files) ---
    # We define the folder name and the regex to identify the step number
    sync_targets = [
        {
            'folder': 'checkpoints', 
            'pattern': re.compile(r'.*_(\d+)\.th$')  # Matches: model_800.th, optim_800.th
        },
        {
            'folder': 'stats', 
            'pattern': re.compile(r'stats_(\d+)\.pt$') # Matches: stats_800.pt
        }
    ]

    for target in sync_targets:
        folder = target['folder']
        pattern = target['pattern']
        
        src_dir = os.path.join(scratch_run_dir, folder)
        dst_dir = os.path.join(project_run_dir, folder)
        
        if os.path.exists(src_dir):
            os.makedirs(dst_dir, exist_ok=True)
            
            # Find the highest step number in this folder
            files = os.listdir(src_dir)
            max_step = -1
            
            for f in files:
                match = pattern.match(f)
                if match:
                    step = int(match.group(1))
                    if step > max_step:
                        max_step = step
            
            if max_step >= 0:
                print(f"  [INFO] Latest step for {folder}/ is {max_step}")
                count = 0
                for f in files:
                    # Sync if it matches the max_step
                    should_copy = False
                    
                    # Check regex match against max_step
                    match = pattern.match(f)
                    if match and int(match.group(1)) == max_step:
                        should_copy = True
                        
                    if "best" in f:
                        should_copy = True
                    
                    if should_copy:
                        src_f = os.path.join(src_dir, f)
                        dst_f = os.path.join(dst_dir, f)
                        shutil.copy2(src_f, dst_f)
                        count += 1
                
                print(f"  [OK] Synced {count} latest files in {folder}/")
            else:
                print(f"  [WARN] No numbered files found in {folder}/")
        else:
            # It's fine if stats/checkpoints don't exist yet (e.g. very early crash)
            pass
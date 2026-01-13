import numpy as np
import os
import shutil
import sys

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

from smdiff.data.octuple.dataset import OctupleDataset

def test_aligned_cropping():
    print("Testing Aligned Cropping...")
    
    # Create dummy data
    # 200 rows, 8 cols
    # Col 1 is position. Let's make it sequence 0..9 repeating
    # This means valid start indices are 0, 10, 20...
    data = np.zeros((200, 8), dtype=np.int64)
    positions = np.array([i % 10 for i in range(200)])
    data[:, 1] = positions
    
    # Set Bar numbers (Col 0) just in case
    bars = np.array([i // 10 for i in range(200)])
    data[:, 0] = bars

    # Save to tmp dir
    tmp_dir = "tests/tmp_data_verification"
    os.makedirs(tmp_dir, exist_ok=True)
    np.save(os.path.join(tmp_dir, "test.npy"), data)

    try:
        # Init dataset with seq_len = 15 (creates some offset from period 10)
        ds = OctupleDataset(tmp_dir, seq_len=15)

        print(f"Dataset loaded with {len(ds)} files.")

        # Fetch 50 samples and verify
        for i in range(50):
            sample = ds[0] # we only have 1 file, idx 0
            
            # Check if start position is 0
            start_pos = sample[0, 1]
            
            # Also check if it's actually valid data from our sequence
            # Our data is simple: pos[k] = k % 10.
            # So if start_pos is 0, sample[1, 1] should be 1.
            next_pos = sample[1, 1]
            
            if start_pos != 0:
                print(f"FAILURE: Sample {i} started at position {start_pos} (Expected 0)")
                sys.exit(1)
                
            if next_pos != 1:
                 print(f"FAILURE: Sample {i} integrity check failed. Next pos {next_pos} != 1")
                 sys.exit(1)

        print("SUCCESS: All 50 samples started at Position 0")
        
    finally:
        # Cleanup
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_aligned_cropping()

import numpy as np
import os

# List your file paths here
files = [
    "data/POP909_trio_octuple.npy", 
    "data/POP909_trio.npy",
    "data/POP909_melody.npy",
    "data/POP909_melody_octuple.npy",
    "data/test.npy"
]

for f in files:
    if not os.path.exists(f):
        print(f"Skipping {f} (Not found)")
        continue
        
    print(f"\n--- Inspecting {f} ---")
    try:
        data = np.load(f, allow_pickle=True)
        print(f"Global Shape: {data.shape}")
        print(f"Global Dtype: {data.dtype}")
        
        # Check first item
        if len(data) > 0:
            item = data[0]
            print(f"Item 0 Type: {type(item)}")
            
            if isinstance(item, np.ndarray):
                print(f"Item 0 Shape: {item.shape}")
                print(f"Item 0 Dtype: {item.dtype}")
                
                # Check for (3, Time) vs (Time, 3)
                if item.ndim == 2:
                    rows, cols = item.shape
                    if rows < 10 and cols > 100:
                        print("  -> LIKELY CHANNEL-FIRST (Channels, Time)")
                    elif rows > 100 and cols < 10:
                        print("  -> LIKELY TIME-FIRST (Time, Channels)")
            else:
                print(f"Item 0 Content (Sample): {item}")
                
    except Exception as e:
        print(f"Error reading {f}: {e}")
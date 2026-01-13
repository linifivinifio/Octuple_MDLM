"""
PyTorch Dataset for Octuple-encoded MIDI data.
This class is deprecated.
"""

import os
import numpy as np
import torch


class OctupleDataset(torch.utils.data.Dataset):
    """Dataset for loading octuple-encoded MIDI files from a directory.
    
    Each .npy file contains a sequence of octuple tokens (L, 8) where each row is:
    [bar, position, program, pitch, duration, velocity, time_sig, tempo]
    """
    
    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        self.data_files = []
        
        # Find all .npy files in the directory
        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.npy'):
                        self.data_files.append(os.path.join(root, file))
        else:
            raise ValueError(f"Data path {data_path} is not a directory")
            
        print(f"Found {len(self.data_files)} data files.")

    def __getitem__(self, idx):
        # Load the file
        file_path = self.data_files[idx]
        data = np.load(file_path)  # Shape: (L, 8)
        
        # Check length
        if data.shape[0] < self.seq_len:
            # Pad if too short
            padding = np.zeros((self.seq_len - data.shape[0], data.shape[1]), dtype=data.dtype)
            data = np.concatenate([data, padding], axis=0)
        
        # Aligned Random Crop
        # We only want to start at the beginning of a bar (Position == 0)
        # to preserve absolute positional embedding alignment.
        if data.shape[0] > self.seq_len:
            # Find all indices where Position (col 1) == 0
            # and that allow for a full sequence length crop
            max_start = data.shape[0] - self.seq_len
            
            # Get boolean mask of valid positions (Position == 0)
            valid_starts_mask = (data[:max_start + 1, 1] == 0)
            valid_start_indices = np.where(valid_starts_mask)[0]
            
            if len(valid_start_indices) > 0:
                start = np.random.choice(valid_start_indices)
            else:
                # Fallback: strict alignment not possible (e.g. no bar starts in valid range)
                # This should be rare for valid MIDI. Use standard random crop.
                start = np.random.randint(0, max_start + 1)
                
            data = data[start:start+self.seq_len]

        # Normalize bar numbers (column 0) to start from 0
        if data.shape[0] > 0:
            data[:, 0] = data[:, 0] - data[:, 0].min()

        return data  # (L, 8)

    def __len__(self):
        return len(self.data_files)

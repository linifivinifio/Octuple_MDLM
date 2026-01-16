import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random

class MusicBERTDataset(Dataset):
    def __init__(self, data_path, max_seq_len=1024, vocab_sizes=[258, 53, 260, 132, 133, 132, 132, 36]):
        # Handle both directory of files and single .npy file (new format)
        if os.path.isfile(data_path) and data_path.endswith('.npy'):
            print(f"Loading dataset from single file: {data_path}")
            self.data = np.load(data_path, allow_pickle=True)
            self.mode = 'memory'
        else:
            print(f"Loading dataset from directory: {data_path}")
            self.files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
            self.mode = 'files'

        self.max_seq_len = max_seq_len
        self.vocab_sizes = vocab_sizes
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.MASK_TOKEN = 1
        self.CLS_TOKEN = 2
        self.EOS_TOKEN = 3
        self.DATA_OFFSET = 4 # Shift data tokens by 4 to avoid collision
        
    def __len__(self):
        if self.mode == 'memory':
            return len(self.data)
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.mode == 'memory':
            # 1. Get raw sequence from memory
            tokens = self.data[idx]
            
            # --- Robustness fixes from base.py ---
            # If x is a 0-d object array wrapping another array, extract it
            if isinstance(tokens, np.ndarray) and tokens.ndim == 0 and tokens.dtype == object:
                tokens = tokens.item()
                
            # If it's a list or object-dtype array, force conversion to int64
            if not isinstance(tokens, np.ndarray) or tokens.dtype == object:
                tokens = np.array(tokens, dtype=np.int64)
        else:
            # 1. Load from file
            file_path = self.files[idx]
            try:
                # Load tokens: (seq_len, 8)
                tokens = np.load(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return self._get_empty_sample()

        if len(tokens) == 0:
            return self._get_empty_sample()

        # 2. Random Crop (Data Augmentation) - BEFORE adding special tokens
        # We need to reserve 2 slots for CLS and EOS tokens
        content_max_len = self.max_seq_len - 2
        
        if len(tokens) > content_max_len:
            # Shift window randomly
            start = np.random.randint(0, len(tokens) - content_max_len + 1)
            tokens = tokens[start : start + content_max_len]

        # Shift tokens to make space for special tokens
        tokens = tokens + self.DATA_OFFSET
        
        # Clip tokens to ensure they are within the vocabulary range
        for i in range(8):
            tokens[:, i] = np.minimum(tokens[:, i], self.vocab_sizes[i] - 1)
        
        # Add CLS and EOS tokens
        # CLS and EOS are duplicated 8 times as per instructions
        cls_token = np.full((1, 8), self.CLS_TOKEN, dtype=tokens.dtype)
        eos_token = np.full((1, 8), self.EOS_TOKEN, dtype=tokens.dtype)
        
        tokens = np.vstack([cls_token, tokens, eos_token])
        
        # (Truncation is handled by Random Crop above, but safety check:)
        if len(tokens) > self.max_seq_len:
             tokens = tokens[:self.max_seq_len]

        # Create labels (copy of tokens)
        labels = tokens.copy()
        
        # Apply Bar-level Masking
        input_ids = self._apply_masking(tokens)
        
        # Padding
        seq_len = len(input_ids)
        pad_len = self.max_seq_len - seq_len
        
        if pad_len > 0:
            pad_token = np.full((pad_len, 8), self.PAD_TOKEN, dtype=int)
            input_ids = np.vstack([input_ids, pad_token])
            labels = np.vstack([labels, pad_token])
            
            # Attention mask: 0 for valid, 1 for padded (for PyTorch Transformer)
            # Wait, PyTorch Transformer src_key_padding_mask: True for padded positions
            attention_mask = np.zeros(self.max_seq_len, dtype=bool)
            attention_mask[seq_len:] = True
        else:
            attention_mask = np.zeros(self.max_seq_len, dtype=bool)
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool)
        }

    def _get_empty_sample(self):
        return {
            'input_ids': torch.full((self.max_seq_len, 8), self.PAD_TOKEN, dtype=torch.long),
            'labels': torch.full((self.max_seq_len, 8), self.PAD_TOKEN, dtype=torch.long),
            'attention_mask': torch.ones(self.max_seq_len, dtype=torch.bool)
        }

    def _apply_masking(self, tokens):
        """
        Apply bar-level masking.
        tokens: (seq_len, 8) - already shifted and with CLS/EOS
        """
        masked_tokens = tokens.copy()
        seq_len = len(tokens)
        
        # Identify bars
        # The first attribute (index 0) is the Bar index.
        # Note: tokens are shifted by DATA_OFFSET.
        # CLS and EOS have special values.
        
        # We only mask data tokens, not CLS/EOS/PAD
        # Data tokens are those >= DATA_OFFSET
        
        # Group indices by Bar value
        bar_indices = {} # bar_value -> list of row indices
        
        for i in range(seq_len):
            # Skip special tokens
            if tokens[i, 0] < self.DATA_OFFSET:
                continue
                
            bar_val = tokens[i, 0]
            if bar_val not in bar_indices:
                bar_indices[bar_val] = []
            bar_indices[bar_val].append(i)
            
        # For each attribute type (0-7)
        for attr_idx in range(8):
            # Collect all units: (bar_val, attr_idx)
            # Since we iterate attr_idx, we just need to iterate bars
            
            for bar_val, indices in bar_indices.items():
                # This (bar_val, attr_idx) is a unit.
                # Decide whether to mask it (15% probability)
                if random.random() < 0.15:
                    # Apply masking to all tokens in this bar for this attribute
                    
                    # 80% -> [MASK]
                    # 10% -> Random token
                    # 10% -> Unchanged
                    
                    r = random.random()
                    if r < 0.8:
                        # Replace with MASK
                        mask_val = self.MASK_TOKEN
                        for idx in indices:
                            masked_tokens[idx, attr_idx] = mask_val
                    elif r < 0.9:
                        # Replace with random token
                        # We need a valid random token for this attribute.
                        # Since we don't know the exact range for each attribute easily here,
                        # we can pick a random value from the vocab range (DATA_OFFSET to vocab_size)
                        # or just random value from 0 to 256 + OFFSET
                        rand_val = random.randint(self.DATA_OFFSET, self.vocab_sizes[attr_idx] - 1)
                        for idx in indices:
                            masked_tokens[idx, attr_idx] = rand_val
                    else:
                        # Keep unchanged
                        pass
                        
        return masked_tokens

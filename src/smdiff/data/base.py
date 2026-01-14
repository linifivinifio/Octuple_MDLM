"""Base dataset utilities for generic data loading."""
import numpy as np
import torch


def cycle(iterable):
    """Infinite iterator that cycles through an iterable."""
    while True:
        for x in iterable:
            yield x


class SimpleNpyDataset(torch.utils.data.Dataset):
    """
    Wraps a numpy array for torch DataLoader compatibility.
    Uses tokenizer_id to determine processing logic (Octuple vs OneHot).
    """
    def __init__(self, data: np.ndarray, seq_len: int, tokenizer_id: str | None = None):
        self.data = data
        self.seq_len = seq_len
        self.tokenizer_id = tokenizer_id or ""
        self.is_octuple = "octuple" in self.tokenizer_id
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 1. Get the raw sequence
        x = self.data[idx]
        
        # --- TYPE FIX ---
        # If x is a 0-d object array wrapping another array, extract it
        if isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype == object:
            x = x.item()
            
        # If it's a list or object-dtype array, force conversion to int64
        # This fixes the "TypeError: can't convert np.ndarray of type numpy.object_"
        if not isinstance(x, np.ndarray) or x.dtype == object:
            x = np.array(x, dtype=np.int64)

        # 2. Random Crop (Data Augmentation)
        length = x.shape[0]
        if length > self.seq_len:
            # Shift window randomly
            start = np.random.randint(0, length - self.seq_len + 1)
            x = x[start : start + self.seq_len]
        
        # 3. Padding (Safety for short sequences)
        elif length < self.seq_len:
            pad_len = self.seq_len - length
            if x.ndim == 1:
                # OneHot (Time,) -> Pad end
                x = np.pad(x, (0, pad_len), 'constant')
            else:
                # Octuple/Trio (Time, Channels) -> Pad time dimension only
                x = np.pad(x, [(0, pad_len), (0, 0)], 'constant', constant_values=-1)

        # 4. Octuple Bar Normalization
        # if self.is_octuple and x.ndim == 2 and x.shape[1] == 8:
        #     # Only run if we have data (non-padding)
        #     if x.shape[0] > 0:
        #         first_bar = x[0, 0]
        #         if first_bar > 0:
        #             x[:, 0] -= first_bar
        #             x[:, 0] = np.maximum(x[:, 0], 0)

        # 5. Return as PyTorch LongTensor
        return torch.from_numpy(x).long()

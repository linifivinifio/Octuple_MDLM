import numpy as np
import torch
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def optim_warmup(H, step, optim):
    lr = H.lr * float(step) / H.warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr



def augment_note_tensor(H, batch):
    """
    Apply pitch augmentation. 
    Handles:
    - Octuple (B, T, 8): Shifts Column 3 (Pitch) only.
    - Trio (B, T, 3): Shifts ALL columns (assuming they are all pitch-based).
    - Melody (B, T): Shifts the sequence.
    """
    if not hasattr(H, 'augment') or not H.augment:
        return batch

    # 1. Convert to Torch
    was_numpy = False
    if isinstance(batch, np.ndarray):
        batch_t = torch.from_numpy(batch).long()
        was_numpy = True
    else:
        batch_t = batch.long()

    B = batch_t.shape[0]
    
    # 2. Identify Data Type & Pitch Column
    is_octuple = False
    pitch_dim = None # If None, shift the whole value
    
    if batch_t.ndim == 3:
        if batch_t.shape[2] == 8:
            # --- OCTUPLE (B, T, 8) ---
            is_octuple = True
            pitch_dim = 3
            # Get Vocab Size (List from config)
            if hasattr(H, 'codebook_size') and isinstance(H.codebook_size, list):
                vocab_size = H.codebook_size[3] 
            else:
                vocab_size = 128
        else:
            # --- TRIO (B, T, 3) ---
            # Treat as OneHot-style: Shift all channels
            is_octuple = False
            pitch_dim = None 
            if hasattr(H, 'codebook_size') and isinstance(H.codebook_size, (list, tuple)):
                vocab_size = int(H.codebook_size[0])
            else:
                vocab_size = 128 # Fallback
    else:
        # --- MELODY ONEHOT (B, T) ---
        is_octuple = False
        pitch_dim = None
        if hasattr(H, 'codebook_size') and isinstance(H.codebook_size, (list, tuple)):
             vocab_size = int(H.codebook_size[0])
        else:
             vocab_size = 128

    # 3. Apply Augmentation
    for i in range(B):
        # Select values to shift
        if is_octuple:
            # Octuple: Only column 3
            vals = batch_t[i, :, pitch_dim]
        elif batch_t.ndim == 3:
            # Trio: All channels (Time, 3)
            vals = batch_t[i]
        else:
            # Melody: (Time,)
            vals = batch_t[i]

        # Filter valid notes (ignore padding=0, start=1 if applicable)
        mask = vals > 1 
        if not mask.any():
            continue

        valid_notes = vals[mask]
        
        min_pitch = valid_notes.min().item()
        max_pitch = valid_notes.max().item()
        
        lower_limit = int(-min_pitch + 2)
        upper_limit = int(vocab_size - max_pitch - 1)
        
        if upper_limit <= lower_limit:
            continue

        shift = int(np.random.randint(lower_limit, upper_limit))
        
        if shift != 0:
            vals[mask] += shift

    if was_numpy:
        return batch_t.numpy()
    return batch_t

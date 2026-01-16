
import torch
import numpy as np
from types import SimpleNamespace
from src.smdiff.models.absorbing_diffusion import AbsorbingDiffusion

def test_masking_strategies():
    print("Starting Masking Strategy Validation...")
    
    # Mock H config
    H = SimpleNamespace(
        codebook_size=[128]*8,
        emb_dim=64,
        latent_shape=[256, 8], # SeqLen 256
        total_steps=1000,
        sampling_batch_size=4,
        loss_type='elbo',
        mask_schedule='random',
        sample_schedule='random',
        masking_strategy=None # To be set per test
    )
    
    # Dummy denoise_fn (not used in q_sample)
    denoise_fn = lambda x, t: [torch.randn(4, 256, 128) for _ in range(8)]
    
    # Mask ID
    mask_id = [129] * 8
    
    # Instantiate Model
    model = AbsorbingDiffusion(H, denoise_fn, mask_id)
    
    # Create Dummy Data: Batch=4, SeqLen=256, 8 channels
    # Channel 0 is Bar index. Let's make it have 4 bars of 64 length each
    x_0 = torch.randint(0, 100, (4, 256, 8))
    for b_idx in range(4):
        for i in range(4):
            x_0[b_idx, i*64:(i+1)*64, 0] = i # Bars 0, 1, 2, 3
            
    device = 'cpu'
    x_0 = x_0.to(device)
    model.to(device)
    
    # Helper to check if mask matches strategy
    def check_mask(mask, strategy, x_0):
        # mask shape: (B, L, 8)
        # x_0 shape: (B, L, 8)
        
        # We need to verify that for masked elements (mask=True), the conditions hold.
        # Note: q_sample_partial also applies random time masking (time_loss_prob).
        # To strictly test structural logic, we can set t = total_steps (prob=1.0)
        pass

    t_full = torch.full((4,), 1000, device=device).long() # t=T -> prob=1.0

    strategies = [
        '1_bar_all', 
        '2_bar_all', 
        '1_bar_attribute', 
        '2_bar_attribute', 
        'rand_attribute',
        'mixed'
    ]
    
    for strategy in strategies:
        print(f"\nTesting Strategy: {strategy}")
        model.masking_strategy = strategy
        
        x_t, _, mask = model.q_sample_partial(x_0, t_full)
        
        # Analysis
        for b_idx in range(4):
            batch_mask = mask[b_idx] # (256, 8)
            masked_indices = torch.nonzero(batch_mask)
            
            if masked_indices.numel() == 0:
                print(f"  Batch {b_idx}: No masking occurred (Possible but rare if t=T?)")
                continue
                
            masked_bars = x_0[b_idx, masked_indices[:, 0], 0].unique()
            masked_channels = masked_indices[:, 1].unique()
            
            print(f"  Batch {b_idx}: Masked Bars: {masked_bars.tolist()}, Masked Channels: {masked_channels.tolist()}")
            
            # Validation assertions
            if strategy == '1_bar_all':
                assert len(masked_bars) == 1, f"Expected 1 bar masked, got {len(masked_bars)}"
                # Channels 3,4,5,7 should be present if bar is fully masked? 
                # Wait, "Mask {Pitch, Dur, Vel, Tempo} ... for 1 random bar"
                # So we expect ONLY channels 3,4,5,7 to be masked.
                assert all(c in [3,4,5,7] for c in masked_channels), f"Unexpected channels masked: {masked_channels}"
                
            elif strategy == '2_bar_all':
                 assert len(masked_bars) <= 2, f"Expected <= 2 bars masked, got {len(masked_bars)}" # <= because random bars could be same
                 assert all(c in [3,4,5,7] for c in masked_channels), f"Unexpected channels masked: {masked_channels}"

            elif strategy == '1_bar_attribute':
                 assert len(masked_bars) == 1, f"Expected 1 bar masked, got {len(masked_bars)}"
                 assert len(masked_channels) == 1, f"Expected 1 channel masked per sample, got {masked_channels}"
                 assert masked_channels[0] in [3,4,5,7], f"Unexpected channel: {masked_channels}"

            elif strategy == '2_bar_attribute':
                 assert len(masked_bars) <= 2, f"Expected <= 2 bars masked, got {len(masked_bars)}"
                 assert len(masked_channels) == 1, f"Expected 1 channel masked per sample, got {masked_channels}"
                 assert masked_channels[0] in [3,4,5,7], f"Unexpected channel: {masked_channels}"
                 
            elif strategy == 'rand_attribute':
                 # rand_attribute is NOT bar aligned, so could be any bars
                 assert len(masked_channels) == 1, f"Expected 1 channel masked per sample, got {masked_channels}"
                 assert masked_channels[0] in [3,4,5,7], f"Unexpected channel: {masked_channels}"

    print("\nAll strategies passed basic structural validation.")

if __name__ == "__main__":
    test_masking_strategies()

import torch
import torch.nn as nn
import os
import sys

# Add PT_encoder_musicBERT to path to allow importing from it without __init__.py
current_dir = os.path.dirname(os.path.abspath(__file__))
pt_encoder_path = os.path.join(current_dir, 'PT_encoder_musicBERT')
if pt_encoder_path not in sys.path:
    sys.path.append(pt_encoder_path)

from smdiff.models.musicbert import MusicBERT, MusicBERTConfig

class MusicBERTDiffusion(nn.Module):
    def __init__(self, H):
        super().__init__()
        
        # Configuration matching the pretrained MusicBERT
        # We use H.codebook_size for the new embeddings/classifiers
        # But we use the structural params from the pretrained model
        
        # CRITICAL FIX: Add +1 to vocab sizes to account for the MASK token.
        # The AbsorbingDiffusion sampler uses H.codebook_size[i] as the mask token index.
        # So if codebook_size is N, the tokens are 0..N-1, and mask is N.
        # The embedding layer must accept indices up to N, so size must be N+1.
        vocab_sizes = [s + 1 for s in H.codebook_size]

        self.config = MusicBERTConfig(
            vocab_sizes=vocab_sizes, # New vocab sizes with mask token support
            element_embedding_size=512,
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            ffn_inner_hidden_size=2048,
            dropout=0.1,
            max_position_embeddings=1024, # Assuming H.NOTES <= 1024
            max_seq_len=1024
        )
        
        self.musicbert = MusicBERT(self.config)
        
        # Load pretrained weights
        # We expect this to be in models/PT_encoder_musicBERT/musicbert_latest.pth
        # We need to construct the absolute path or relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, 'PT_encoder_musicBERT', 'musicbert_latest.pth')
        
        if os.path.exists(weights_path):
            print(f"Loading pretrained MusicBERT weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Filter out keys with shape mismatches (embeddings and classifiers)
            # PyTorch's strict=False does not ignore shape mismatches, so we must remove them manually.
            filtered_state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('element_embeddings') or k.startswith('classifiers'))}
            
            missing, unexpected = self.musicbert.load_state_dict(filtered_state_dict, strict=False)
            print(f"Missing keys (expected due to vocab change): {missing}")
            # print(f"Unexpected keys: {unexpected}")
        else:
            print(f"WARNING: Pretrained weights not found at {weights_path}")

        # Freeze the encoder layers
        # We want to freeze: transformer_encoder, positional_encoding, linear, norm
        # We want to TRAIN: element_embeddings, classifiers
        
        modules_to_freeze = [
            self.musicbert.transformer_encoder,
            self.musicbert.positional_encoding,
            self.musicbert.linear,
            self.musicbert.norm
        ]
        
        for module in modules_to_freeze:
            if isinstance(module, torch.nn.Parameter):
                module.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = False
                    
        print("Frozen MusicBERT encoder layers.")
        
    def forward(self, x, t=None):
        # x: (batch, seq_len, 8)
        # t: unused, but kept for compatibility if needed
        
        # MusicBERT returns a list of logits
        return self.musicbert(x)
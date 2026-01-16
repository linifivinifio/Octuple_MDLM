# %% [markdown]
# <!-- ![alt text](modelstructure.png "model-structure") -->
# <!-- ![alt text](modelstructure.png "model-structure") -->
# 
# <p align="center">
#   <img src="modelstructure.png" alt="model-structure">
# </p>

# %%
# all import statements
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../../../'))
if src_path not in sys.path:
    sys.path.append(src_path)

from smdiff.models.musicbert import MusicBERT, MusicBERTConfig
from smdiff.data.musicbert import MusicBERTDataset

# %%
# Configuration
TARGET_BATCH_SIZE = 256 # The effective batch size we want to simulate
BATCH_SIZE = 32 # Micro-batch size: Small enough to fit in GPU memory (Try 8 or 16)
GRAD_ACCUM_STEPS = TARGET_BATCH_SIZE // BATCH_SIZE # Number of steps to accumulate

MAX_SEQ_LEN = 1024
# Optimized vocab sizes for OctupleMIDI (TimeSig, Tempo, Bar, Pos, Instr, Pitch, Dur, Vel)
# +4 for special tokens (PAD, MASK, CLS, EOS)
VOCAB_SIZES = [258, 53, 260, 132, 133, 132, 132, 36]

# Use absolute path to ensure data is found on the server
# user requested specifically to run with trio_octuple data
DATA_PATH = os.path.abspath(os.path.join(current_dir, '../../../../data/POP909_trio_octuple.npy'))

# Create Dataset
# Ensure the data path exists and has .npy files
if not os.path.exists(DATA_PATH):
    print(f"Warning: {DATA_PATH} does not exist. Please check the path.")
else:
    dataset = MusicBERTDataset(DATA_PATH, max_seq_len=MAX_SEQ_LEN, vocab_sizes=VOCAB_SIZES)
    print(f"Dataset size: {len(dataset)}")

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# %%
# Model Configuration
config = MusicBERTConfig(
    vocab_sizes=VOCAB_SIZES,
    element_embedding_size=512,
    hidden_size=512,
    num_layers=4,
    num_attention_heads=8,
    ffn_inner_hidden_size=2048,
    dropout=0.1,
    max_position_embeddings=MAX_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MusicBERT(config).to(device)

print(model)

# %%


# %% [markdown]
# # Training 

# %%
# Training Configuration
TOTAL_STEPS = 1250*3
WARMUP_STEPS = 250*3
PEAK_LR = 5e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPS = 1e-6

optimizer = Adam(model.parameters(), lr=1.0, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY) # lr set by scheduler

# Learning Rate Scheduler
def lr_lambda(step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS * PEAK_LR
    else:
        # Linear decay
        decay_steps = TOTAL_STEPS - WARMUP_STEPS
        current_decay_step = step - WARMUP_STEPS
        return max(0.0, PEAK_LR * (1 - current_decay_step / decay_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore PAD token (0)

# Ensure checkpoints directory exists
# /home/lziltener/symbolic-music-discrete-diffusion-fork/logs/train_33886.err
os.makedirs('logs/checkpoints', exist_ok=True)
CHECKPOINT_PATH = 'logs/checkpoints/musicbert_latest.pth'

if 'train_loader' in locals():
    # Calculate epochs
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    
    if steps_per_epoch > 0:
        num_epochs = int(np.ceil(TOTAL_STEPS / steps_per_epoch))

        print(f"Micro-batch size: {BATCH_SIZE}")
        print(f"Gradient Accumulation steps: {GRAD_ACCUM_STEPS}")
        print(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
        print(f"Effective steps per epoch: {steps_per_epoch}")
        print(f"Total epochs needed: {num_epochs}")

        # Training Loop
        model.train()
        global_step = 0
        optimizer.zero_grad() # Initialize gradients
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            print(f"Starting Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                logits_list = model(input_ids, attention_mask=attention_mask)
                
                loss = 0
                # Calculate loss for each of the 8 attributes
                for i in range(8):
                    # Use specific vocab size for this attribute
                    vocab_size = VOCAB_SIZES[i]
                    output_flat = logits_list[i].view(-1, vocab_size)
                    target_flat = labels[:, :, i].reshape(-1)
                    loss += criterion(output_flat, target_flat)
                
                # Normalize loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                
                # Add to epoch loss (multiply back to get actual loss value for logging)
                epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                
                # Step optimizer only after accumulating enough gradients
                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                    # Print progress occasionally (based on updates)
                    if global_step % 10 == 0:
                         current_lr = scheduler.get_last_lr()[0]
                         # loss.item() is scaled, so multiply by GRAD_ACCUM_STEPS for display
                         print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}, LR: {current_lr:.6f}")

                    if global_step >= TOTAL_STEPS:
                        break
            
            # Calculate average loss over the number of micro-batches
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
            
            # Save model weights
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Model weights saved to {CHECKPOINT_PATH}")
            
            if global_step >= TOTAL_STEPS:
                print("Reached total training steps.")
                break

    else:
        print("Train loader is empty.")
else:
    print("Train loader not defined.")



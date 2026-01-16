"""
Training loop for symbolic music diffusion models.
Integrates with the unified registry/config system while preserving
compatibility with legacy hparams infrastructure.
"""
import copy
import time
import os
import wandb
import numpy as np
import torch
from tqdm import tqdm

from .data import SimpleNpyDataset
from .utils.log_utils import (
    log, save_model, save_samples, save_stats,
    load_model, load_stats, log_stats
)
from .utils.sampler_utils import get_sampler, get_samples
from .utils.train_utils import EMA, optim_warmup, augment_note_tensor
from .data.base import cycle
from .cluster import copy_final_model_to_home


def main(H):
    """
    Main training loop.
    Args:
        H: Hyperparameters object with all training config
    """

    # --- DATA SETUP ---
    # Set seed for reproducibility before data splitting
    if hasattr(H, 'seed') and H.seed is not None:
        log(f"Setting global seed to {H.seed}")
        torch.manual_seed(H.seed)
        np.random.seed(H.seed)
        # random.seed(H.seed) # if needed

    data_np = np.load(H.dataset_path, allow_pickle=True)
    log("Tokenizer ID: " + H.tokenizer_id)
    midi_data = SimpleNpyDataset(data_np, H.NOTES, tokenizer_id=getattr(H, 'tokenizer_id', None))
    
    if getattr(H, 'wandb', False):
        run_name = H.wandb_name if H.wandb_name else f"{H.model_id}_{H.tracks}_{H.masking_strategy}"
        wandb.init(
            project=H.wandb_project, 
            name=run_name, 
            config=dict(H),
            dir=H.log_dir
        )
    
    # Split train/val
    val_idx = int(len(midi_data) * H.validation_set_size)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(midi_data, range(val_idx, len(midi_data))),
        batch_size=H.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(midi_data, range(val_idx)),
        batch_size=H.batch_size,
    )

    log(f'Total train batches: {len(train_loader)}, eval: {len(val_loader)}')
    if H.epochs:
        H.train_steps = int(H.epochs * len(train_loader))
        log(f'Training for {H.epochs} epochs = {H.train_steps} steps')

    # --- MODEL & OPTIMIZER ---
    sampler = get_sampler(H).cuda()
    optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)
    scaler = torch.amp.GradScaler("cuda")

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)
    else:
        ema_sampler = None

    # --- STATE TRACKING ---
    best_val_loss = float('inf') # Track best validation loss
    history = {
        'losses': [],
        'val_losses': [],
        'mean_losses': [],
        'elbo': [],
        'val_elbos': [],
        'cons_var': ([], [], [], []), 
    }
    
    start_step = 0

    # --- RESUME LOGIC ---
    if H.load_step > 0:
        start_step = H.load_step + 1
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, [H.log_dir]).cuda()
        log("Loaded model checkpoint")
        
        if H.ema:
            try:
                ema_sampler = load_model(ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, [H.log_dir])
            except Exception:
                ema_sampler = copy.deepcopy(sampler)
                
        if H.load_optim:
            optim = load_model(optim, f'{H.sampler}_optim', H.load_step, H.load_dir, [H.log_dir])
            for param_group in optim.param_groups:
                param_group['lr'] = H.lr

        try:
            train_stats = load_stats(H, H.load_step)
            if train_stats:
                for k in history.keys():
                    if k in train_stats:
                        val = train_stats[k]
                        if isinstance(val, np.ndarray): val = val.tolist()
                        history[k] = val
                
                # Restore best_val_loss if it exists, otherwise calculate from history
                if 'best_val_loss' in train_stats:
                    best_val_loss = train_stats['best_val_loss']
                elif len(history['val_losses']) > 0:
                    best_val_loss = min(history['val_losses'])
                    
                H.steps_per_log = train_stats.get("steps_per_log", H.steps_per_log)
                log(f"Resumed stats from step {H.load_step}. Best Val Loss: {best_val_loss:.4f}")
        except Exception:
            log('No stats file found, starting fresh stats.')

    # --- HELPER FUNCTIONS ---
    
    def run_validation(step):
        """Runs validation loop, logs metrics, and saves BEST model."""
        nonlocal best_val_loss # Access outer scope variable
        log(f"Evaluating step {step}")
        sampler.eval()
        
        valid_loss, valid_elbo, num_batches = 0.0, 0.0, 0
        for x in tqdm(val_loader, desc="Validation", leave=False):
            with torch.no_grad():
                stats = sampler.train_iter(x.cuda())
                valid_loss += stats['loss'].item()
                if H.sampler == 'absorbing' and 'vb_loss' in stats:
                    valid_elbo += stats['vb_loss'].item()
                num_batches += 1
        
        if num_batches > 0:
            valid_loss /= num_batches
            valid_elbo /= num_batches

        history['val_losses'].append(valid_loss)
        history['val_elbos'].append(valid_elbo)
        
        log_msg = f"Validation at step {step}: loss={valid_loss:.6f}"
        if H.sampler == 'absorbing':
            log_msg += f", elbo={valid_elbo:.6f}"
        
        # --- BEST MODEL LOGIC ---
        if valid_loss < best_val_loss:
            log_msg += " (NEW BEST!)"
            best_val_loss = valid_loss
            # Save "best" version specifically
            # 1. Save Main Model (Overwrite)
            ckpt_dir = os.path.join(H.log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(sampler.state_dict(), best_path)
            
            # 2. Save EMA Model (Overwrite)
            if H.ema:
                ema_best_path = os.path.join(ckpt_dir, "ema_best.pt")
                torch.save(ema_sampler.state_dict(), ema_best_path)
            
            # 3. Save Info File (So we know WHICH step was best)
            stats_dir = os.path.join(H.log_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            
            info_path = os.path.join(stats_dir, "best_stats.txt")
            with open(info_path, "w") as f:
                f.write(f"step: {step}\nloss: {best_val_loss:.6f}\n")
        
        log(log_msg)
        
        if getattr(H, 'wandb', False):
            val_metrics = {
                "val/loss": valid_loss, 
                "val/best_loss": best_val_loss,
                "val/step": step
            }
            if H.sampler == 'absorbing':
                val_metrics["val/elbo"] = valid_elbo
            wandb.log(val_metrics, step=step)

    def run_sampling(step):
        """Generates and saves samples."""
        log(f"Sampling step {step}")
        model_to_sample = ema_sampler if H.ema else sampler
        model_to_sample.eval()
        
        samples = get_samples(
            model_to_sample,
            H.sample_steps,
            b=getattr(H, 'show_samples', 16)
        )
        save_samples(samples, step, H.log_dir)

    def save_checkpoint(step):
        """Saves periodic checkpoint and stats."""
        log(f"Saving checkpoint at step {step}")
        save_model(sampler, H.sampler, step, H.log_dir)
        save_model(optim, f'{H.sampler}_optim', step, H.log_dir)
        if H.ema:
            save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

        stats_to_save = {k: np.array(v) for k, v in history.items()}
        stats_to_save.update({
            'steps_per_log': H.steps_per_log,
            'steps_per_eval': H.steps_per_eval,
            'best_val_loss': best_val_loss # Persist this!
        })
        save_stats(H, stats_to_save, step)

    def flush_logs(step, current_losses_buffer, current_vb_losses_buffer, step_time):
        """Calculates mean loss from buffer and logs to console/wandb."""
        if len(current_losses_buffer) == 0: return

        mean_loss = np.mean(current_losses_buffer)
        history['mean_losses'].append(mean_loss)
        
        mean_vb_loss = 0.0
        if len(current_vb_losses_buffer) > 0:
            mean_vb_loss = np.mean(current_vb_losses_buffer)
        
        log_data = {'mean_loss': mean_loss, 'step_time': step_time}
        if H.sampler == 'absorbing' and len(current_vb_losses_buffer) > 0:
            log_data['vb_loss'] = mean_vb_loss
            
        log_stats(step, log_data)

        if getattr(H, 'wandb', False):
            wandb_metrics = {
                "train/loss": mean_loss,
                "train/step_time": step_time,
                "train/lr": optim.param_groups[0]['lr'],
                "train/step": step
            }
            if H.sampler == 'absorbing' and len(current_vb_losses_buffer) > 0:
                wandb_metrics["train/vb_loss"] = mean_vb_loss
                
            wandb.log(wandb_metrics, step=step)

    # --- TRAINING LOOP ---
    
    log(f"Starting training loop from {start_step} to {H.train_steps}...")
    log(f"Sampler params total: {sum(p.numel() for p in sampler.parameters())}")
    
    train_iterator = cycle(train_loader)
    current_losses_buffer = [] 
    current_vb_losses_buffer = []

    for step in range(start_step, H.train_steps):
        sampler.train()
        if H.ema: ema_sampler.train()
        step_start_time = time.time()

        # 1. Warmup
        if H.warmup_iters and step <= H.warmup_iters:
            optim_warmup(H, step, optim)

        # 2. Train Step
        x = augment_note_tensor(H, next(train_iterator))
        x = x.cuda(non_blocking=True)

        if H.amp:
            optim.zero_grad()
            with torch.amp.autocast("cuda"):
                stats = sampler.train_iter(x)
            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = sampler.train_iter(x)
            if torch.isnan(stats['loss']).any():
                log(f'Skipping step {step} with NaN loss')
                continue
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        # 3. Record stats
        loss_val = stats['loss'].item()
        current_losses_buffer.append(loss_val)
        history['losses'].append(loss_val) 
        
        if 'vb_loss' in stats:
            vb_val = stats['vb_loss'].item()
            current_vb_losses_buffer.append(vb_val)
            history['elbo'].append(vb_val)

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            ema.update_model_average(ema_sampler, sampler)

        # 4. Periodic Actions
        if step % H.steps_per_log == 0:
            flush_logs(step, current_losses_buffer, current_vb_losses_buffer, time.time() - step_start_time)
            current_losses_buffer = []
            current_vb_losses_buffer = []

        if step % H.steps_per_sample == 0 and step > 0:
            run_sampling(step)

        if H.steps_per_eval and step % H.steps_per_eval == 0 and step > 0:
            run_validation(step)

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_checkpoint(step)

    # --- FINAL WRAP UP ---
    log("Training loop finished. Performing final operations...")
    
    if len(current_losses_buffer) > 0:
        flush_logs(H.train_steps, current_losses_buffer, current_vb_losses_buffer, 0.0)

    run_validation(H.train_steps)
    run_sampling(H.train_steps)
    save_checkpoint(H.train_steps)

    if getattr(H, 'wandb', False):
        wandb.finish()
        
    if hasattr(H, 'project_log_dir') and H.log_dir != H.project_log_dir:
        copy_final_model_to_home(H.log_dir, H.project_log_dir)
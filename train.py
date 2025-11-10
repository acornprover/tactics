"""
Training script for BPE tokenized GPT on proof tactics.
"""

import os
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from model import GPT, ModelConfig
from data import load_data, create_dataloaders, decode, sample_generation
from config import model_config, training_config
from tokenizer import BPETokenizer


def get_lr(step: int, config) -> float:
    """
    Cosine learning rate schedule with warmup.

    Args:
        step: Current training step
        config: Training configuration

    Returns:
        Learning rate for this step
    """
    # Warmup
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    # Cosine decay
    # Calculate total steps (approximately)
    max_steps = config.max_epochs * 1000  # Rough estimate
    if step > max_steps:
        return config.min_lr

    decay_ratio = (step - config.warmup_steps) / (max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, data_loader, device, max_batches=None):
    """
    Estimate loss on a dataset.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None = all)

    Returns:
        Average loss
    """
    model.eval()
    losses = []

    for i, (x, y) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def train(model_cfg=None, train_cfg=None):
    """
    Main training loop.

    Args:
        model_cfg: Model configuration (uses default if None)
        train_cfg: Training configuration (uses default if None)
    """
    # Use default configs if not provided
    if model_cfg is None:
        model_cfg = model_config
    if train_cfg is None:
        train_cfg = training_config

    # Set random seed
    torch.manual_seed(train_cfg.seed)

    # Create checkpoint directory
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # Set device
    device = train_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BPETokenizer.load(model_cfg.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Update model config with actual vocab size from tokenizer
    model_cfg.vocab_size = tokenizer.vocab_size

    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset = load_data(
        tokenizer, train_cfg.data_dir, train_cfg.train_split, model_cfg.context_length
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, train_cfg.batch_size
    )

    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = GPT(model_cfg).to(device)

    # Compile model (PyTorch 2.0+)
    if train_cfg.compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )

    # Training state
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    if train_cfg.resume_from:
        print(f"\nResuming from checkpoint: {train_cfg.resume_from}")
        checkpoint = torch.load(train_cfg.resume_from, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming from epoch {start_epoch}, step {global_step}")
        print(f"Best val loss so far: {best_val_loss:.4f}")

    # Training loop
    print("\nStarting training...")
    print(f"{'Epoch':<6} {'Step':<8} {'Train Loss':<12} {'Val Loss':<12} {'LR':<10} {'Time':<8}")
    print("-" * 70)

    for epoch in range(start_epoch, train_cfg.max_epochs):
        model.train()
        epoch_start = time.time()

        # Training epoch
        train_loss_accum = 0.0
        for batch_idx, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        ):
            x, y = x.to(device), y.to(device)

            # Forward pass
            _, loss = model(x, y)

            # Backward pass
            loss = loss / train_cfg.gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % train_cfg.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip
                )

                # Update learning rate
                lr = get_lr(global_step, train_cfg)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

            train_loss_accum += loss.item() * train_cfg.gradient_accumulation_steps

            # Logging
            if global_step % train_cfg.log_interval == 0 and batch_idx > 0:
                avg_train_loss = train_loss_accum / train_cfg.log_interval
                train_loss_accum = 0.0

            # Evaluation
            if global_step % train_cfg.eval_interval == 0 and global_step > 0:
                val_loss = estimate_loss(model, val_loader, device, max_batches=50)
                lr = optimizer.param_groups[0]["lr"]

                print(
                    f"{epoch + 1:<6} {global_step:<8} {loss.item():<12.4f} {val_loss:<12.4f} {lr:<10.2e} {time.time() - epoch_start:<8.1f}s"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_cfg.__dict__,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "tokenizer_path": model_cfg.tokenizer_path,
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(train_cfg.checkpoint_dir, "best_model.pt"),
                    )
                    print(f"  → Saved best model (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1

                # Sample generation
                if global_step % (train_cfg.eval_interval * 5) == 0:
                    print("\n  Sample generation:")
                    prompt = "@T\ntheorem "
                    sample = sample_generation(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=200,
                        temperature=train_cfg.sample_temperature,
                        top_p=train_cfg.sample_top_p,
                        device=device,
                    )
                    print(f"  {sample[:300]}...\n")

            # Checkpointing
            if global_step % train_cfg.save_interval == 0 and global_step > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": model_cfg.__dict__,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "tokenizer_path": model_cfg.tokenizer_path,
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        train_cfg.checkpoint_dir, f"checkpoint_step_{global_step}.pt"
                    ),
                )

        # End of epoch evaluation
        val_loss = estimate_loss(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"{epoch + 1:<6} {global_step:<8} {'-':<12} {val_loss:<12.4f} {lr:<10.2e} {epoch_time:<8.1f}s"
        )

        # Early stopping
        if patience_counter >= train_cfg.patience:
            print(f"\nEarly stopping: validation loss hasn't improved for {train_cfg.patience} evaluations")
            break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()

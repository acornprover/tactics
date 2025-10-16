"""
Training configuration for character-level GPT on proof tactics.

Based on the training plan in TRAINING_PLAN.md.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the GPT model architecture."""

    # Model architecture (~5M params - faster training)
    vocab_size: int = 256  # Byte-level encoding
    context_length: int = 256  # Sequence length (reduced for speed)
    d_model: int = 256  # Model dimension
    n_layers: int = 6  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads (head_dim = 32)
    d_mlp: int = 1024  # MLP dimension (4 × d_model)

    # Regularization
    dropout: float = 0.1  # Dropout probability

    # Architecture choices
    use_bias: bool = False  # Bias in linear layers (False for modern GPT)
    tie_embeddings: bool = True  # Tie input/output embeddings

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_mlp == 4 * self.d_model, "d_mlp should be 4 × d_model"


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Data
    data_dir: str = "data/proofs"  # Directory containing proof files
    train_split: float = 0.9  # 90% train, 10% validation

    # Training
    batch_size: int = 32  # Number of sequences per batch
    max_epochs: int = 30  # Maximum number of epochs
    gradient_accumulation_steps: int = 1  # For larger effective batch size

    # Optimizer
    learning_rate: float = 3e-4  # Peak learning rate
    weight_decay: float = 0.1  # AdamW weight decay
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.95  # Adam beta2
    grad_clip: float = 1.0  # Gradient clipping

    # Learning rate schedule
    warmup_steps: int = 1000  # LR warmup steps
    min_lr: float = 3e-5  # Minimum learning rate for cosine decay

    # Checkpointing and logging
    checkpoint_dir: str = "checkpoints"
    resume_from: str = ""  # Path to checkpoint to resume from (empty = start fresh)
    log_interval: int = 100  # Log every N steps
    eval_interval: int = 1000  # Evaluate every N steps
    save_interval: int = 5000  # Save checkpoint every N steps

    # Early stopping
    patience: int = 10  # Stop if val loss doesn't improve for N evals

    # Sampling (for validation)
    sample_temperature: float = 0.8  # Sampling temperature
    sample_top_p: float = 0.9  # Nucleus sampling threshold
    num_samples: int = 3  # Number of samples to generate during validation

    # System
    device: str = "cuda"  # cuda or cpu
    compile: bool = False  # Use torch.compile (PyTorch 2.0+)
    seed: int = 42  # Random seed for reproducibility


# Default configurations
model_config = ModelConfig()
training_config = TrainingConfig()

# Character-Level GPT Training Plan for Proof Tactics

## Dataset Overview
- **Source**: `data/training.txt`
- **Size**: 3.68M characters
- **Format**: Structured proof certificates with @T (theorem), @G (goal), @C (constraint), @P (premise) markers
- **Domain**: Mathematical proofs for algebraic structures

## Training Strategy

### Data Recycling
- **Epochs**: ~60 epochs
- **Effective tokens**: ~216M (3.6M chars × 60 epochs)
- **Rationale**: Chinchilla scaling (~20 tokens/param) suggests 10-12M parameters for this token budget

### Train/Validation Split
- **Training**: 90% (~3.3M chars)
- **Validation**: 10% (~368K chars)
- **Note**: Consider stratified split to ensure diverse proof types in validation set

## Model Architecture (~10M parameters)

### Core Parameters
```
vocab_size = 256          # Byte-level encoding for math symbols
context_length = 512      # Start with 512, increase to 1024 if needed
d_model = 320             # Model dimension
n_layers = 8              # Transformer layers
n_heads = 10              # Attention heads (head_dim = 32)
d_mlp = 1280              # MLP dimension (4 × d_model)
```

### Architecture Choices
- **Normalization**: RMSNorm (more stable than LayerNorm)
- **Activation**: GELU or SiLU in MLP
- **Embeddings**: Tied input/output embeddings (reduces params)
- **Positional**: Learned or RoPE (rotary position embeddings)
- **Dropout**: 0.1-0.2 (start lower, increase if overfitting)

### Total Parameter Count
Approximately 10-12M parameters

## Training Hyperparameters

### Optimizer
```
optimizer = AdamW
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
```

### Learning Rate Schedule
```
peak_lr = 3e-4
warmup_steps = 1000
schedule = cosine_decay
final_lr = 3e-5
total_epochs = 50-80
```

### Batch Configuration
```
batch_strategy = "pack sequences"
tokens_per_batch = 64k-128k
use_gradient_accumulation = true
```

### Regularization
```
dropout = 0.1              # Start conservative
label_smoothing = 0.0      # Consider 0.1 if overfitting
```

## Training Procedure

### Monitoring
- **Primary metric**: Validation loss (cross-entropy)
- **Secondary metrics**:
  - Training loss
  - Perplexity
  - Gradient norms
  - Learning rate
- **Domain-specific**: Track generated proof structure validity

### Early Stopping
- Monitor validation loss
- Stop if loss plateaus for 5-10 epochs
- Watch for signs of memorization:
  - Val loss bottoms then rises
  - Near-duplicate samples in generations
  - Train/val loss divergence

### Checkpointing
- Save checkpoint every epoch
- Keep best 3 checkpoints by val loss
- Save optimizer state for resumption

## Inference & Sampling

### Sampling Strategy
```
temperature = 0.8-1.0     # Start at 0.8 for more focused outputs
top_p = 0.9               # Nucleus sampling
top_k = None              # Use top_p instead
```

### Evaluation
- Generate sample proofs from validation set theorems
- Check structural validity (@T, @G, @C, @P markers)
- Manual review of proof coherence
- Compare to ground truth proofs

## Scaling Adjustments

### If Underfitting (val loss plateaus high)
- Increase model size → 15M params
- Increase context length → 1024
- Decrease dropout → 0.05
- Train longer

### If Overfitting (val loss rises, train continues to drop)
- Decrease model size → 5-7M params
- Increase dropout → 0.2-0.3
- Add label smoothing → 0.1
- Reduce epochs
- Increase weight decay → 0.2

## Implementation Notes

### Data Preprocessing
- Byte-level tokenization (vocab=256)
- Pack sequences efficiently to minimize padding
- Ensure proof boundaries are respected (don't split mid-proof)

### Hardware Considerations
- Model size targets modest GPU (e.g., single RTX 3090/4090)
- Gradient accumulation for larger effective batch sizes
- Mixed precision (fp16/bf16) for faster training

### Reproducibility
- Set random seeds
- Log all hyperparameters
- Version control code and configs
- Save training curves

## Success Criteria

1. **Convergence**: Val loss decreases steadily and plateaus
2. **Generalization**: Train/val loss gap < 0.5 nats
3. **Generation quality**: Proofs follow correct structure
4. **Domain validity**: Generated tactics are mathematically sensible
5. **Diversity**: Model doesn't memorize training proofs

## Next Steps

1. Implement data loader with sequence packing
2. Build character-level GPT architecture
3. Set up training loop with logging
4. Run baseline experiment with suggested hyperparameters
5. Iterate based on validation metrics and generated samples

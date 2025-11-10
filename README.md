# tactics

Code for training and evaluating a "tactics" model, which suggests proof steps generatively.

## Quick Start

### 1. Generate Training Data

Generate proof data from acornlib:

```bash
acorn --lib /path/to/acornlib training ./data/proofs
```

This creates one proof file per theorem in `data/proofs/`. Format uses @T (theorem prefix), @G (goal), @C (counterfactual), @P (proof) markers.

When you regenerate proof data, you typically want to clean up old checkpoints and tokenizers.

```bash
rm ./checkpoints/*
```

### 2. Train Tokenizer

Train a BPE tokenizer on your proof dataset:

```bash
uv run train_tokenizer.py
```

This will:

- Read all proof files from `data/proofs/`
- Train a BPE tokenizer with vocab_size=4096
- Save `tokenizer.json` to `checkpoints/` directory
- Show compression ratio (expect ~3-4x compression)

Options: `--vocab_size 4096`, `--data_dir data/proofs`, `--output_dir checkpoints`

### 3. Train Model

Train the model from scratch:

```bash
uv run train.py
```

This will:

- Load tokenizer from `checkpoints/tokenizer.json`
- Tokenize proof files from `data/proofs/`
- Train GPT model (~9M parameters)
- Save checkpoints to `checkpoints/`
- Save best model as `checkpoints/best_model.pt`

Configuration (edit `config.py`):

- `vocab_size`: 4096 (set from tokenizer)
- `context_length`: 256 tokens
- `max_epochs`: 30
- `batch_size`: 32
- `learning_rate`: 3e-4

### 4. Resume Training (Optional)

Resume from checkpoint:

```bash
uv run resume_training.py
```

Or specify in `config.py`: `training_config.resume_from = "checkpoints/checkpoint_step_5000.pt"`

### 5. Export to ONNX

Export for use by Acorn:

```bash
uv run export_onnx.py
```

This creates a timestamped directory in `export/` with both files:

```
export/tactics-2025-11-10-14-30-45/
├── model.onnx
└── tokenizer.json
```

Custom export directory: `uv run export_onnx.py checkpoints/best_model.pt export/my-model`

## Tokenization

Uses **BPE (Byte-Pair Encoding)** instead of character-level:

**Benefits:**

- Efficient compression: "theorem add_comm" → ~4 tokens
- Longer effective context: 256 tokens ≈ 768-1024 characters
- Domain-specific vocabulary learned from your proofs
- Common terms like "theorem", "proof", "forall" become single tokens

**Files:**

- `checkpoints/tokenizer.json` - Trained tokenizer (vocabulary and merge rules)

## Model Architecture

- **Type**: BPE-tokenized GPT (decoder-only transformer)
- **Vocab size**: 4096
- **Context length**: 256 tokens (~768-1024 chars effective)
- **Model dimension**: 256
- **Layers**: 6
- **Attention heads**: 8
- **Parameters**: ~9M
- **Features**: RMSNorm, causal self-attention, tied embeddings

## Troubleshooting

**"Tokenizer file not found"** - Run `uv run train_tokenizer.py` first

**Out of memory** - Reduce `batch_size` or `context_length` in `config.py`

**Vocab size mismatch** - Delete old checkpoints after retraining tokenizer

---

## Reference Information

### Dataset

- ~6,600 proof files, 3.7MB text
- Mathematical proofs for algebraic structures
- Structured format: @T, @G, @C, @P markers

### Training Details

- Optimizer: AdamW (lr=3e-4, weight_decay=0.1)
- Schedule: Cosine decay, 1000-step warmup, min_lr=3e-5
- Data split: 90% train / 10% validation
- Early stopping: Patience of 10 evaluations

### Tokenization Stats

- Dataset size: 3.7MB, ~6,600 files
- Tokenized: ~1.1M tokens (3.4x compression ratio)
- Effective context: 256 tokens ≈ 870 chars average

### Files

- `train.py` - Training script
- `model.py` - GPT architecture
- `data.py` - Data loading
- `config.py` - Configuration
- `tokenizer.py` - BPE tokenizer
- `train_tokenizer.py` - Train tokenizer
- `export_onnx.py` - ONNX export

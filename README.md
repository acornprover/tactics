# tactics

Code for training and evaluating a "tactics" model, which suggests proof steps generatively.

## Data Generation

To generate training data from acornlib:

```bash
acorn --lib /path/to/acornlib --generate-training ./data/proofs
```

This dumps out all proof certificates in a structured format, with one proof per file in `data/proofs/`.

## Training

### Initial Training

To train a model from scratch:

```bash
uv run train.py
```

This will:

- Load proof files from `data/proofs/`
- Create a character-level GPT model (~5M parameters)
- Train for the number of epochs specified in `config.py`
- Save checkpoints to `checkpoints/`
- Save the best model as `checkpoints/best_model.pt`

Training configuration can be modified in `config.py`:

- `max_epochs`: Number of training epochs
- `context_length`: Maximum sequence length (default: 256)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Peak learning rate (default: 3e-4)

### Resume from Checkpoint

To resume training from a saved checkpoint:

```bash
uv run resume_training.py
```

This loads `checkpoints/best_model.pt` and continues training from where it left off.

Alternatively, you can manually specify a checkpoint by modifying `config.py`:

```python
training_config.resume_from = "checkpoints/checkpoint_step_5000.pt"
```

## Export to ONNX

To export the trained model to ONNX format for use by the Acorn binary:

```bash
uv run export_onnx.py
```

This will:

- Load the best checkpoint from `checkpoints/best_model.pt`
- Export to `tactics-YYYY-MM-DD-HH-MM-SS.onnx` with a timestamp
- Verify the ONNX model is valid
- Test inference with ONNX Runtime

You can also specify a custom checkpoint or output path:

```bash
uv run export_onnx.py checkpoints/checkpoint_step_10000.pt custom_name.onnx
```

## Model Architecture

- **Type**: Character-level GPT (decoder-only transformer)
- **Vocab size**: 256 (byte-level encoding)
- **Context length**: 256 characters
- **Model dimension**: 256
- **Layers**: 6
- **Attention heads**: 8
- **Parameters**: ~5M
- **Features**: RMSNorm, causal self-attention, tied embeddings

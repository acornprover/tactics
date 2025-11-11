"""
Export trained model to ONNX format for Rust inference.
"""

import os
import shutil
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from datetime import datetime
from pathlib import Path
from model import GPT, ModelConfig


class GPTWithFlattenedCache(nn.Module):
    """Wrapper to flatten KV cache inputs/outputs for ONNX export."""

    def __init__(self, model: GPT):
        super().__init__()
        self.model = model
        self.n_layers = model.config.n_layers

    def forward(self, input_ids, *past_kv_flat):
        """
        Forward with flattened cache inputs.

        Args:
            input_ids: Token indices (B, T)
            *past_kv_flat: Flattened past key-values (key0, val0, key1, val1, ...)

        Returns:
            logits and flattened present key-values
        """
        # Reconstruct past_key_values list from flattened inputs
        past_key_values = []
        for i in range(self.n_layers):
            k = past_kv_flat[i * 2]
            v = past_kv_flat[i * 2 + 1]
            past_key_values.append((k, v))

        # Run model
        logits, present_key_values = self.model(
            input_ids,
            targets=None,
            past_key_values=past_key_values,
            use_cache=True
        )

        # Flatten present_key_values for output
        outputs = [logits]
        for k, v in present_key_values:
            outputs.extend([k, v])

        return tuple(outputs)


def export_to_onnx(
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_dir: str = None,
    opset_version: int = 18,
):
    """
    Export PyTorch model to ONNX format along with tokenizer.

    Args:
        checkpoint_path: Path to the trained checkpoint
        output_dir: Where to save the export (default: timestamped directory in export/)
        opset_version: ONNX opset version
    """
    # Generate timestamped directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = f"export/tactics-{timestamp}"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {output_dir}")
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Reconstruct model config (tokenizer_path is stored separately in checkpoint)
    model_config_dict = dict(checkpoint["model_config"])
    tokenizer_path = checkpoint["tokenizer_path"]

    # Remove tokenizer_path from model_config if it's there
    model_config_dict.pop("tokenizer_path", None)

    model_cfg = ModelConfig(**model_config_dict)
    model_cfg.tokenizer_path = tokenizer_path
    print(f"Model config: {model_cfg}")

    # Create and load model
    base_model = GPT(model_cfg)
    base_model.load_state_dict(checkpoint["model"])
    base_model.eval()

    # Wrap model for ONNX export with flattened cache
    model = GPTWithFlattenedCache(base_model)
    model.eval()

    # Export ONNX model
    print("\nExporting model to ONNX with KV caching...")
    onnx_path = output_path / "model.onnx"

    # Create dummy input for cached generation
    # Input: single token (batch_size=1, seq_len=1)
    dummy_input = torch.randint(0, model_cfg.vocab_size, (1, 1), dtype=torch.long)

    # Create dummy past key-value cache (flattened)
    # For first token: use cache_len=1 with zeros (will be masked out)
    # Shape: (batch_size, n_heads, cache_seq_len, head_dim)
    dummy_past_kv_flat = []
    cache_seq_len = 1  # Minimal size for ONNX export
    head_dim = model_cfg.d_model // model_cfg.n_heads

    for _ in range(model_cfg.n_layers):
        k = torch.zeros(1, model_cfg.n_heads, cache_seq_len, head_dim)
        v = torch.zeros(1, model_cfg.n_heads, cache_seq_len, head_dim)
        dummy_past_kv_flat.extend([k, v])

    # Build input and output names for ONNX
    input_names = ["input_ids"]
    output_names = ["logits"]

    for i in range(model_cfg.n_layers):
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present_key_values.{i}.key", f"present_key_values.{i}.value"])

    # Configure dynamic axes for variable sequence lengths
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    for i in range(model_cfg.n_layers):
        # Past KV can have variable sequence length
        dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "kv_sequence_length"}
        dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "kv_sequence_length"}
        # Present KV will have kv_sequence_length + 1
        dynamic_axes[f"present_key_values.{i}.key"] = {0: "batch_size", 2: "total_sequence_length"}
        dynamic_axes[f"present_key_values.{i}.value"] = {0: "batch_size", 2: "total_sequence_length"}

    # Prepare inputs tuple: (input_ids, *past_kv_flat)
    export_args = (dummy_input,) + tuple(dummy_past_kv_flat)

    # Export with KV caching and dynamic axes (using legacy exporter)
    torch.onnx.export(
        model,
        export_args,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
        dynamo=False,  # Use legacy exporter for better dynamic_axes support
    )

    print(f"✓ Exported model to {onnx_path}")

    # Convert external data to embedded (single file)
    print("  Converting to single-file format...")
    onnx_model_proto = onnx.load(str(onnx_path))

    # Save with embedded data
    onnx.save(
        onnx_model_proto,
        str(onnx_path),
        save_as_external_data=False,
    )

    # Clean up the .data file if it exists
    data_file = str(onnx_path) + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"  ✓ All weights embedded in model.onnx")

    # Copy tokenizer
    print("\nCopying tokenizer...")
    tokenizer_src = Path(model_cfg.tokenizer_path) / "tokenizer.json"
    tokenizer_dst = output_path / "tokenizer.json"

    if not tokenizer_src.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_src}")

    shutil.copy(tokenizer_src, tokenizer_dst)
    print(f"✓ Copied tokenizer to {tokenizer_dst}")

    # Create config.json
    print("\nCreating config.json...")
    config = {
        "model_type": "gpt",
        "vocab_size": model_cfg.vocab_size,
        "context_length": model_cfg.context_length,
        "d_model": model_cfg.d_model,
        "n_layers": model_cfg.n_layers,
        "n_heads": model_cfg.n_heads,
        "head_dim": model_cfg.d_model // model_cfg.n_heads,
        "d_mlp": model_cfg.d_mlp,
        "dropout": model_cfg.dropout,
        "use_bias": model_cfg.use_bias,
        "tie_embeddings": model_cfg.tie_embeddings,
        "use_cache": True
    }

    import json
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Created config at {config_path}")

    # Verify the ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Test inference with ONNX Runtime
    print("\nTesting ONNX Runtime inference with KV caching...")
    ort_session = ort.InferenceSession(str(onnx_path))

    # Test with single token input and cache
    test_input = np.random.randint(0, model_cfg.vocab_size, (1, 1), dtype=np.int64)

    # Create initial cache (zeros for first token)
    ort_inputs = {"input_ids": test_input}
    for i in range(model_cfg.n_layers):
        cache_shape = (1, model_cfg.n_heads, 1, head_dim)
        ort_inputs[f"past_key_values.{i}.key"] = np.zeros(cache_shape, dtype=np.float32)
        ort_inputs[f"past_key_values.{i}.value"] = np.zeros(cache_shape, dtype=np.float32)

    # Run first inference
    ort_outputs = ort_session.run(None, ort_inputs)

    # Check outputs
    expected_logits_shape = (1, 1, model_cfg.vocab_size)
    actual_logits_shape = ort_outputs[0].shape

    if actual_logits_shape != expected_logits_shape:
        raise RuntimeError(f"Logits shape mismatch: expected {expected_logits_shape}, got {actual_logits_shape}")

    # Verify we got present_key_values back
    expected_num_outputs = 1 + (2 * model_cfg.n_layers)  # logits + (key, value) per layer
    if len(ort_outputs) != expected_num_outputs:
        raise RuntimeError(f"Expected {expected_num_outputs} outputs, got {len(ort_outputs)}")

    # Verify cache shapes (should be (1, n_heads, 2, head_dim) after first token)
    for i in range(model_cfg.n_layers):
        key_idx = 1 + (2 * i)
        val_idx = 1 + (2 * i) + 1
        expected_cache_shape = (1, model_cfg.n_heads, 2, head_dim)  # cache_len=1 + new_token=1
        if ort_outputs[key_idx].shape != expected_cache_shape:
            raise RuntimeError(
                f"Layer {i} key cache shape mismatch: expected {expected_cache_shape}, got {ort_outputs[key_idx].shape}"
            )
        if ort_outputs[val_idx].shape != expected_cache_shape:
            raise RuntimeError(
                f"Layer {i} value cache shape mismatch: expected {expected_cache_shape}, got {ort_outputs[val_idx].shape}"
            )

    # Verify logits are reasonable (not all zeros, not NaN, not infinite)
    logits = ort_outputs[0]
    if np.isnan(logits).any():
        raise RuntimeError("Output contains NaN values")
    if np.isinf(logits).any():
        raise RuntimeError("Output contains infinite values")
    if np.abs(logits).max() < 1e-6:
        raise RuntimeError("Output appears to be all zeros")

    print(f"  ✓ Logits shape: {actual_logits_shape}")
    print(f"  ✓ Cache shape: {expected_cache_shape} (per layer)")
    print(f"  ✓ Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  ✓ Inference test passed!")

    print("\n" + "="*60)
    print("✓ Export successful!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"  ├── model.onnx")
    print(f"  ├── tokenizer.json")
    print(f"  └── config.json")
    print(f"\nModel info:")
    print(f"  Vocab size: {model_cfg.vocab_size}")
    print(f"  Context length: {model_cfg.context_length}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return output_dir


if __name__ == "__main__":
    import sys

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    export_to_onnx(checkpoint_path, output_dir)

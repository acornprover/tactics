"""
Export trained model to ONNX format for Rust inference.
"""

import os
import shutil
import torch
import onnx
import onnxruntime as ort
import numpy as np
from datetime import datetime
from pathlib import Path
from model import GPT, ModelConfig


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
    model = GPT(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Export ONNX model
    print("\nExporting model to ONNX...")
    onnx_path = output_path / "model.onnx"

    # Create dummy input (batch_size=1, seq_len=model.context_length)
    # Using fixed shapes for better compatibility with ONNX Runtime
    dummy_input = torch.randint(
        0, model_cfg.vocab_size, (1, model_cfg.context_length), dtype=torch.long
    )

    # Export with fixed shapes (more reliable for inference)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
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
        "d_mlp": model_cfg.d_mlp,
        "dropout": model_cfg.dropout,
        "use_bias": model_cfg.use_bias,
        "tie_embeddings": model_cfg.tie_embeddings
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
    print("\nTesting ONNX Runtime inference...")
    ort_session = ort.InferenceSession(str(onnx_path))

    # Test with model's context length
    test_input = np.random.randint(
        0, model_cfg.vocab_size, (1, model_cfg.context_length), dtype=np.int64
    )
    ort_inputs = {"input": test_input}
    ort_outputs = ort_session.run(None, ort_inputs)

    expected_shape = (1, model_cfg.context_length, model_cfg.vocab_size)
    actual_shape = ort_outputs[0].shape

    if actual_shape != expected_shape:
        raise RuntimeError(f"Output shape mismatch: expected {expected_shape}, got {actual_shape}")

    # Verify logits are reasonable (not all zeros, not NaN, not infinite)
    logits = ort_outputs[0]
    if np.isnan(logits).any():
        raise RuntimeError("Output contains NaN values")
    if np.isinf(logits).any():
        raise RuntimeError("Output contains infinite values")
    if np.abs(logits).max() < 1e-6:
        raise RuntimeError("Output appears to be all zeros")

    print(f"  ✓ Shape check: {actual_shape}")
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

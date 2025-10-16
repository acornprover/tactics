"""
Export trained model to ONNX format for Rust inference.
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from datetime import datetime
from model import GPT, ModelConfig


def export_to_onnx(
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_path: str = None,
    opset_version: int = 18,
):
    """
    Export PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to the trained checkpoint
        output_path: Where to save the ONNX model (default: timestamped filename)
        opset_version: ONNX opset version
    """
    # Generate timestamped filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_path = f"tactics-{timestamp}.onnx"

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Reconstruct model config
    model_cfg = ModelConfig(**checkpoint["model_config"])
    print(f"Model config: {model_cfg}")

    # Create and load model
    model = GPT(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("\nExporting to ONNX...")

    # Create dummy input (batch_size=1, seq_len=model.context_length)
    # Using fixed shapes for better compatibility with ONNX Runtime
    dummy_input = torch.randint(
        0, model_cfg.vocab_size, (1, model_cfg.context_length), dtype=torch.long
    )

    # Export with fixed shapes (more reliable for inference)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
    )

    print(f"✓ Exported to {output_path}")

    # Convert external data to embedded (single file)
    print("\nConverting to single-file format...")
    onnx_model_proto = onnx.load(output_path)

    # Save with embedded data
    onnx.save(
        onnx_model_proto,
        output_path,
        save_as_external_data=False,
    )

    # Clean up the .data file if it exists
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"✓ Removed external data file, all weights now embedded in {output_path}")

    # Verify the ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Test inference with ONNX Runtime
    print("\nTesting ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)

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

    print("\n✓ ONNX export successful!")
    print(f"\nModel saved to: {output_path}")
    print(f"Vocab size: {model_cfg.vocab_size}")
    print(f"Max context length: {model_cfg.context_length}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return output_path


if __name__ == "__main__":
    import sys

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    export_to_onnx(checkpoint_path, output_path)

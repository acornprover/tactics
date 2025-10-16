"""
Export trained model to ONNX format for Rust inference.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from datetime import datetime
from model import GPT, ModelConfig


def export_to_onnx(
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_path: str = None,
    opset_version: int = 17,
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
    dummy_input = torch.randint(
        0, model_cfg.vocab_size, (1, model_cfg.context_length), dtype=torch.long
    )

    # Export with dynamic axes for flexible batch size and sequence length
    dynamic_axes = {
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    print(f"✓ Exported to {output_path}")

    # Verify the ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Test inference with ONNX Runtime
    print("\nTesting ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)

    # Test with different sequence lengths
    for seq_len in [10, 50, model_cfg.context_length]:
        test_input = np.random.randint(
            0, model_cfg.vocab_size, (1, seq_len), dtype=np.int64
        )
        ort_inputs = {"input": test_input}
        ort_outputs = ort_session.run(None, ort_inputs)

        print(f"  seq_len={seq_len}: input shape {test_input.shape} -> output shape {ort_outputs[0].shape}")

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

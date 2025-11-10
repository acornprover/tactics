"""
Script to train a BPE tokenizer on the proof dataset.

This should be run once before training the model to create the tokenizer.
"""

import argparse
from pathlib import Path
from tokenizer import BPETokenizer


def load_proof_texts(data_dir: str):
    """
    Load all proof texts from the data directory.

    Args:
        data_dir: Directory containing proof .txt files

    Yields:
        Text content of each proof file
    """
    data_path = Path(data_dir)
    proof_files = sorted(data_path.glob("*.txt"))

    print(f"Found {len(proof_files)} proof files")

    for proof_file in proof_files:
        with open(proof_file, "r", encoding="utf-8") as f:
            yield f.read()


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on proof dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/proofs",
        help="Directory containing proof .txt files"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=4096,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save trained tokenizer"
    )

    args = parser.parse_args()

    print(f"Training BPE tokenizer with vocab_size={args.vocab_size}")
    print(f"Loading texts from: {args.data_dir}")

    # Load all proof texts
    texts = list(load_proof_texts(args.data_dir))
    print(f"Loaded {len(texts)} proof files")

    # Calculate total size
    total_chars = sum(len(text) for text in texts)
    print(f"Total characters: {total_chars:,}")

    # Train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts, vocab_size=args.vocab_size)

    # Save tokenizer
    print(f"\nSaving tokenizer to: {args.output_dir}")
    tokenizer.save(args.output_dir)

    print(f"\nTokenizer training complete!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test the tokenizer on a sample
    if texts:
        sample_text = texts[0][:200]  # First 200 chars
        tokens = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(tokens)

        print("\n" + "="*80)
        print("Sample tokenization test:")
        print("="*80)
        print(f"Original text ({len(sample_text)} chars):")
        print(sample_text[:100] + "...")
        print(f"\nTokens ({len(tokens)} tokens):")
        print(tokens[:20], "...")
        print(f"\nCompression ratio: {len(sample_text) / len(tokens):.2f}x")
        print(f"\nDecoded text matches: {decoded == sample_text}")


if __name__ == "__main__":
    main()

"""
BPE Tokenizer for the GPT model using HuggingFace tokenizers library.
"""

from pathlib import Path
from typing import List, Optional
import json


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer using HuggingFace tokenizers library.

    This tokenizer learns subword units from the training data, which can
    provide better compression and capture domain-specific patterns.
    """

    def __init__(self, tokenizer: Optional["HFTokenizer"] = None, vocab_size: int = 4096):
        """
        Initialize BPE tokenizer.

        Args:
            tokenizer: Pre-trained HuggingFace tokenizer instance (optional)
            vocab_size: Target vocabulary size (used when training new tokenizer)
        """
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text using BPE tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Train or load a tokenizer first.")

        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text using BPE tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Train or load a tokenizer first.")

        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return self._vocab_size

    def train(self, texts: List[str], vocab_size: int = 4096) -> None:
        """
        Train a new BPE tokenizer on the provided texts.

        Args:
            texts: List of text strings to train on
            vocab_size: Target vocabulary size
        """
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        # Initialize a BPE tokenizer
        self._tokenizer = HFTokenizer(BPE(unk_token="<UNK>"))

        # Use byte-level pre-tokenization (similar to GPT-2)
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = ByteLevelDecoder()

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<UNK>", "<PAD>"],
            show_progress=True,
        )

        # Train the tokenizer
        self._tokenizer.train_from_iterator(texts, trainer=trainer)
        self._vocab_size = vocab_size

    def save(self, path: str) -> None:
        """
        Save BPE tokenizer to disk.

        Args:
            path: Directory path to save tokenizer files
        """
        if self._tokenizer is None:
            raise RuntimeError("Cannot save uninitialized tokenizer")

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save the tokenizer
        self._tokenizer.save(str(path_obj / "tokenizer.json"))

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load BPE tokenizer from disk.

        Args:
            path: Directory path containing tokenizer files

        Returns:
            BPETokenizer instance
        """
        from tokenizers import Tokenizer as HFTokenizer

        path_obj = Path(path)
        tokenizer_path = path_obj / "tokenizer.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        # Load tokenizer
        hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))

        # Get vocab size from the loaded tokenizer
        vocab_size = hf_tokenizer.get_vocab_size()

        return cls(tokenizer=hf_tokenizer, vocab_size=vocab_size)

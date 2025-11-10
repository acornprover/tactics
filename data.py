"""
Data loading and preprocessing for BPE tokenized training.
"""

import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
from tokenizer import BPETokenizer


class ProofDataset(Dataset):
    """Dataset that loads individual proof files and creates sliding window chunks."""

    def __init__(self, proof_files: List[str], tokenizer: BPETokenizer, context_length: int):
        """
        Args:
            proof_files: List of paths to proof files
            tokenizer: BPE tokenizer instance
            context_length: Maximum sequence length
        """
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.chunks = []

        # Precompute all chunks from all proofs
        for proof_file in proof_files:
            with open(proof_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Tokenize using BPE
            tokens = tokenizer.encode(text)

            # Create sliding windows with 50% overlap
            stride = context_length // 2

            if len(tokens) <= context_length:
                # Short proof: just use it as-is
                self.chunks.append(tokens)
            else:
                # Long proof: create multiple chunks
                for i in range(0, len(tokens) - stride, stride):
                    chunk = tokens[i : i + context_length]
                    # Only keep substantial chunks (at least half context length)
                    if len(chunk) >= context_length // 2:
                        self.chunks.append(chunk)

        print(f"Created {len(self.chunks):,} training chunks from {len(proof_files):,} proofs")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return input/target sequences for a chunk.

        Args:
            idx: Index of the chunk

        Returns:
            (input_seq, target_seq) where target_seq is shifted by 1
        """
        tokens = self.chunks[idx]

        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Pad if necessary (need context_length + 1 for input/target pairs)
        if len(tokens) < self.context_length + 1:
            padding = torch.full(
                (self.context_length + 1 - len(tokens),), 0, dtype=torch.long
            )
            tokens = torch.cat([tokens, padding])
        elif len(tokens) > self.context_length + 1:
            # Truncate to exact size
            tokens = tokens[: self.context_length + 1]

        # Input is all but last token, target is all but first token
        x = tokens[:-1]
        y = tokens[1:]

        return x, y


def load_data(
    tokenizer: BPETokenizer,
    data_dir: str = "data/proofs",
    train_split: float = 0.9,
    context_length: int = 256,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[ProofDataset, ProofDataset]:
    """
    Load proof files and split into train and validation sets.

    Args:
        tokenizer: BPE tokenizer instance
        data_dir: Directory containing proof files
        train_split: Fraction of data to use for training
        context_length: Maximum sequence length
        shuffle: Whether to shuffle proof files before splitting
        seed: Random seed for shuffling

    Returns:
        (train_dataset, val_dataset)
    """
    # Get all proof files
    proof_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    if len(proof_files) == 0:
        raise ValueError(f"No proof files found in {data_dir}")

    print(f"Found {len(proof_files):,} proof files in {data_dir}")

    # Shuffle files if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(proof_files)
        print(f"Shuffled proof files (seed={seed})")

    # Split into train and validation by file list
    split_idx = int(len(proof_files) * train_split)
    train_files = proof_files[:split_idx]
    val_files = proof_files[split_idx:]

    print(f"Train: {len(train_files):,} proofs")
    print(f"Val: {len(val_files):,} proofs")

    # Create datasets
    train_dataset = ProofDataset(train_files, tokenizer, context_length)
    val_dataset = ProofDataset(val_files, tokenizer, context_length)

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: ProofDataset,
    val_dataset: ProofDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle within epoch
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def decode(tokens: torch.Tensor, tokenizer: BPETokenizer) -> str:
    """
    Decode tokens back to string using the tokenizer.

    Args:
        tokens: Tensor of token indices
        tokenizer: BPE tokenizer instance

    Returns:
        Decoded string
    """
    if tokens.dim() > 1:
        tokens = tokens[0]  # Take first sequence if batched

    # Convert to list of ints
    tokens = tokens.cpu().tolist()

    return tokenizer.decode(tokens)


def sample_generation(
    model,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    """
    Generate text from the model.

    Args:
        model: The language model
        tokenizer: BPE tokenizer instance
        prompt: Starting prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        Generated text
    """
    model.eval()

    # Encode the prompt using tokenizer
    prompt_tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions for the last position
            logits = model(tokens)
            logits = logits[:, -1, :] / temperature

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

    return decode(tokens[0], tokenizer)

"""
Data loading and preprocessing for character-level training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class CharDataset(Dataset):
    """Character-level dataset for proof tactics."""

    def __init__(self, data: str, context_length: int):
        """
        Args:
            data: String of training data
            context_length: Maximum sequence length
        """
        self.data = data
        self.context_length = context_length

        # Convert to byte-level encoding (vocab_size = 256)
        self.tokens = torch.tensor([ord(c) % 256 for c in data], dtype=torch.long)

    def __len__(self):
        # Number of possible sequences
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sequence and its target (next tokens).

        Args:
            idx: Index of the sequence

        Returns:
            (input_seq, target_seq) where target_seq is shifted by 1
        """
        # Get sequence of length context_length + 1
        chunk = self.tokens[idx : idx + self.context_length + 1]

        # Input is all but last token, target is all but first token
        x = chunk[:-1]
        y = chunk[1:]

        return x, y


def load_data(
    data_path: str, train_split: float = 0.9, context_length: int = 512
) -> Tuple[CharDataset, CharDataset]:
    """
    Load and split data into train and validation sets.

    Args:
        data_path: Path to the training data file
        train_split: Fraction of data to use for training
        context_length: Maximum sequence length

    Returns:
        (train_dataset, val_dataset)
    """
    # Read the data
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read()

    print(f"Loaded {len(data):,} characters from {data_path}")

    # Split into train and validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data):,} characters")
    print(f"Val: {len(val_data):,} characters")

    # Create datasets
    train_dataset = CharDataset(train_data, context_length)
    val_dataset = CharDataset(val_data, context_length)

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: CharDataset,
    val_dataset: CharDataset,
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
        shuffle=True,
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


def decode(tokens: torch.Tensor) -> str:
    """
    Decode tokens back to string.

    Args:
        tokens: Tensor of token indices

    Returns:
        Decoded string
    """
    if tokens.dim() > 1:
        tokens = tokens[0]  # Take first sequence if batched

    return "".join(chr(int(t)) for t in tokens.cpu().numpy())


def sample_generation(
    model,
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
        prompt: Starting prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        Generated text
    """
    model.eval()

    # Encode the prompt
    tokens = torch.tensor([[ord(c) % 256 for c in prompt]], dtype=torch.long).to(device)

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

    return decode(tokens[0])

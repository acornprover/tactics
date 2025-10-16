"""
Resume training from a checkpoint.
"""

from train import train
from config import model_config, training_config

# Update config to resume from best checkpoint
training_config.resume_from = "checkpoints/best_model.pt"

# Train will continue from where it left off
train(model_config, training_config)

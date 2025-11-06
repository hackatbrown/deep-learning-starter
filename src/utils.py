import os
import yaml
import random
import numpy as np
import torch

def load_config(path="src/config.yaml"):
    """Load configuration from a YAML file."""
    with open(path, "r") as file:
        return yaml.safe_load(file)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)

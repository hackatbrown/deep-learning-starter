import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tensorflow.keras.datasets import mnist as tf_mnist

from model_pytorch import SimpleNet
from model_tensorflow import build_model
from evaluate import evaluate_pytorch, evaluate_tensorflow
from utils import ensure_dir, load_config, set_seed


def train_pytorch(config):
    """Train a simple PyTorch model on MNIST."""
    # TODO: Set device
    # TODO: Load MNIST data (with transforms)
    # TODO: Initialize model and optimizer

    # TODO: Implement training loop
    # for each epoch:
    #   - zero gradients
    #   - forward pass
    #   - compute loss
    #   - backward pass
    #   - update weights

    # TODO: Evaluate model on test set
    # TODO: Save trained model
    pass


def train_tensorflow(config):
    """Train a simple TensorFlow model on MNIST."""
    # TODO: Load and normalize MNIST data
    # TODO: Build model
    # TODO: Train model (model.fit)
    # TODO: Evaluate model
    # TODO: Save trained model
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], default="pytorch")
    parser.add_argument("--config", default="src/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)

    if args.framework == "pytorch":
        train_pytorch(config)
    else:
        train_tensorflow(config)


if __name__ == "__main__":
    main()

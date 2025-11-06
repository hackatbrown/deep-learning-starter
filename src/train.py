import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tensorflow.keras.datasets import mnist as tf_mnist
import numpy as np

from model_pytorch import SimpleNet
from model_tensorflow import build_model
from evaluate import evaluate_pytorch, evaluate_tensorflow
from utils import ensure_dir, load_config, set_seed


def train_pytorch(config):
    """Train a simple PyTorch model on MNIST."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=config["batch_size"], shuffle=False)

    # Model and optimizer
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{config['epochs']} - Loss: {loss.item():.4f}")

    # Evaluation
    accuracy = evaluate_pytorch(model, test_loader, device)

    # Save model
    ensure_dir("experiments/saved_models")
    torch.save(model.state_dict(), "experiments/saved_models/model_pytorch.pth")
    print(f"✅ Model saved! Test accuracy: {accuracy * 100:.2f}%\n")


def train_tensorflow(config):
    """Train a simple TensorFlow model on MNIST."""
    print("Using TensorFlow backend")

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = build_model()
    model.summary()

    # Training
    history = model.fit(
        x_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.1,
        verbose=1
    )

    # Evaluation
    acc = evaluate_tensorflow(model, x_test, y_test)

    # Save model
    ensure_dir("experiments/saved_models")
    model.save("experiments/saved_models/model_tensorflow.h5")
    print(f"✅ Model saved! Test accuracy: {acc * 100:.2f}%\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], default="pytorch",
                        help="Choose which deep learning framework to use.")
    parser.add_argument("--config", default="src/config.yaml", help="Path to config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)

    print(f"Starting training with {args.framework}...\n")
    if args.framework == "pytorch":
        train_pytorch(config)
    else:
        train_tensorflow(config)


if __name__ == "__main__":
    main()

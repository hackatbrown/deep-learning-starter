import torch

def evaluate_pytorch(model, test_loader, device):
    """Evaluate a PyTorch model on the test dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            # TODO: Move data to device
            # TODO: Forward pass
            # TODO: Get predictions (argmax)
            # TODO: Count correct predictions
            pass
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def evaluate_tensorflow(model, test_images, test_labels):
    """Evaluate a TensorFlow model on the test dataset."""
    # TODO: Use model.evaluate
    # TODO: Print accuracy
    pass

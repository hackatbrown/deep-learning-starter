import torch
import torch.nn.functional as F

def evaluate_pytorch(model, test_loader, device):
    """Evaluate a PyTorch model on the test dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def evaluate_tensorflow(model, test_images, test_labels):
    """Evaluate a TensorFlow model on the test dataset."""
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc

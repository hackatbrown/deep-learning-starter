import torch
from torchvision import datasets, transforms

def load_mnist(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

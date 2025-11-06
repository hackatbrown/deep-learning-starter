import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # TODO: Define layers (3 fully connected layers)
        # Example:
        # self.fc1 = nn.Linear(28 * 28, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # TODO: Flatten input
        # TODO: Apply ReLU after first and second layers
        # TODO: Return final output
        pass

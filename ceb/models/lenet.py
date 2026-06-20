import torch.nn as nn
import torch

class LeNet5(nn.Module):
    def __init__(self, output_shape, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.output_shape = output_shape

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            # always check what dimensionality is here!
            nn.Linear(256, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, output_shape),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

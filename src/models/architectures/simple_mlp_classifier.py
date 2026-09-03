import torch
from torch import nn

class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim : int, num_classes : int):
        super(SimpleMLPClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
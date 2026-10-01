import torch
from torch import nn

class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim : int, num_classes : int, hidden_dim : int = 512, dropout_rate : float = 0.0):
        super(SimpleMLPClassifier, self).__init__()

        self.dense1 = nn.Linear(input_dim, hidden_dim)

        self.relu1 = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x
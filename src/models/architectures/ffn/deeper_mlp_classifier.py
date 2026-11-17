import torch
from torch import nn
from typing import List

class DeeperMLPClassifier(nn.Module):
    def __init__(self, input_dim : int, num_classes : int, hidden_dim1_size : int = 512, hidden_dim2_size : int = 254, dropout_rate : float = 0.0, activation_fn : nn.Module = nn.ReLU):
        """
        Allows to have two hidden dim layers
        """
        super(DeeperMLPClassifier, self).__init__()

        # define network
        self.dense1 = nn.Linear(input_dim, hidden_dim1_size)
        self.act1 = activation_fn()

        self.dense2 = nn.Linear(hidden_dim1_size, hidden_dim2_size)
        self.act2 = activation_fn()

        self.dropout = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_dim2_size, num_classes)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        """
        """
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dropout(x) # change pos of dropout?
        x = self.dense3(x)

        return x


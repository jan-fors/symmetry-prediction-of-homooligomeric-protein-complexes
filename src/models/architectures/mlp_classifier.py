import torch
from torch import nn
from typing import List

class MLPClassifier(nn.Module):
    def __init__(self, input_dim : int, num_classes : int, hidden_dims : List[int] = [512, 254], dropout_rate : float = 0.0, activation_fn : nn.Module = nn.ReLU):
        """
        Allows to have two hidden dim layers
        """
        super(MLPClassifier, self).__init__()

        self.dense1 = nn.Linear(input_dim, hidden_dims[0])
        self.act1 = activation_fn()

        self.dense2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.act2 = activation_fn()

        self.dropout = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_dims[1], num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dropout(x) # change pos of dropout?
        x = self.dense3(x)

        return x
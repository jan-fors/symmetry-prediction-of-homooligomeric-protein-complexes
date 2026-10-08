import torch
from torch import nn

class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim : int, num_classes : int, hidden_dim : int = 512, dropout_rate : float = 0.0, layer_norm : bool = False, activation_fn : nn.Module = nn.ReLU):
        super(SimpleMLPClassifier, self).__init__()

        self.dense1 = nn.Linear(input_dim, hidden_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = nn.Identity()

        self.act1 = activation_fn()

        self.dropout = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        x = self.dense1(x)
        x = self.layer_norm(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x
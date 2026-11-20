"""
RobertaLMHead robertalmhead.py
Code taken from https://github.com/microsoft/seq2symm/tree/main
"""

import torch
from torch import nn
from src.models.layers.activation.gelu import gelu

class RobertaLMHead(nn.Module):
    """
    Head for masked language modeling.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate : float = 0.1, layer_norm : bool = False):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer added
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim) # Changed to LayerNorm
        else:
            self.layer_norm = nn.Identity()
        self.dense2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, mask):   ## ESM2 features size: torch.Size([1, N, 1280])

        x = self.dense1(features)

        x = gelu(x)
        x = self.dropout(x)  # Dropout applied after activation
        x = self.layer_norm(x)
        # average over residues in protein.. 
        if mask is not None:
            mask = mask.unsqueeze(-1).to(x.dtype)  # [B, residues, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(1)
        x = self.dense2(x)
        return x

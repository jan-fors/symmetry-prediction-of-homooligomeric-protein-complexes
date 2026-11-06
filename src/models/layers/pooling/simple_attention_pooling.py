"""
AttentionPooling attention_pooling.py

Layer that does attention pooling by calculating weights for each residue

Uses a basic query vector that can be learned and pools using the attention weights
"""

import torch
from torch import nn

class AttentionPooling(nn.Module):
    def __init__(self):
        """
        TODO weighted average over length usuing calculated weights from attention
        """
        super().__init__()
        pass

    def forward():
        pass
"""
AttentionPooling attention_pooling.py

Layer that does attention pooling by calculating weights for each residue

Uses a basic query vector that can be learned and pools using the attention weights
"""
# imports
import torch
from torch import nn
from src.models.layers.attention.hetero_query_attention import HeteroQueryAttention

class HeteroAttentionPooling(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int = 1, max_sequence_len : int = 1024):
        """
        TODO weighted average over length usuing calculated weights from attention
        """
        super().__init__()

        self.attn = HeteroQueryAttention(embed_dim, num_heads, max_sequence_len)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        """
        """

        out, weights = self.attn(x)
        pooled = out.mean(dim=1) 

        return pooled
"""
AttentionPooling attention_pooling.py

Layer that does attention pooling by calculating weights for each residue

Uses a basic query vector that can be learned and pools using the attention weights
"""
# imports
import torch
from torch import nn
from src.models.layers.attention.homo_query_attention import HomoQueryAttention

class HomoAttentionPooling(nn.Module):
    """
    Turns a padded batch of chains into one vector per chain.
    Contains the LearnedQueryAttention layer, so other models only need this layer.
    """
 
    def __init__(self, embed_dim, num_queries: int = 1, mode: str = "concat"):
        super().__init__()
        if mode not in ("concat", "mean"):
            raise ValueError(f"mode must be 'concat' or 'mean', got {mode!r}")
        self.mode = mode
        self.attention = HomoQueryAttention(embed_dim, num_queries)
        # input size for the next layer (e.g. the first nn.Linear of the MLP)
        self.output_dim = num_queries * embed_dim if mode == "concat" else embed_dim
 
    def forward(self, x, key_padding_mask=None):
        """
        x:                (batch, seq_len, embed_dim)
        key_padding_mask: (batch, seq_len) bool, True = padding, ignore this token
        returns:
          pooled:  (batch, num_queries * embed_dim) for "concat",
                   (batch, embed_dim)               for "mean"
          weights: (batch, num_queries, seq_len)
        """
        out, weights = self.attention(x, key_padding_mask)   # (b, n, d)
        if self.mode == "concat":
            pooled = out.flatten(start_dim=1)   # answers side by side
        else:
            pooled = out.mean(dim=1)            # average over the queries
        return pooled, weights     
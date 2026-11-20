"""
Attention pooling with a fixed number of learned queries.

"""

import torch
import torch.nn as nn
 
 
class HomoQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_queries: int = 1):
        super().__init__()
        self.num_queries = num_queries
        # one learned query vector per "question", independent of the chain length
        self.Q = nn.Parameter(torch.randn(num_queries, embed_dim) / embed_dim ** 0.5)
        self.scale = embed_dim ** -0.5
 
    def forward(self, x, key_padding_mask=None):
        """
        x:                (batch, seq_len, embed_dim)
        key_padding_mask: (batch, seq_len) bool, True = padding, ignore this token
        returns:
          out:     (batch, num_queries, embed_dim)   one answer per query
          weights: (batch, num_queries, seq_len)     how much each query uses each residue
        """
        # compare every query with every residue: (n, d) @ (b, d, s) -> (b, n, s)
        scores = self.Q @ x.transpose(-2, -1) * self.scale
 
        if key_padding_mask is not None:
            # padding gets -inf, so exp(-inf) = 0 and it gets no weight
            scores = scores.masked_fill(key_padding_mask[:, None, :], float("-inf"))
 
        weights = scores.softmax(dim=-1)   # (b, n, s), every row sums to 1
        out = weights @ x                  # (b, n, s) @ (b, s, d) -> (b, n, d)
        return out, weights
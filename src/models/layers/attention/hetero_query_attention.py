
"""
Multi-head attention with directly learned query matrices.
 
The embedding dimension d is split into num_heads parts of size head_dim = d / num_heads.
Each head has its own learned queries and attends using only its slice of the embedding.
 
    Q = learned parameter          (heads, seq_len, head_dim)
    K = V = x split into heads     (batch, heads, seq_len, head_dim)
    out_h = softmax(Q_h K_h^T / sqrt(head_dim)) V_h
    out   = concatenation of all heads   -> (batch, seq_len, d)
 
With num_heads=1 this is exactly the original single-head layer.
"""
import torch
import torch.nn as nn
 
 
class HeteroQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads: int = 1, max_seq_len: int = 1024):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # one learned query vector per head and output position
        self.Q = nn.Parameter(
            torch.randn(num_heads, max_seq_len, self.head_dim) / self.head_dim ** 0.5
        )
        self.scale = self.head_dim ** -0.5
 
    def forward(self, x):
        """
        x:                (batch, seq_len, embed_dim), seq_len <= max_seq_len
        attn_mask:        (seq_len, seq_len) bool, True = position i may NOT see token j
        key_padding_mask: (batch, seq_len) bool, True = ignore this token
        returns:
          out:     (batch, seq_len, embed_dim)
          weights: (batch, num_heads, seq_len, seq_len)
        """
        batch, seq_len, embed_dim = x.shape
        if seq_len > self.Q.shape[1]:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.Q.shape[1]}")
 
        Q = self.Q[:, :seq_len]   # (h, s, hd), same for every batch item
 
        # split the embedding into heads: (b, s, d) -> (b, s, h, hd) -> (b, h, s, hd)
        K = x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = K
 
        # every head compares its queries with its slice of the tokens
        scores = torch.einsum("hqd,bhkd->bhqk", Q, K) * self.scale   # (b, h, s, s)
 
        weights = scores.softmax(dim=-1)   # (b, h, s, s), every row sums to 1
        out = weights @ V                  # (b, h, s, hd)
 
        # put the heads back together: (b, h, s, hd) -> (b, s, h, hd) -> (b, s, d)
        out = out.transpose(1, 2).reshape(batch, seq_len, embed_dim)
        return out, weights

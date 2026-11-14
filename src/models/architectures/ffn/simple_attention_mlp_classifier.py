"""

"""
# imports
import torch
from torch import nn
from src.models.layers.pooling.homo_attention_pooling import HomoAttentionPooling

class SimpleAttentionMLPClassifier(nn.Module):
    def __init__(self, embed_dim : int, num_classes : int, num_heads : int = 1, hidden_dim : int = 512, dropout_rate : float = 0.0, layer_norm : bool = False, activation_fn : nn.Module = nn.ReLU):
        super(SimpleAttentionMLPClassifier, self).__init__()

        self.attn_pool = HomoAttentionPooling(embed_dim, num_heads)

        self.dense1 = nn.Linear(int(self.attn_pool.output_dim), hidden_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = nn.Identity()

        self.act1 = activation_fn()

        self.dropout = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        """
        """
        x, _weights = self.attn_pool(x)
        x = self.dense1(x)
        x = self.layer_norm(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x
"""
MultiheadedRobertaLMHead multiheadedrobertalmhead.py
Code taken from https://github.com/microsoft/seq2symm/tree/main
"""

import torch
from torch import nn
from src.models.layers.activation.gelu import gelu
from src.models.architectures.robertaml.robertalmhead import RobertaLMHead

class MultiheadedRobertaLMHead(nn.Module):
    def __init__(self,
                    embed_dim,
                    hidden_layer_sizes=[256, 100],
                    output_dim=20,layer_norm : bool = False):
        
        super().__init__()

        n_classes = output_dim
        self.common_head = RobertaLMHead(embed_dim, hidden_layer_sizes[0], hidden_layer_sizes[1], layer_norm=layer_norm)

        # fine heads
        self.fine_heads_one_per_class = []
        for cl in range(n_classes):
            self.head2_fc1 = nn.Linear(hidden_layer_sizes[1], 1)
            self.fine_heads_one_per_class.append(self.head2_fc1)
        self.fine_heads_one_per_class = nn.ModuleList(self.fine_heads_one_per_class)

        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_layer_sizes[1]) # Changed to LayerNorm
        else:
            self.layer_norm1 = nn.Identity()


    def forward(self, features):
        x = self.common_head(features)     
        x = self.layer_norm1(gelu(x))

        self.fine_heads_one_per_class = self.fine_heads_one_per_class.to(features.device)
        out = torch.cat([self.fine_heads_one_per_class[head](x) for head in range(len(self.fine_heads_one_per_class))], dim=1)

        return out
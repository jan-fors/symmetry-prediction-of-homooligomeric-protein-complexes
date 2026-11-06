"""
MultiheadedRobertaLMHead multiheadedrobertalmhead.py
Code taken from https://github.com/microsoft/seq2symm/tree/main
"""

import torch
from torch import nn
from src.models.layers.activation.gelu import gelu
from src.models.architectures.robertaml.robertalmhead import RobertaLMHead

class MultiheadedNoSharedRobertaLMHead(nn.Module):
    """
    Example usage:
    esm2_model.lm_head = MultiheadedNoSharedRobertaLMHead(
        embed_dim=ESM2_EMBEDDINGS_SIZE, hidden_layer_sizes=[256], 
        output_dim=self.n_classes, dropout_rate=0.2)  
    """
    def __init__(self,
                    embed_dim,
                    hidden_layer_sizes=[256],
                    output_dim=20, dropout_rate=0.2, layer_norm : bool = True):
        
        super().__init__()

        n_classes = output_dim

        # fine heads
        self.fine_heads_one_per_class = []
        for cl in range(n_classes):
            self.head = RobertaLMHead(embed_dim=embed_dim, hidden_dim=hidden_layer_sizes[0], output_dim=1, dropout_rate=dropout_rate, layer_norm=layer_norm)
            self.fine_heads_one_per_class.append(self.head)
        self.fine_heads_one_per_class = nn.ModuleList(self.fine_heads_one_per_class)

    def forward(self, features):
        self.fine_heads_one_per_class = self.fine_heads_one_per_class.to(features.device)
        out = torch.cat([self.fine_heads_one_per_class[head](features) for head in range(len(self.fine_heads_one_per_class))], dim=1)

        return out
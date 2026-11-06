"""
RobertaLMHeadDeeper robertalmheaddeeper.py
Code taken from https://github.com/microsoft/seq2symm/tree/main
"""

import torch
from torch import nn
from src.models.layers.activation.gelu import gelu

class RobertaLMHeadDeeper(nn.Module):
    """Head for masked language modeling.
     Example usage:
     esm2_model.lm_head = RobertaLMHeadDeeper(embed_dim=ESM2_EMBEDDINGS_SIZE, hidden_dim=[256,50],
          output_dim=self.n_classes, dropout_rate=0.2)        
    #"""        

    def __init__(self, embed_dim, hidden_dim, output_dim, dropout_rate=0.1, layer_norm : bool = False):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, hidden_dim[0])
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer added
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim[0]) # Changed to LayerNorm
        else:
            self.layer_norm1 = nn.Identity()
        self.dense2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        if layer_norm:
            self.layer_norm2 = nn.LayerNorm(hidden_dim[1]) # Changed to LayerNorm
        else:
            self.layer_norm2 = nn.Identity()
        self.dense3 = nn.Linear(hidden_dim[1], output_dim)        

    def forward(self, features, mask):   ## ESM2 features size: torch.Size([1, N, 1280])
        x = self.dense1(features)
        x = gelu(x)
        x = self.dropout(x)  # Dropout applied after activation
        x = self.layer_norm1(x)

        # average over residues in protein.. 
        if mask is not None:
            mask = mask.unsqueeze(-1).to(x.dtype)  # [B, residues, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(1)
            
        x = self.dense2(x)
        x = gelu(x)
        x = self.dropout(x)  # Dropout applied after activation
        x = self.layer_norm2(x)
        x = self.dense3(x)
        return x
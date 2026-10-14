

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer added
        self.layer_norm = ESM1bLayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):   ## ESM2 features size: torch.Size([1, N, 1280])
        x = self.dense1(features)
        x = gelu(x)
        x = self.dropout(x)  # Dropout applied after activation
        x = self.layer_norm(x)
        # average over residues in protein.. 
        x = x.mean(1)
        x = self.dense2(x)
        return x

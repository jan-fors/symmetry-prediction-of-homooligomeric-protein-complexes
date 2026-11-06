import torch
from torch import nn

class ESM2Autoencoder(nn.Module):
    def __init__(self):
        super(ESM2Autoencoder, self).__init__()

        self.encoder = nn.Sequential(

        )

        self.decoder = nn.Sequential(

        )

    def forward(self, x):
        encoded = self.encoder(x)

        # TODO save latent variables of run?
        
        decoded = self.decoder(encoded)
        return decoded
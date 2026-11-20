"""
ridge_regression.py

Also called l2 regularization
"""
import torch.nn as nn

class RidgeRegression(nn.Module):
    def __init__(self):
        super(RidgeRegression, self).__init__()

    def forward(self, model):
        """
        """
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        return l2_norm
"""
lasso_regression.py

Also called l1 regularization
"""
import torch.nn as nn

class LassoRegression(nn.Module):
    def __init__(self):
        super(LassoRegression, self).__init__()

    def forward(self, model):
        """
        """
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        return l1_norm
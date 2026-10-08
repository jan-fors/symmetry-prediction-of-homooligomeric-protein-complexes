import torch
from torch import nn
# from src.utils.constants import (
#     PREDICTION_THRESHOLD
# )

def validate(X, y, model : nn.Module, loss_fn : nn.Module):
    """
    """
    # forward
    logits = model(X)

    #predictions = (logits >= PREDICTION_THRESHOLD).int()

    # calcualte loss
    loss = loss_fn(logits, y, model)
    
    return loss, logits
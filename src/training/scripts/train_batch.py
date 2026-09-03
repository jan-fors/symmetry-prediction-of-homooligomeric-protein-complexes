import torch
from torch import nn
import torch.optim as optim
from src.utils.constants import (
    PREDICTION_THRESHOLD
)

def train_batch(X, y, model : nn.Module, loss_fn : nn.Module, optim : optim.Optimizer):
    """
    """
    # forward
    logits = model(X)

    predictions = (logits >= PREDICTION_THRESHOLD).int()
      
    # calcualte loss
    loss = loss_fn(logits, y)

    # backprop
    loss.backward()
    optim.step()
    optim.zero_grad()

    return loss, predictions
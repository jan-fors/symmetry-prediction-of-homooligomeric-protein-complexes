import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.training.scripts.validate import validate
import numpy as np
from src.evaluation.metrics.accuracy import accuracy
from src.evaluation.metrics.f1_score import f1_score
from src.evaluation.metrics.auc_pr import auc_pr
import logging

logger = logging.getLogger(__name__)


def train_epoch(model, loader, loss_fn, optimizer, device):
    """
    """
    model.train()

    losses = []

    for batch, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        logits = model(X)

        loss = loss_fn(logits, y, model)
        losses.append(loss.item())

        loss.backward()

        optimizer.step()

    return float(np.mean(losses))

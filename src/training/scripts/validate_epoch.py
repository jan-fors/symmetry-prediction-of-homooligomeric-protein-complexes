import torch
from torch import nn
import numpy as np

def validate_epoch(model, loss_fn, loader, device):
    """
    """
    model.eval()

    all_logits = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for batch, (X, y) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            loss = loss_fn(logits, y, model)

            losses.append(loss.item())
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)

    return float(np.mean(losses)), logits, labels
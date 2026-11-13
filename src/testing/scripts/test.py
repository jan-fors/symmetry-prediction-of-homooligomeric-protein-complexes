import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from src.training.scripts.validate_epoch import validate_epoch
from src.evaluation.metrics.compute_metrics import compute_metrics
import logging

logger = logging.getLogger(__name__)


def test(device, test_dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    """ """
    _, test_logits, test_labels = validate_epoch(model, loss_fn, test_dataloader, device)
    return test_logits, test_labels, compute_metrics(test_labels, test_logits)

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.training.scripts.train_batch import train_batch
from src.training.scripts.validate import validate
import numpy as np
from src.evaluation.metrics.accuracy import accuracy
from src.evaluation.metrics.f1_score import f1_score
from src.evaluation.metrics.auc_pr import auc_pr
import logging

logger = logging.getLogger(__name__)


def train_epoch(
    device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    epoch,
):
    """ """
    # TRAIN
    training_size = len(train_dataloader.dataset)

    # metrics
    metrics = {
        "train": {"mean-loss": None},
        "val": {
            "mean-loss": None,
            "accuracy": None,
            "macro-f1": None,
            "weighted-f1": None,
        #    "auc-pr": None,
        },
    }

    # set to train
    model.train()

    sum_tloss = 0.0

    for batch, (X, y) in enumerate(train_dataloader):
        # train epoch
        X = X.to(device)
        y = y.to(device)

        running_tloss, running_tpredictions = train_batch(
            X, y, model, loss_fn, optimizer
        )

        if batch % 100 == 0:
            logger.info(
                "Processed %d/%d items, Training Loss: %f",
                batch * train_dataloader.batch_size,
                training_size,
                running_tloss,
            )

        sum_tloss += running_tloss

    metrics["train"]["mean-loss"] = float(sum_tloss / (batch + 1))

    # set to evaluation mode
    model.eval()

    # EVAL
    sum_vloss = 0.0
    predictions = []
    truth = []

    # Disable gradient computation
    with torch.no_grad():
        for batch, (X, y) in enumerate(val_dataloader):
            X = X.to(device)
            y = y.to(device)

            running_vloss, running_vpredictions = validate(X, y, model, loss_fn)

            running_vpredictions = running_vpredictions.cpu()
            y = y.cpu()

            sum_vloss += running_vloss
            predictions.extend(running_vpredictions.unbind(0))
            truth.extend(y.unbind(0))

    metrics["val"]["mean-loss"] = float(sum_vloss / (batch + 1))
    metrics["val"]["accuracy"] = accuracy(truth, predictions)
    metrics["val"]["macro-f1"] = f1_score(truth, predictions, "macro")
    metrics["val"]["weighted-f1"] = f1_score(truth, predictions, "weighted")
    #metrics["val"]["auc-pr"] = auc_pr(truth, predictions)

    logger.info(
        "\nValidation Loss\t%f, \nValidation Accuracy\t%f, \nValidation Macro-F1\t%f, \nValidation Weighted-F1\t%f",
        metrics["val"]["mean-loss"],
        metrics["val"]["accuracy"],
        metrics["val"]["macro-f1"],
        metrics["val"]["weighted-f1"],
       # metrics["val"]["auc-pr"],
    )

    return metrics

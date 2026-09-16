from rich import print
from tqdm import tqdm
from src.io.writers.save_model import save_model
from pathlib import Path
from torch.utils.data import DataLoader
from src.training.scripts.train_epoch import train_epoch
import logging

logger = logging.getLogger(__name__)


def train(
    device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model,
    loss_fn,
    optimizer,
    num_epochs,
    output_dir: Path,
) -> Path:
    """
    Return path to best model
    """
    metrics = {
        "train": {"mean-loss": []},
        "val": {
            "mean-loss": [],
            "accuracy": [],
            "macro-f1": [],
            "weighted-f1": [],
     #       "auc-pr": [],
        },
    }

    best_model_path = None
    best_vloss = 1000000
    for epoch in range(num_epochs):
        logger.info(f"Train epoch {epoch}")
        # train & validate model
        epoch_metrics = train_epoch(
            device, train_dataloader, val_dataloader, model, loss_fn, optimizer, epoch
        )

        if epoch_metrics["val"]["mean-loss"] < best_vloss:
            # save best model
            best_vloss = epoch_metrics["val"]["mean-loss"]
            best_model_path = save_model(model, output_dir)

        metrics["train"]["mean-loss"].append(epoch_metrics["train"]["mean-loss"])
        metrics["val"]["mean-loss"].append(epoch_metrics["val"]["mean-loss"])
        metrics["val"]["accuracy"].append(epoch_metrics["val"]["accuracy"])
        metrics["val"]["macro-f1"].append(epoch_metrics["val"]["macro-f1"])
        metrics["val"]["weighted-f1"].append(epoch_metrics["val"]["weighted-f1"])
    #    metrics["val"]["auc-pr"].append(epoch_metrics["val"]["auc-pr"])

    return best_model_path, metrics

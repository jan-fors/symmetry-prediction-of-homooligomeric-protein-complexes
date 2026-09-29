import numpy as np
from src.io.writers.save_model import save_model
from pathlib import Path
from torch.utils.data import DataLoader
from src.training.scripts.train_epoch import train_epoch
from src.training.scripts.validate_epoch import validate_epoch
from src.evaluation.metrics.compute_metrics import compute_metrics
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
    model.to(device)

    best_score, best_epoch, best_state = -np.inf, -1, None

    history = []

    for epoch in range(num_epochs):
        # train epoch
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)

        # validate epoch
        val_loss, val_logits, val_labels = validate_epoch(model, loss_fn, val_dataloader, device)

        metrics = compute_metrics(val_labels, val_logits) # threshold 0.5

        record = {"epoch": epoch, "train_loss": train_loss,
                  "val_loss": val_loss, **metrics}
        
        history.append(record)

        logger.info(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| AP {metrics['ap_macro']:.4f} | F1 {metrics['f1_macro']:.4f}")

        score = metrics["f1_macro"]

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = model.state_dict()
            best_model_path = save_model(model, output_dir)
        else:
            #TODO
            pass

    model.load_state_dict(best_state)
    logger.info(f"best epoch {best_epoch}, f1-macro = {best_score:.4f}")

    return model, history

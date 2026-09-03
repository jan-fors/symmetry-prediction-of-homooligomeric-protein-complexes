from rich import print
from tqdm import tqdm
from src.io.writers.save_model import save_model
from pathlib import Path
from torch.utils.data import DataLoader
from src.training.scripts.train_epoch import train_epoch

def train(train_dataloader : DataLoader, val_dataloader : DataLoader, model, loss_fn, optimizer, num_epochs, output_dir : Path) -> Path:
    """
    Return path to best model
    """
    best_model_path = None
    best_vloss = 1000000
    for epoch in range(num_epochs):
        # train & validate model
        avg_vloss = train_epoch(train_dataloader, val_dataloader, model, loss_fn, optimizer, epoch)

        if avg_vloss < best_vloss:
            # save best model
            best_vloss = avg_vloss
            best_model_path = save_model(model, output_dir)

    return best_model_path
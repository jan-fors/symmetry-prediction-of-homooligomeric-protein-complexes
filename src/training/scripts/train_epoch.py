import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.training.scripts.train_batch import train_batch
from src.training.scripts.validate import validate
import numpy as np
from src.training.metrics.accuracy import accuracy
from src.training.metrics.f1_score import f1_score
from tqdm import tqdm

def train_epoch(
    device, 
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    epoch
):
    """
    """
    print("-"*80)
    # TRAIN
    training_size = len(train_dataloader.dataset)

    # set to train
    model.train()

    for batch, (X, y) in tqdm(enumerate(train_dataloader), desc=f"Train Epoch {epoch}", total=training_size/train_dataloader.batch_size):
        # train epoch
        X = X.to(device)
        y = y.to(device)

        batch_loss, batch_predictions = train_batch(X, y , model, loss_fn, optimizer)

    # set to evaluation mode
    model.eval()

    # EVAL
    sum_vloss = 0.0
    predictions = np.array([])
    truth = np.array([])

    # Disable gradient computation
    with torch.no_grad():
        for batch, (X, y) in enumerate(val_dataloader):
            X = X.to(device)
            y = y.to(device)

            running_vloss, running_vpredictions = validate(X, y, model, loss_fn)
            
            running_vpredictions = running_vpredictions.cpu()
            y = y.cpu()

            sum_vloss += running_vloss
            predictions = np.append(predictions, running_vpredictions)
            truth = np.append(truth, y)

    avg_loss = sum_vloss / (batch + 1)

    print("Val. Loss:", float(avg_loss))
    print("Accuracy:", accuracy(truth, predictions))
    print("F1-Score:", f1_score(truth, predictions))

    return avg_loss
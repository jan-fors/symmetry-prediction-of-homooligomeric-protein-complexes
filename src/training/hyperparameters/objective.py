import torch
from torch.utils.data import DataLoader
from src.training.scripts.train_epoch import train_epoch
from src.training.scripts.validate_epoch import validate_epoch
from src.evaluation.metrics.compute_metrics import compute_metrics

from src.training.hyperparameters.parameters.num_epochs import get_num_epochs
from src.training.hyperparameters.parameters.loss import get_loss
from src.training.hyperparameters.parameters.model import get_model
from src.training.hyperparameters.parameters.optimizer import get_optimizer


def objective(
    trial,
    config: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device,
):
    """
    basically one run of the model with defined terms

    - create config/ select hyperparameters
    - run epochs
        - train
        - validate
    - calcualte score
    - handle pruning etc
    """
    # num_epochs
    num_epochs = get_num_epochs(trial, config["training"])

    # build loss_fn
    loss_fn = get_loss(trial, config["loss"])

    # build model
    model = get_model(trial, config["model"])

#    print(model)

    model.to(device)
   
    # build optimizer
    optimizer = get_optimizer(trial, model, config["optimizer"])

    for epoch in range(num_epochs):
        # train epoch
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)

        # validate epoch
        val_loss, val_logits, val_labels = validate_epoch(
            model, loss_fn, val_dataloader, device
        )

    metrics = compute_metrics(val_labels, val_logits)

    return metrics[config["metric"]]



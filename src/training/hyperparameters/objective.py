import torch
from torch.utils.data import DataLoader
from src.training.scripts.train_epoch import train_epoch
from src.training.scripts.validate_epoch import validate_epoch
from src.evaluation.metrics.compute_metrics import compute_metrics
from src.models.losses.registry import build_loss_fn
from src.models.architectures.registry import build_model
from src.models.optimizers.registry import build_optimizer


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
    if type(config["training"]["epochs"]) == int:
        num_epochs = config["training"]["epochs"]
    else:
        num_epochs = _get_correct_suggest(trial, config["training"]["epochs"]["type"])(
            "num_epochs",
            _get_type(config["training"]["epochs"]["type"])(config["training"]["epochs"]["from"]),
            _get_type(config["training"]["epochs"]["type"])(config["training"]["epochs"]["to"]),
            log= config["training"]["epochs"]["log"]
        )

    # build loss_fn
    loss_fns = []

    for l_fn in config["loss"]:
        if type(l_fn["factor"]) == float:
            loss_fns.append({"name": l_fn["name"], "factor": l_fn["factor"]})
        else:
            loss_fns.append(
                {
                    "name": l_fn["name"],
                    "factor": _get_correct_suggest(trial, l_fn["factor"]["type"])(
                        l_fn["name"], _get_type(l_fn["factor"]["type"])(l_fn["factor"]["from"]), _get_type(l_fn["factor"]["type"])(l_fn["factor"]["to"]), log=l_fn["factor"]["log"]
                    ),
                }
            )

    loss_fn = build_loss_fn(loss_fns)

    # build model
    model = build_model(config["model"])  # TODO if needed

    model.to(device)

    # build optimizer
    optimizer_params = {}

    if type(config["optimizer"]["name"]) == str:
        optimizer_params["name"] = config["optimizer"]["name"]
    else:
        optimizer_params["name"] = _get_correct_suggest(
            trial, config["optimizer"]["name"]["type"]
        )("optimizer_type", config["optimizer"]["name"]["options"])

    if type(config["optimizer"]["lr"]) == float:
        optimizer_params["lr"] = config["optimizer"]["lr"]
    else:
        optimizer_params["lr"] = _get_correct_suggest(
            trial, config["optimizer"]["lr"]["type"]
        )(
            "learning_rate",
            _get_type(config["optimizer"]["lr"]["type"])(config["optimizer"]["lr"]["from"]),
            _get_type(config["optimizer"]["lr"]["type"])(config["optimizer"]["lr"]["to"]),
            log=config["optimizer"]["lr"]["log"]
        )

    if "momentum" in config["optimizer"].keys():
        if type(config["optimizer"]["momentum"]) == float:
            optimizer_params["momentum"] = config["optimizer"]["momentum"]
        else:
            optimizer_params["momentum"] = _get_correct_suggest(
                trial, config["optimizer"]["momentum"]["type"]
            )(
                "learning_rate",
                _get_type(config["optimizer"]["momentum"]["type"])(config["optimizer"]["momentum"]["from"]),
                _get_type(config["optimizer"]["momentum"]["type"])(config["optimizer"]["momentum"]["to"]),
                log=config["optimizer"]["momentum"]["log"]
            )

    optimizer = build_optimizer(optimizer_params, model)

    for epoch in range(num_epochs):
        # train epoch
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)

        # validate epoch
        val_loss, val_logits, val_labels = validate_epoch(
            model, loss_fn, val_dataloader, device
        )

    metrics = compute_metrics(val_labels, val_logits)

    return metrics[config["metric"]]


def _get_correct_suggest(trial, type_str: str):
    """ """
    if type_str == "int":
        return trial.suggest_int

    if type_str == "float":
        return trial.suggest_float

    if type_str == "categorical":
        return trial.suggest_categorical

def _get_type(type_str : str):
    """
    """
    if type_str == "int":
        return int

    if type_str == "float":
        return float

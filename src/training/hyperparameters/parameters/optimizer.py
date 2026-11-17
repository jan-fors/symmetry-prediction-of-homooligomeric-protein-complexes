"""
"""
# imports
from src.training.hyperparameters.utils.suggest import get_correct_suggest
from src.training.hyperparameters.utils.typestr import get_type
from src.models.optimizers.registry import build_optimizer

# main fn
def get_optimizer(trial, model, optimizer_config : dict):
    """
    """
    trial_optimizer_config = {}

    if type(optimizer_config["name"]) == str:
        trial_optimizer_config["name"] = optimizer_config["name"]
    else:
        trial_optimizer_config["name"] = get_correct_suggest(
            trial, optimizer_config["name"]["type"]
        )("optimizer_type", optimizer_config["name"]["options"])

    if type(optimizer_config["lr"]) == float:
        trial_optimizer_config["lr"] = optimizer_config["lr"]
    else:
        trial_optimizer_config["lr"] = get_correct_suggest(
            trial, optimizer_config["lr"]["type"]
        )(
            "learning_rate",
            get_type(optimizer_config["lr"]["type"])(optimizer_config["lr"]["from"]),
            get_type(optimizer_config["lr"]["type"])(optimizer_config["lr"]["to"]),
            log=optimizer_config["lr"]["log"]
        )

    if "momentum" in optimizer_config.keys():
        if type(optimizer_config["momentum"]) == float:
            trial_optimizer_config["momentum"] = optimizer_config["momentum"]
        else:
            trial_optimizer_config["momentum"] = get_correct_suggest(
                trial, optimizer_config["momentum"]["type"]
            )(
                "learning_rate",
                get_type(optimizer_config["momentum"]["type"])(optimizer_config["momentum"]["from"]),
                get_type(optimizer_config["momentum"]["type"])(optimizer_config["momentum"]["to"]),
                log=optimizer_config["momentum"]["log"]
            )

    optimizer = build_optimizer(trial_optimizer_config, model)
    return optimizer
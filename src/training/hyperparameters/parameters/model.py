""" """

# imports
from src.training.hyperparameters.utils.suggest import get_correct_suggest
from src.training.hyperparameters.utils.typestr import get_type
from src.models.architectures.registry import build_model


# main fn
def get_model(trial, model_config: dict):
    """
    Builds a model and might allow to optimize some parameters
    """
    trial_model_config = {"name": model_config["name"], "params": {}}

    for key in model_config["params"].keys():
        if type(model_config["params"][key]) == dict:  # use variable params
            parameter = model_config["params"][key]
            trial_model_config["params"][key] = get_correct_suggest(
                trial, parameter["type"]
            )(
                key,
                get_type(parameter["type"])(parameter["from"]),
                get_type(parameter["type"])(parameter["to"]),
                log=parameter["log"],
            )
        else:
            trial_model_config["params"][key] = model_config["params"][key]

    return build_model(trial_model_config)

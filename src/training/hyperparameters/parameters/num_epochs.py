"""
"""
# imports
from src.training.hyperparameters.utils.suggest import get_correct_suggest
from src.training.hyperparameters.utils.typestr import get_type

# main fn
def get_num_epochs(trial, training_config : dict) -> int:
    """
    """
    # check if epochs should get optimized
    if type(training_config["epochs"]) == int:
        return training_config["epochs"]
    else:
        return get_correct_suggest(trial, training_config["epochs"]["type"])(
            "num_epochs",
            get_type(training_config["epochs"]["type"])(training_config["epochs"]["from"]),
            get_type(training_config["epochs"]["type"])(training_config["epochs"]["to"]),
            log=training_config["epochs"]["log"]
        )
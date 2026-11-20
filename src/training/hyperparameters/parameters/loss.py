"""
"""
# imports
from src.training.hyperparameters.utils.suggest import get_correct_suggest
from src.training.hyperparameters.utils.typestr import get_type
from src.models.losses.registry import build_loss_fn

# main fn
def get_loss(trial, loss_config : dict):
    """
    """
    loss_fns = []
    
    for l_fn in loss_config:
        if type(l_fn["factor"]) == float:
            loss_fns.append({"name": l_fn["name"], "factor": l_fn["factor"]})
        else:
            loss_fns.append(
                {
                    "name": l_fn["name"],
                    "factor": get_correct_suggest(trial, l_fn["factor"]["type"])(
                        l_fn["name"], get_type(l_fn["factor"]["type"])(l_fn["factor"]["from"]), 
                        get_type(l_fn["factor"]["type"])(l_fn["factor"]["to"]), 
                        log=l_fn["factor"]["log"]
                    ),
                }
            )

    loss_fn = build_loss_fn(loss_fns)

    return loss_fn
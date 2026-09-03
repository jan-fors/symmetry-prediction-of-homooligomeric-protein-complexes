"""
Optimizer registry.py


"""
import torch.optim as optim
from torch import nn

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam
}

def build_optimizer(optim_config : dict, model : nn.Module) -> optim.Optimizer:
    """
    """
    # load opmizier params
    optim_name = optim_config["name"]
    optim_class = OPTIMIZER_REGISTRY[optim_name]

    learning_rate = optim_config["lr"]

    return optim_class(model.parameters(), lr=learning_rate)
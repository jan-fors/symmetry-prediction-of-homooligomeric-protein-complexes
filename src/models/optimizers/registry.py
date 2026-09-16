"""
Optimizer registry.py


"""
import torch.optim as optim
from torch import nn
import logging 
logger = logging.getLogger(__name__)

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam
}

def build_optimizer(optim_config : dict, model : nn.Module) -> optim.Optimizer:
    """
    """
    logger.info("Building optimizer")

    # load opmizier params
    logger.info(f"Using {optim_config['name']} with lr={optim_config['lr']}")

    optim_name = optim_config["name"]
    optim_class = OPTIMIZER_REGISTRY[optim_name]

    learning_rate = optim_config["lr"]

    return optim_class(model.parameters(), lr=learning_rate)
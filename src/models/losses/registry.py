"""
Loss registry.py

Should allow for selection of multiple losses etc.

"""
from typing import List
import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)

LOSS_REGISTRY = {
    "bce": nn.BCEWithLogitsLoss
}

def build_loss_fn(loss_fns : List[dict]):
    """
    """
    logger.info("Defining loss function")
    
    if len(loss_fns) == 1: # Return the selected loss function
        return LOSS_REGISTRY[loss_fns[0]["name"]]()
    else: # Create loss combination
        return


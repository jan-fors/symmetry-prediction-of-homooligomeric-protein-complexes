"""
Loss registry.py

Should allow for selection of multiple losses etc.

"""
from typing import List
import torch
from torch import nn
from src.models.losses.margin_loss import MultiLabelRankingLossWithIndicatorTarget
from src.models.losses.combined_loss import CombinedLoss
import logging
logger = logging.getLogger(__name__)

LOSS_REGISTRY = {
    "bce": nn.BCEWithLogitsLoss,
    "margin_loss": MultiLabelRankingLossWithIndicatorTarget
}

def build_loss_fn(loss_fns : List[dict]):
    """
    """
    logger.info("Defining loss function")
    
    if len(loss_fns) == 1: # Return the selected loss function
        return LOSS_REGISTRY[loss_fns[0]["name"]]()
    else: # Create loss combination
        losses = []

        for i in range(len(loss_fns)):
            losses.append((loss_fns[i]["factor"], LOSS_REGISTRY[loss_fns[i]["name"]]))

        return CombinedLoss(losses)


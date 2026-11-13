"""
Loss registry.py

Should allow for selection of multiple losses etc.

"""
from typing import List
import torch
from torch import nn
from src.models.losses.margin_loss import MultiLabelRankingLossWithIndicatorTarget
from src.models.losses.combined_loss import CombinedLoss
from src.models.losses.lasso_regression import LassoRegression
from src.models.losses.ridge_regression import RidgeRegression
import logging
logger = logging.getLogger(__name__)

LOSS_REGISTRY = {
    "bce": nn.BCEWithLogitsLoss,
    "margin_loss": MultiLabelRankingLossWithIndicatorTarget
}

REGULARIZATION_REGISTRY = {
    "l1": LassoRegression,
    "l2": RidgeRegression 
}

def build_loss_fn(loss_fns : List[dict]):
    """
    """
    logger.info("Defining loss function")
    
    losses = []
    regularizations = []

    for i in range(len(loss_fns)):
        name = loss_fns[i]["name"]
        factor = loss_fns[i]["factor"]
        if name in LOSS_REGISTRY.keys():
            losses.append((factor, LOSS_REGISTRY[name]))
        elif name in REGULARIZATION_REGISTRY.keys():
            regularizations.append((factor, REGULARIZATION_REGISTRY[name]))
        else:
            logger.warning(f"{name} not in ")


    return CombinedLoss(losses, regularizations)


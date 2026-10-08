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
    
    if len(loss_fns) == 1: # Return the selected loss function
        if loss_fns[0]["name"] in LOSS_REGISTRY.keys():
            loss_fn = LOSS_REGISTRY[loss_fns[0]["name"]]
            return CombinedLoss([(1.0, loss_fn)], None)
        else:
            logger.error("Must provide basic loss function, not just regularization.")
    else: # Create loss combination
        losses = []
        regularizations = []

        for i in range(len(loss_fns)):
            if loss_fns[i]["name"] in LOSS_REGISTRY.keys():
                losses.append((loss_fns[i]["factor"], LOSS_REGISTRY[loss_fns[i]["name"]]))
            elif loss_fns[i]["name"] in REGULARIZATION_REGISTRY.keys():
                regularizations.append((loss_fns[i]["factor"], REGULARIZATION_REGISTRY[loss_fns[i]["name"]]))
            else:
                logger.warning(f"{loss_fns[i]['name']} not in ")


        return CombinedLoss(losses, regularizations)


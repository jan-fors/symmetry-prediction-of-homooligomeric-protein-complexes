"""
Model registry.py

Select a model and specify parameters in the config?

"""
from src.models.architectures.simple_mlp_classifier import SimpleMLPClassifier
from src.models.architectures.mlp_classifier import MLPClassifier

import logging
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "simple_mlp_classifier": SimpleMLPClassifier,
    "mlp_classifier": MLPClassifier,
    "robertalmhead": None
}

def build_model(model_config : dict):
    """
    """
    logger.info("Building model")

    logger.info(f"Using model: {model_config['name']}")
    model_name = model_config["name"]

    model_class = MODEL_REGISTRY[model_name]

    return model_class(**model_config["params"])


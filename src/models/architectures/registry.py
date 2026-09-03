"""
Model registry.py

Select a model and specify parameters in the config?

"""
from src.models.architectures.simple_mlp_classifier import SimpleMLPClassifier

MODEL_REGISTRY = {
    "simple_mlp_classifier": SimpleMLPClassifier,
}

def build_model(model_config : dict):
    """
    """
    model_name = model_config["name"]

    model_class = MODEL_REGISTRY[model_name]

    return model_class(**model_config["params"])


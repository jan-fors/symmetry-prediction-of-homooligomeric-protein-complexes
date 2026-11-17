from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler

SAMPLER_REGISTRY = {
    "random_sampler": RandomSampler,
    "tpe_sampler": TPESampler
}

def build_sampler(config : dict):
    """
    """
    return SAMPLER_REGISTRY[config["sampler"]]()
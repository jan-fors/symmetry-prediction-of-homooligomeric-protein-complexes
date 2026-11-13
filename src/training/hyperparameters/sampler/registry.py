from optuna.samplers import RandomSampler

SAMPLER_REGISTRY = {
    "random_sampler": RandomSampler
}

def build_sampler(config : dict):
    """
    """
    return SAMPLER_REGISTRY[config["sampler"]]()
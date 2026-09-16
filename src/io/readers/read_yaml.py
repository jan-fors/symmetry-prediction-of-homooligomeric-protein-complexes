import yaml
from pathlib import Path
import os
from src.io.writers.verbose_print import v_print



def read_yaml_to_dict(yaml_path : Path) -> dict:
    """
    """
    if not os.path.exists(yaml_path):
        raise ValueError(f"Path to yaml: {yaml_path} does not exist.")

    with open(yaml_path, "r") as f:
        loaded_data = yaml.safe_load(f)
    
    return loaded_data
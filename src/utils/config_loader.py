from pathlib import Path
from src.io.readers.read_yaml import read_yaml_to_dict

def load_config(config_path : Path) -> dict:
    """
    """
    return read_yaml_to_dict(config_path)

def config_is_valid(config : dict) -> bool:
    """
    """
    return
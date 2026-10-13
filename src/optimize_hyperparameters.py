import logging
import argparse
import torch
import optuna
from pathlib import Path
from src.utils.config_loader import load_config

def optimize_hyperparameters(config_path : Path):
    """
    """
    # load experiment config
    config = load_config(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    optimize_hyperparameters(args.config)
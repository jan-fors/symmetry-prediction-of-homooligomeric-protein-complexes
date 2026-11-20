from pathlib import Path
from datetime import datetime
import shutil
import os

def create_experiment_output_dir(config_path : Path, config : dict) -> Path:
    """
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config["experiment_name"]

    experiment_output_folder = Path(config["output_dir"]) / Path(timestamp + "_" + experiment_name)
    os.makedirs(experiment_output_folder, exist_ok=True)

    # copy config
    shutil.copy2(config_path, experiment_output_folder / Path("config.yml"))

    return experiment_output_folder
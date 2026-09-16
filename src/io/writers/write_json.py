from pathlib import Path
import json

def write_json(data : dict, file_path : Path):
    """
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
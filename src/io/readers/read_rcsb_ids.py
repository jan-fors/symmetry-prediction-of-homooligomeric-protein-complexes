from pathlib import Path
from typing import List
import os

def read_rcsb_ids(file_path : Path) -> List[str]:
    """
    Takes a comma separated file as input. Returns a list containing all the rcsb ids from that file.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"The file {file_path} does not exist.")

    with open(file_path, "r") as f:
        res = f.readline()
        res = res.split(",")
        res = [x.strip() for x in res]

    return res

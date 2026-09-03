from pathlib import Path

def read_cluster_ids_from_file(file : Path):
    """
    """
    with open(file, "r") as f:
        cluster_ids = f.readlines()

    return [x.strip() for x in cluster_ids]
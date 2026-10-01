from pathlib import Path
from typing import List
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def get_unique_labels(config : dict) -> List[str]:
    """
    """
    metadata_path = config["metadata_file"]

    if "label_groups" in config.keys():
        label_groups = config["label_groups"]
    else:
        label_groups = None

    df = pd.read_csv(metadata_path)
    symmetries = df["SYMM"].to_list()
    symmetries = [x.strip() for x in symmetries]

    unique = set()

    for i in symmetries:
        items = i.split(" ")

        for it in items:
            unique.add(it)

    unique_labels = list(unique)
    to_remove = []
  
    if label_groups != None:
        for label in unique_labels:
            for key in label_groups.keys():
                if label in label_groups[key]:
                    to_remove.append(label)

        for t in to_remove:
            unique_labels.remove(t)

        for key in label_groups.keys():
            unique_labels.append(key)

    unique_labels.sort()
        
    logger.info(f"Identified {len(unique_labels)} unique labels.")

    return unique_labels
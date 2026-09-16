from pathlib import Path
from typing import List
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def get_unique_labels(metadata_path : Path) -> List[str]:
    """
    """
    df = pd.read_csv(metadata_path)
    symmetries = df["SYMM"].to_list()

    unique = set()

    for i in symmetries:
        items = i.split(" ")

        for it in items:
            unique.add(it)

    unique_labels = list(unique)

    logger.info(f"Identified {len(unique_labels)} unique labels.")

    return unique_labels
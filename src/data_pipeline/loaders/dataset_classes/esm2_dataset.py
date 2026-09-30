"""
TODO just placeholder

Custom dataset class for training, validation and testing.

Allows to select which types of input are to selected.
Class needs access to label file as well as the raw data dir.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import pandas as pd
from src.utils.constants import LABEL_COLUMN
import logging

logger = logging.getLogger(__name__)


class ESM2EmbeddingDataset(Dataset):
    def __init__(
        self,
        esm2_dir: Path,
        metadata_file: Path,
        cluster_ids: List[int],
        transform=None,
        label_encoder=None,
    ):
        """ """
        # read metadata_file
        metadata = pd.read_csv(metadata_file)

        # create subset with relevant ones
        metadata_subset = metadata[metadata["CLUSTER"].isin(cluster_ids)]

        self.transform = transform
        self.label_encoder = label_encoder

        # create self.data
        self.data = []
        self.labels = []
        dropped = 0

        for index, row in metadata_subset.iterrows():
            chain_id = row["CHAINID"]
            rcsb_code = chain_id.split("_")[0]

            embedding_file_path = esm2_dir / Path(rcsb_code) / Path(chain_id + ".pt")

            try:
                embedding = torch.load(embedding_file_path)
                self.data.append(embedding)
                symm = row[LABEL_COLUMN]
                self.labels.append(symm)
            except Exception as e:
                logger.warning(e)
                dropped += 1

        logger.info(f"Dropped {dropped} items because of wrong path or format")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            for t in self.transform:
                sample = t(sample)

        if self.label_encoder:
            labels = label.split(" ")
            label = self.label_encoder(labels)

        return sample, label

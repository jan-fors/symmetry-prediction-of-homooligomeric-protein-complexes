import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.data_pipeline.split.split import read_split_ids
from src.data_pipeline.loaders.dataset_classes.esm2_dataset import ESM2EmbeddingDataset
from src.data_pipeline.preprocessing.transformers.registry import create_transformations


DATASET_CLASS_REGISTRY = {
    "esm2": ESM2EmbeddingDataset
}

def build_datasets(dataset_config_data : dict, label_encoder = None) -> tuple[Dataset, Dataset, Dataset]:
    """
    Takes the experiment config and generates all datasets (train, val and test)

    Retrurn:
        - train_dataset: Dataset
        - val_dataset: Dataset
        - test_dataset: Dataset
    """
    # read split ids
    split_dir = dataset_config_data["split"]
    train_ids, val_ids, test_ids = read_split_ids(split_directory=split_dir)

    # read metadatafilepath
    metadata_file_path = dataset_config_data["metadata_file"]

    # build datasets
    features = dataset_config_data["features"]
    

    if len(features) == 1: # return the dataset class
        feature_class = features[0]["class"]
        dataset_class = DATASET_CLASS_REGISTRY[feature_class]
        data_dir = features[0]["data_dir"]

        # create transformations
        transforms = create_transformations(features[0]["augmentations"])

        train = dataset_class(data_dir, metadata_file_path, train_ids, label_encoder=label_encoder, transform=transforms)
        val = dataset_class(data_dir, metadata_file_path, val_ids, label_encoder=label_encoder, transform=transforms)
        test = dataset_class(data_dir, metadata_file_path, test_ids, label_encoder=label_encoder, transform=transforms)

        return train, val, test
    else:  # return concat dataset
        return None, None, None
    
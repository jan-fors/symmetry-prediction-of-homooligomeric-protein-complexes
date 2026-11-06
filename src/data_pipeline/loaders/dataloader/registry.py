import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import logging
logger = logging.getLogger(__name__)

def collate_proteins(samples):
    features, labels = zip(*samples)
    labels = torch.stack([torch.as_tensor(y) for y in labels])


    lengths = [f.size(0) for f in features]

    if len(set(lengths)) == 1:
        return torch.stack(features), labels, None

    lengths = torch.tensor(lengths)
    padded = pad_sequence(features, batch_first=True)
    positions = torch.arange(padded.size(1))
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

    return padded, labels, mask


COLLATE_REGISTRY = {
    "collate_proteins": collate_proteins
}

def create_dataloaders(dataloader_config_data : dict, train : Dataset, val : Dataset, test : Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    """
    logger.info("Creating dataloaders")
    # read config data
    logger.info(f"Batch size: {dataloader_config_data['batch_size']}, shuffle: {dataloader_config_data['shuffle']}")
    batch_size = dataloader_config_data["batch_size"]
    shuffle = dataloader_config_data["shuffle"]

    # create dataloader
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, collate_fn=COLLATE_REGISTRY[dataloader_config_data["collate_fn"]])
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=shuffle, collate_fn=COLLATE_REGISTRY[dataloader_config_data["collate_fn"]])
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, collate_fn=COLLATE_REGISTRY[dataloader_config_data["collate_fn"]])

    return train_dataloader, val_dataloader, test_dataloader



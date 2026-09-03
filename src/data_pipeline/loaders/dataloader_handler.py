from torch.utils.data import Dataset, DataLoader

def create_dataloaders(dataloader_config_data : dict, train : Dataset, val : Dataset, test : Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    """
    # read config data
    batch_size = dataloader_config_data["batch_size"]
    shuffle = dataloader_config_data["shuffle"]

    # create dataloader
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader, test_dataloader
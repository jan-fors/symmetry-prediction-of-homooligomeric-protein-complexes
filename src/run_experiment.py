import argparse
import os
import torch
from pathlib import Path
from src.utils.config_loader import load_config
from src.data_pipeline.loaders.dataset_classes.registry import build_datasets
from src.data_pipeline.loaders.dataloader_handler import create_dataloaders
from src.models.losses.registry import build_loss_fn
from src.models.optimizers.registry import build_optimizer
from src.models.architectures.registry import build_model
from src.training.scripts.train import train
from src.testing.scripts.test import test
from src.utils.unique_labels import get_unique_labels
from src.data_pipeline.preprocessing.encoders.label import MultiLabelEncoder
from datetime import datetime

def run_experiment(config_path : Path):
    """
    """
    # load experiment config
    test_config_path = Path(config_path)
    config = load_config(test_config_path)

    # check if config is valid
    # TODO

    # create output_dir
    timestamp = datetime.now().strftime("%Y%m%m_%H%M%S")
    experiment_name = test_config_path.stem

    experiment_output_folder = Path(config["output_dir"]) / Path(timestamp + "_" + experiment_name)
    os.makedirs(experiment_output_folder, exist_ok=True)

    # define label encoder
    unique_labels = get_unique_labels(config["dataset"]["metadata_file"])
    label_encoder = MultiLabelEncoder(unique_labels)

    # build data
    train_dataset, val_dataset, test_dataset = build_datasets(config["dataset"], label_encoder)

    # create dataloader
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataloader_config_data=config["dataloader"],
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
    )

    # build loss
    loss_fn = build_loss_fn(config["loss"])

    # build model
    model = build_model(config["model"])

    # build optimizer
    optimizer = build_optimizer(config["optimizer"], model)

    # epoch loop
    epochs = config["training"]["epochs"]
    best_model_path = train(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, experiment_output_folder)

    # test model
    model = build_model(config["model"])
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    tloss = test(test_dataloader, model, loss_fn)
    print(f"Test Loss {tloss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_experiment(args.config)
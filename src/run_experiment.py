import logging
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
from src.io.writers.plot import plot_metrics, save_classwise_scores_plot
from src.io.writers.write_table import save_metrics_to_csv
from src.evaluation.metrics.per_class import per_class_metrics
from src.io.writers.write_json import write_json

def run_experiment(config_path : Path):
    """
    """
    # load experiment config
    test_config_path = Path(config_path)
    config = load_config(test_config_path)

    # check if config is valid
    # TODO

    # create output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = test_config_path.stem

    experiment_output_folder = Path(config["output_dir"]) / Path(timestamp + "_" + experiment_name)
    os.makedirs(experiment_output_folder, exist_ok=True)

    # define logging
    logging.basicConfig(
        filename=experiment_output_folder/Path("run.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        encoding="utf-8"
    )
    logger = logging.getLogger(__name__) 
    logger.info("run experiment started")
    

    if torch.cuda.is_available():
        device = torch.device("cuda") # Use the first available CUDA device
        logger.info(f"CUDA (GPU) is available. Using device: {device}")
        # You can also specify a specific GPU, e.g., torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        logger.info(f"CUDA (GPU) not available. Using device: {device}")

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

    # move model to device
    logger.info(f"Moving model to {device}.")
    model.to(device)

    # build optimizer
    optimizer = build_optimizer(config["optimizer"], model)

    # epoch loop
    logger.info(f"Starting with Training. Plan {config['training']['epochs']} epochs.")
    epochs = config["training"]["epochs"]
    best_model_path, train_metrics = train(device, train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, experiment_output_folder)

    # save and plot train metrics
    save_metrics_to_csv(train_metrics, experiment_output_folder/Path("train_metrics.csv"))
    plot_metrics(train_metrics, experiment_output_folder/Path("train_metrics.png"))

    # test model
    model = build_model(config["model"])
    model.to(device)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_predictions, test_truth, test_metrics = test(device, test_dataloader, model, loss_fn)

    # calculate per class metrics
    test_metrics["per-class"] = per_class_metrics(label_encoder, test_predictions, test_truth)

    # save and plot per-class metrics
    write_json(test_metrics, experiment_output_folder/Path("test_metrics.json"))
    save_classwise_scores_plot(test_metrics, ["auc-pr"] ,experiment_output_folder/Path("per_class_metrics.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_experiment(args.config)
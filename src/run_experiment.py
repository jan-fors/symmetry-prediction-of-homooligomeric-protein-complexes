import logging
import argparse
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
from src.io.writers.output_dir import create_experiment_output_dir
from src.data_pipeline.preprocessing.encoders.label import MultiLabelEncoder
from src.io.writers.plot import plot_metrics, plot_class_metrics
from src.io.writers.write_table import save_metrics_to_csv
from src.evaluation.metrics.compute_metrics import per_class_metrics
from src.io.logging.logging import init_logging


def run_experiment(config_path : Path):
    """
    """
    # load experiment config
    config = load_config(config_path)

    # create output_dir
    exp_out_dir = create_experiment_output_dir(config_path, config)

    # define logging
    init_logging(exp_out_dir)

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
    label_encoder = MultiLabelEncoder(config["dataset"])

    # build data
    train_dataset, val_dataset, test_dataset = build_datasets(config["dataset"], label_encoder)

    # create dataloader
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataloader_config_data=config["dataloader"],
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
    )

    # build model
    model = build_model(config["model"])
    logger.info(model)

    # move model to device
    logger.info(f"Moving model to {device}.")
    model.to(device)

    # build loss
    loss_fn = build_loss_fn(config["loss"])

    # build optimizer
    optimizer = build_optimizer(config["optimizer"], model)

    # epoch loop
    logger.info(f"Starting with Training. Plan {config['training']['epochs']} epochs.")

    epochs = config["training"]["epochs"]
    model, history = train(device, train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, exp_out_dir)

    # test model
    test_logits, test_labels, test_metrics = test(device, test_dataloader, model, loss_fn)

    # save and plot train metrics
    save_metrics_to_csv(history, exp_out_dir/Path("train_metrics.csv"))
    plot_metrics(history, exp_out_dir/Path("train_metrics.png"))

    # calculate per class metrics
    report = per_class_metrics(labels=test_labels, logits=test_logits, class_names=label_encoder.get_labels())
    # print(report.head())
    report.to_csv(exp_out_dir/Path("per_class_test_metrics.csv"))
    plot_class_metrics(str(exp_out_dir/Path("per_class_test_metrics.csv")), save_path=exp_out_dir/Path("per_class_test_metrics.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_experiment(args.config)
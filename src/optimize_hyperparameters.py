import logging
import argparse
import torch
import optuna
from pathlib import Path
from src.utils.config_loader import load_config
from optuna.trial import TrialState
from src.training.hyperparameters.objective import objective
from functools import partial
from src.data_pipeline.loaders.dataset_classes.registry import build_datasets
from src.data_pipeline.loaders.dataloader.dataloader_handler import create_dataloaders
from src.data_pipeline.preprocessing.encoders.label import MultiLabelEncoder
from src.io.writers.output_dir import create_experiment_output_dir
from src.io.logging.logging import init_logging
from src.training.hyperparameters.sampler.registry import build_sampler


def optimize_hyperparameters(config_path: Path):
    """ """
    # load experiment config
    config = load_config(config_path)

    # read optuna params
    n_trials = config["n_trials"]
    direction = config["direction"]

    # # create output_dir
    exp_out_dir = create_experiment_output_dir(config_path, config)

    # define logging
    init_logging(exp_out_dir)

    logger = logging.getLogger(__name__)
    logger.info("run experiment started")

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the first available CUDA device
        logger.info(f"CUDA (GPU) is available. Using device: {device}")
        # You can also specify a specific GPU, e.g., torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        logger.info(f"CUDA (GPU) not available. Using device: {device}")

    # define label encoder
    label_encoder = MultiLabelEncoder(config["dataset"])

    # build data
    train_dataset, val_dataset, test_dataset = build_datasets(
        config["dataset"], label_encoder
    )

    # create dataloader
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataloader_config_data=config["dataloader"],
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
    )

    study = optuna.create_study(direction=direction, sampler=build_sampler(config))

    # TODO add sampler
    study.optimize(
        partial(
            objective,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
        ),
        n_trials=n_trials,
        timeout=600,
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info("  Number of finished trials: ", str(len(study.trials)))
    logger.info("  Number of pruned trials: ", str(len(pruned_trials)))
    logger.info("  Number of complete trials: ", str(len(complete_trials)))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: ", str(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # TODO create


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    optimize_hyperparameters(args.config)

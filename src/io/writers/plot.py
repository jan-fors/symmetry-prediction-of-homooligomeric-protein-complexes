import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path

def plot_metrics(metrics, save_path=None):
    """
    Plot all metric values against epoch number.

    Parameters
    ----------
    metrics : dict
        Nested dictionary containing metric lists, for example:
        {
            "train": {"mean-loss": [...]},
            "val": {
                "mean-loss": [...],
                "accuracy": [...],
                "macro-f1": [...],
                "weighted-f1": [...],
                "auc-pr": [...]
            }
        }

    save_path : str, optional
        Path where the plot should be saved.
    """

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create enough distinct colors for all train/validation metrics
    metric_names = [
        f"{split} - {metric_name}"
        for split, split_metrics in metrics.items()
        for metric_name in split_metrics
    ]

    colors = sns.color_palette("tab10", n_colors=len(metric_names))

    color_index = 0

    for split, split_metrics in metrics.items():
        for metric_name, values in split_metrics.items():
            if not values:
                continue

            epochs = range(1, len(values) + 1)

            ax.plot(
                epochs,
                values,
                label=f"{split} {metric_name}",
                color=colors[color_index],
                marker="o",
                linewidth=2,
                markersize=4,
            )

            color_index += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training and Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def natural_sort_key(value):
    """
    Sort strings naturally, e.g. C1, C2, C10 instead of C1, C10, C2.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]

def save_classwise_scores_plot(
    scores,
    metrics,
    output_path="classwise_scores.png",
    figsize=(12, 6),
    dpi=300
):
    """
    Save a grouped bar plot of selected class-wise evaluation scores.

    Parameters
    ----------
    scores : dict
        Dictionary containing a "per-class" key.

    metrics : str or list[str]
        Metric or metrics to plot, for example:
        "accuracy"
        or
        ["accuracy", "macro-f1"]

    output_path : str or pathlib.Path
        Location where the plot will be saved.

    figsize : tuple
        Figure size.

    dpi : int
        Resolution of the saved plot.
    """

    # Allow a single metric to be passed as a string
    if isinstance(metrics, str):
        metrics = [metrics]

    # Sort classes naturally: C1, C2, ..., C10
    classes = sorted(
        scores["per-class"].keys(),
        key=natural_sort_key
    )

    # Validate requested metrics
    available_metrics = set(scores["per-class"][classes[0]].keys())
    missing_metrics = set(metrics) - available_metrics

    if missing_metrics:
        raise ValueError(
            f"Unknown metric(s): {sorted(missing_metrics)}. "
            f"Available metrics are: {sorted(available_metrics)}"
        )

    x = np.arange(len(classes))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=figsize)

    for i, metric in enumerate(metrics):
        values = [
            scores["per-class"][class_name][metric]
            for class_name in classes
        ]

        offset = (i - (len(metrics) - 1) / 2) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=metric
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Class-wise Evaluation Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    # Create output directory if necessary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
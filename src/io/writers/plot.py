import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
import pandas as pd

def plot_metrics(history, save_path=None):
    """
    Plot per-epoch metrics against epoch number.

    Parameters
    ----------
    history : list of dict
        One dict per epoch, as returned by `fit`, e.g.
        {"epoch": 1, "train_loss": ..., "val_loss": ..., "val_ap_macro": ...}
    save_path : str, optional
        Path where the plot should be saved.
    """
    if not history:
        return None

    # Pivot: {"train_loss": [...], "val_ap_macro": [...], ...}
    series = {}
    for record in history:
        for key, value in record.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            series.setdefault(key, []).append(value)

    epochs = [record.get("epoch", i) for i, record in enumerate(history, start=1)]

    # Losses on their own axis, everything else (bounded 0-1) on the second
    loss_keys = [k for k in series if "loss" in k.lower()]
    score_keys = [k for k in series if k not in loss_keys]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("tab10", n_colors=len(series))
    colors = dict(zip(series, palette))

    for ax, keys, title in (
        (axes[0], loss_keys, "Loss"),
        (axes[1], score_keys, "Validation metrics"),
    ):
        for key in keys:
            values = series[key]
            ax.plot(
                epochs[: len(values)],
                values,
                label=key.replace("_", " "),
                color=colors[key],
                marker="o",
                linewidth=2,
                markersize=4,
            )
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if keys:
            ax.legend()

    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
  

def plot_class_metrics(
    metrics,
    figsize=(18, 11),
    min_support=1,
    sort_by="f1",
    save_path=None,
):
    """
    Plot per-class metrics using bar charts.

    Panels:
      1. Grouped horizontal bars for precision, recall, F1, AP, and AUROC
      2. Horizontal bars for true support
      3. Grouped bars for TP, FP, and FN

    Parameters
    ----------
    metrics : pd.DataFrame or str
        Metrics DataFrame or path to a CSV file.
    figsize : tuple
        Figure size.
    min_support : int
        Only include classes with at least this many true samples.
    sort_by : str
        Column used to sort classes, e.g. "f1", "support", or "auroc".
    save_path : str or None
        Optional path for saving the plot.

    Returns
    -------
    fig, axes
    """

    # Load metrics
    if isinstance(metrics, str):
        df = pd.read_csv(metrics)
    else:
        df = metrics.copy()

    # Remove summary rows
    df = df[~df["class"].isin(["MACRO", "MICRO"])].copy()

    # Convert columns to numeric
    numeric_columns = [
        "support",
        "prevalence",
        "predicted",
        "precision",
        "recall",
        "f1",
        "ap",
        "auroc",
        "tp",
        "fp",
        "fn",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Exclude classes with no true examples
    df = df[df["support"] >= min_support].copy()

    if df.empty:
        raise ValueError("No classes remain after applying min_support.")

    if sort_by not in df.columns:
        raise ValueError(f"Unknown sort_by column: {sort_by}")

    metric_columns = ["precision", "recall", "f1", "ap", "auroc"]

    # AP/AUROC may be NaN for classes with insufficient examples
    df[metric_columns] = df[metric_columns].fillna(0)

    # Sort classes
    df = df.sort_values(sort_by, ascending=True).reset_index(drop=True)

    sns.set_theme(style="whitegrid", context="talk")

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[2.5, 1.2],
        width_ratios=[3.4, 1.3],
    )

    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_support = fig.add_subplot(grid[0, 1])
    ax_confusion = fig.add_subplot(grid[1, :])

    classes = df["class"].astype(str)
    y = np.arange(len(df))

    # ------------------------------------------------------------------
    # Panel 1: grouped horizontal bar chart for performance metrics
    # ------------------------------------------------------------------

    metric_colors = {
        "precision": "#4C78A8",
        "recall": "#F58518",
        "f1": "#54A24B",
        "ap": "#B279A2",
        "auroc": "#E45756",
    }

    n_metrics = len(metric_columns)
    bar_height = 0.15

    for index, metric in enumerate(metric_columns):
        offset = (index - (n_metrics - 1) / 2) * bar_height

        bars = ax_metrics.barh(
            y + offset,
            df[metric],
            height=bar_height,
            color=metric_colors[metric],
            label=metric.upper(),
            edgecolor="none",
        )

        # Add metric values to the end of the bars
        for bar, value in zip(bars, df[metric]):
            if value > 0:
                ax_metrics.text(
                    value + 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                )

    ax_metrics.set_yticks(y)
    ax_metrics.set_yticklabels(classes)
    ax_metrics.set_xlim(0, 1.10)
    ax_metrics.set_xlabel("Score")
    ax_metrics.set_ylabel("Class")
    ax_metrics.set_title("Per-class performance", weight="bold")

    ax_metrics.axvline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax_metrics.legend(
        title="Metric",
        loc="lower right",
        ncol=3,
        fontsize=10,
    )

    # Add alternating background bands
    for index in range(len(df)):
        if index % 2 == 0:
            ax_metrics.axhspan(
                index - 0.5,
                index + 0.5,
                color="black",
                alpha=0.025,
                zorder=0,
            )

    # ------------------------------------------------------------------
    # Panel 2: support
    # ------------------------------------------------------------------

    support_bars = ax_support.barh(
        y,
        df["support"],
        color="#72B7B2",
        edgecolor="none",
    )

    ax_support.set_yticks(y)
    ax_support.set_yticklabels([])
    ax_support.set_xlabel("True support")
    ax_support.set_title("Class support", weight="bold")

    # Use a logarithmic scale for highly imbalanced support
    if df["support"].max() / max(df["support"].min(), 1) >= 20:
        ax_support.set_xscale("log")

    for bar, value in zip(support_bars, df["support"]):
        ax_support.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {int(value)}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # ------------------------------------------------------------------
    # Panel 3: TP / FP / FN
    # ------------------------------------------------------------------

    confusion_data = df.set_index("class")[["tp", "fp", "fn"]].fillna(0)

    confusion_data.plot(
        kind="bar",
        ax=ax_confusion,
        width=0.78,
        color=["#54A24B", "#E45756", "#F2CF5B"],
        edgecolor="none",
    )

    ax_confusion.set_title("Prediction counts", weight="bold")
    ax_confusion.set_xlabel("")
    ax_confusion.set_ylabel("Count")
    ax_confusion.tick_params(axis="x", rotation=45)

    ax_confusion.legend(
        ["TP", "FP", "FN"],
        title="",
        ncol=3,
        loc="upper left",
    )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    fig.suptitle(
        "Per-class Classification Metrics",
        fontsize=24,
        fontweight="bold",
    )

    for ax in [ax_metrics, ax_support, ax_confusion]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, {
        "performance": ax_metrics,
        "support": ax_support,
        "confusion": ax_confusion,
    }
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
    metrics_to_show=("precision", "recall", "f1", "ap", "auroc"),
    figsize=None,
    show_values=False,
    save_path=None,
):
    """
    Plot per-class scores as a grouped bar chart.
 
    x-axis: all classes in natural order (C1, C2, ..., C10, D1, D2, ...)
    bars:   one bar per metric for each class
    Missing values (e.g. AUROC for classes without positives) leave a gap.
 
    Parameters
    ----------
    metrics : pd.DataFrame or str
        Metrics DataFrame or path to a CSV file. Must contain a "class" column.
    metrics_to_show : sequence of str
        Metric columns to display, in this order.
    figsize : tuple or None
        Figure size. Chosen automatically from the number of classes if None.
    show_values : bool
        Write the score above each bar.
    save_path : str or None
        Optional path for saving the plot.
 
    Returns
    -------
    fig, ax
    """
    # Load
    df = pd.read_csv(metrics) if isinstance(metrics, str) else metrics.copy()
    df["class"] = df["class"].astype(str)
 
    # Drop summary rows (these are not classes)
    df = df[~df["class"].str.upper().isin(["MACRO", "MICRO"])]
 
    cols = [m for m in metrics_to_show if m in df.columns]
    if not cols:
        raise ValueError("None of the requested metrics are in the data.")
    for c in cols + ["support"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
 
    # Natural sort: C1, C2, ..., C10, D1, D2, ...
    def natural_key(name):
        return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]
 
    df = df.set_index("class").loc[sorted(df["class"], key=natural_key)]
 
    n_classes, n_metrics = len(df), len(cols)
    x = np.arange(n_classes)
    group_width = 0.82
    bar_width = group_width / n_metrics
 
    if figsize is None:
        figsize = (max(10, n_classes * (0.22 * n_metrics + 0.35)), 6)
 
    colors = {
        "precision": "#4C78A8",
        "recall": "#F58518",
        "f1": "#54A24B",
        "ap": "#B279A2",
        "auroc": "#E45756",
    }
    fallback = plt.get_cmap("tab10").colors
 
    fig, ax = plt.subplots(figsize=figsize)
 
    for k, metric in enumerate(cols):
        offset = (k - (n_metrics - 1) / 2) * bar_width
        values = df[metric].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            np.nan_to_num(values, nan=0.0),
            width=bar_width,
            color=colors.get(metric, fallback[k % 10]),
            label=metric.upper(),
            edgecolor="white",
            linewidth=0.5,
        )
        if show_values:
            for bar, v in zip(bars, values):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + 0.01,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
 
    # Class labels, with support underneath if available
    if "support" in df.columns:
        labels = [
            f"{c}\nn={int(s)}" if not np.isnan(s) else c
            for c, s in zip(df.index, df["support"])
        ]
    else:
        labels = list(df.index)
 
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, n_classes - 0.4)
    ax.set_ylim(0, 1.12 if show_values else 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("Score")
    ax.set_xlabel("Class")
 
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
 
    ax.legend(
        ncol=n_metrics,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=10,
    )
    ax.set_title("Per-class scores", fontsize=14, fontweight="bold", pad=34)
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
 
    return fig, ax

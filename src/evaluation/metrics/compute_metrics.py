import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import pandas as pd

def compute_metrics(labels, logits, threshold : float = 0.5) -> dict:
    """
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    present = labels.sum(axis=0) > 0

    return {
        "ap_macro": average_precision_score(
            labels[:, present], logits[:, present], average="macro"),
        "ap_micro": average_precision_score(
            labels[:, present], logits[:, present], average="micro"),
        "auroc_macro": roc_auc_score(
            labels[:, present], logits[:, present], average="macro"),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "exact_match": (preds == labels).all(axis=1).mean(),
    }

def per_class_metrics(labels, logits, thresholds=0.5, class_names=None):
    """
    Per-label metrics for a multilabel model.

    Parameters
    ----------
    labels : array (N, L) of 0/1
    logits : array (N, L), raw model outputs
    thresholds : float or array (L,)
        Scalar, or the per-label thresholds from `tune_thresholds`.
    class_names : list of str, optional

    Returns
    -------
    pandas.DataFrame, one row per label plus macro/micro summary rows.
    """
    labels = np.asarray(labels).astype(int)
    logits = np.asarray(logits)
    n_labels = labels.shape[1]

    thresholds = np.full(n_labels, thresholds, dtype=float) \
        if np.isscalar(thresholds) else np.asarray(thresholds, dtype=float)

    if class_names is None:
        class_names = [f"label_{i}" for i in range(n_labels)]

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= thresholds).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0, labels=range(n_labels)
    )

    # Ranking metrics are undefined for a label with no positives (or no
    # negatives, for AUROC) -- compute per column and leave those as NaN.
    ap = np.full(n_labels, np.nan)
    auroc = np.full(n_labels, np.nan)
    for j in range(n_labels):
        pos = labels[:, j].sum()
        if 0 < pos < len(labels):
            ap[j] = average_precision_score(labels[:, j], logits[:, j])
            auroc[j] = roc_auc_score(labels[:, j], logits[:, j])

    tp = ((preds == 1) & (labels == 1)).sum(axis=0)
    fp = ((preds == 1) & (labels == 0)).sum(axis=0)
    fn = ((preds == 0) & (labels == 1)).sum(axis=0)

    df = pd.DataFrame({
        "support": support,
        "prevalence": labels.mean(axis=0),
        "predicted": preds.sum(axis=0),
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
        "auroc": auroc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }, index=pd.Index(class_names, name="class"))

    # Summary rows. Macro = unweighted mean over labels; micro = pooled counts.
    macro = df[["precision", "recall", "f1", "ap", "auroc"]].mean()
    micro_p = tp.sum() / max(tp.sum() + fp.sum(), 1)
    micro_r = tp.sum() / max(tp.sum() + fn.sum(), 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)

    summary = pd.DataFrame(
        [
            {**macro.to_dict(), "support": support.sum()},
            {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
             "support": support.sum()},
        ],
        index=pd.Index(["MACRO", "MICRO"], name="class"),
    )

    return pd.concat([df, summary])
import torch
import numpy as np
from src.evaluation.metrics.accuracy import accuracy
from src.evaluation.metrics.f1_score import f1_score
from src.evaluation.metrics.auc_pr import auc_pr

def per_class_metrics(label_encoder, predicted_labels : list, true_labels : list):
    """
    """
    per_class_metrics = {}

    n_labels = label_encoder.get_n_labels()

    for label_nr in range(n_labels):
        empty_tensor = torch.zeros(n_labels)
        one_hot_tensor = empty_tensor
        one_hot_tensor[label_nr] = 1

        label = label_encoder.decode(one_hot_tensor)[0]

        pred = np.array([x[label_nr] for x in predicted_labels])
        truth = np.array([x[label_nr] for x in true_labels])

        per_class_metrics[label] = {
            "accuracy": accuracy(truth, pred),
            "weighted-f1": f1_score(truth, pred, "weighted"),
            "macro-f1": f1_score(truth, pred, "macro"),
            "auc-pr": auc_pr(truth, pred)

        }

    return per_class_metrics
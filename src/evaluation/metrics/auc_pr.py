from sklearn.metrics import precision_recall_curve, auc
import numpy as np

def auc_pr(y_true : np.array, y_test : np.array):
    """
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_test)
    auc_pr = auc(recall, precision)

    return auc_pr
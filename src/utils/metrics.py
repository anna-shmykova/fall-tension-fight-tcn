# src/utils/metrics.py

import numpy as np
from typing import List, Dict
from sklearn.metrics import precision_recall_curve, auc


# ---------------------------------------------------------
# AUPRC (Area Under Precision-Recall Curve)
# ---------------------------------------------------------
def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute AUPRC for binary classification.

    Parameters
    ----------
    y_true : (N,) array of {0,1}
    y_prob : (N,) array of probabilities (after sigmoid)

    Returns
    -------
    float : area under precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


# ---------------------------------------------------------
# Threshold sweep for binary classification
# ---------------------------------------------------------
def threshold_sweep_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.01,
    end: float = 0.99,
    step: float = 0.01,
) -> List[Dict[str, float]]:
    """
    Sweep thresholds and compute precision/recall/F1.

    Returns a list of dicts:
        [
            {
              "threshold": ...,
              "precision": ...,
              "recall": ...,
              "f1": ...,
              "tp": ...,
              "fp": ...,
              "fn": ...
            },
            ...
        ]
    """
    results = []

    thresholds = np.arange(start, end + 1e-9, step)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results.append({
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

    return results
"""
Evaluation metrics for multi-task fault diagnostics.
IT22577924 — Karunanayake K.P.A.W.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        task_name: str = "") -> dict:
    """
    Compute full metrics suite for one classification task.

    Parameters
    ----------
    y_true     : ground-truth labels [N]
    y_pred     : predicted labels [N]
    task_name  : string label for reporting

    Returns
    -------
    dict with: accuracy, precision_macro, recall_macro, f1_macro, mcc,
               confusion_matrix, n_samples, n_classes
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    rec   = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    f1    = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    mcc   = matthews_corrcoef(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "task":             task_name,
        "accuracy":         float(acc),
        "precision_macro":  float(prec),
        "recall_macro":     float(rec),
        "f1_macro":         float(f1),
        "mcc":              float(mcc),
        "confusion_matrix": cm.tolist(),
        "n_samples":        int(len(y_true)),
        "n_classes":        int(len(labels)),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: list = None) -> dict:
    """Per-class F1 scores."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    if class_names is None:
        class_names = [str(l) for l in labels]
    return {class_names[i]: float(scores[i]) for i in range(min(len(class_names), len(scores)))}


FAULT_TYPE_NAMES  = ["normal", "LG", "LL", "LLG", "LLL", "HIF"]
FAULT_PHASE_NAMES = ["none", "A", "B", "C", "AB/ABG", "BC/BCG", "CA/CAG", "ABC"]
DETECT_NAMES      = ["normal", "fault"]

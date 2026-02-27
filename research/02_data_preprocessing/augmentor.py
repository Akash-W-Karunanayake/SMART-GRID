"""
Augmentor — SMOTE oversampling for HIF class imbalance.
Operates in flattened feature space; reconstructs sequence shape after synthesis.
IT22577924 — Karunanayake K.P.A.W.
"""
import logging
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def smote_oversample(features: np.ndarray, labels: np.ndarray,
                     target_labels: list = None,
                     random_state: int = 42) -> tuple:
    """
    Apply SMOTE to balance one or more minority classes against the majority.

    Parameters
    ----------
    features      : np.ndarray [N, D]  — flattened feature vectors
    labels        : np.ndarray [N]     — class labels (label_type)
    target_labels : list[int]          — minority classes to oversample
                                         (default [5] = HIF; pass [4, 5] for LLL+HIF)
    random_state  : int

    Returns
    -------
    (features_resampled [N', D], labels_resampled [N'])
    """
    if target_labels is None:
        target_labels = [5]

    class_counts = dict(zip(*np.unique(labels, return_counts=True)))
    logger.info(f"Class distribution before SMOTE: {class_counts}")

    # Determine target count: match the majority class
    majority_count = max(class_counts.values())
    sampling_strategy = {
        lbl: majority_count
        for lbl in target_labels
        if lbl in class_counts and class_counts[lbl] < majority_count
    }

    if not sampling_strategy:
        logger.info("No minority classes need SMOTE; returning original data")
        return features, labels

    # k_neighbors must be < the smallest minority class count
    min_target_count = min(class_counts[lbl] for lbl in sampling_strategy)
    k_neighbors = min(5, min_target_count - 1)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    try:
        X_res, y_res = smote.fit_resample(features, labels)
        new_counts = dict(zip(*np.unique(y_res, return_counts=True)))
        logger.info(f"Class distribution after SMOTE: {new_counts}")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"SMOTE failed ({e}); returning original data unchanged")
        return features, labels


def flatten_sequences(voltage_seqs: list, current_seqs: list) -> np.ndarray:
    """
    Flatten list of sequence arrays into 2D feature matrix.

    Parameters
    ----------
    voltage_seqs : list of [T, N_buses, 6]
    current_seqs : list of [T, N_branches, 3]

    Returns
    -------
    np.ndarray [N_samples, T*(N_buses*6 + N_branches*3)]
    """
    flat = []
    for V, I in zip(voltage_seqs, current_seqs):
        flat.append(np.concatenate([V.flatten(), I.flatten()]))
    return np.stack(flat, axis=0)


def unflatten_sequences(features: np.ndarray,
                        T: int, N_buses: int, N_branches: int) -> tuple:
    """
    Reconstruct voltage_seqs and current_seqs from flattened feature matrix.

    Returns
    -------
    (voltage_seqs: list[np.ndarray], current_seqs: list[np.ndarray])
    """
    v_size = T * N_buses * 6
    voltage_seqs = []
    current_seqs = []
    for row in features:
        V = row[:v_size].reshape(T, N_buses, 6)
        I = row[v_size:].reshape(T, N_branches, 3)
        voltage_seqs.append(V)
        current_seqs.append(I)
    return voltage_seqs, current_seqs


def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute inverse-frequency weights for weighted cross-entropy loss.

    Returns
    -------
    dict {class_label: weight}
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {}
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = total / (len(classes) * cnt)
    logger.info(f"Class weights: {weights}")
    return weights

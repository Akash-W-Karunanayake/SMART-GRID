"""
Physics-Aware Augmentation for fault signal data.
Replaces SMOTE (which creates implausible signals via linear interpolation
of flattened time series) with physically motivated transforms.

Augmentation methods:
  1. Gaussian noise injection (SNR 20-40 dB)
  2. Random magnitude scaling (±5% on pre-fault steady-state)
  3. Random phase rotation (multiply phasors by e^{jθ}, θ ~ U(-5°, 5°))
  4. Random channel dropout (zero 1-2 measurement channels, 10% probability)

IT22577924 — Karunanayake K.P.A.W.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def augment_sample(voltage_seq: np.ndarray, current_seq: np.ndarray,
                   rng: np.random.Generator = None) -> tuple:
    """
    Apply a random subset of physics-aware augmentations to a single sample.

    Parameters
    ----------
    voltage_seq : np.ndarray [T, N_buses, 6]    — [mag, ang] × 3 phases
    current_seq : np.ndarray [T, N_branches, 6] — [mag, ang] × 3 phases

    Returns
    -------
    (voltage_aug, current_aug) — augmented copies
    """
    if rng is None:
        rng = np.random.default_rng()

    v = voltage_seq.copy()
    c = current_seq.copy()

    # Apply each augmentation with independent probability
    if rng.random() < 0.7:
        v, c = _add_gaussian_noise(v, c, rng)
    if rng.random() < 0.5:
        v, c = _magnitude_scaling(v, c, rng)
    if rng.random() < 0.4:
        v, c = _phase_rotation(v, c, rng)
    if rng.random() < 0.1:
        v, c = _channel_dropout(v, c, rng)

    return v, c


def augment_dataset(voltage_seqs: list, current_seqs: list,
                    labels_type: np.ndarray,
                    target_counts: dict = None,
                    random_state: int = 42) -> tuple:
    """
    Augment minority classes to reach target counts using physics-aware transforms.

    Parameters
    ----------
    voltage_seqs  : list of [T, N_buses, 6]
    current_seqs  : list of [T, N_branches, 6]
    labels_type   : np.ndarray [N] — class labels
    target_counts : dict {class_label: target_count}
                    If None, balances all classes to majority count.
    random_state  : int

    Returns
    -------
    (aug_voltage_seqs, aug_current_seqs, aug_labels)
    """
    rng = np.random.default_rng(random_state)

    class_counts = dict(zip(*np.unique(labels_type, return_counts=True)))
    logger.info(f"Class distribution before augmentation: {class_counts}")

    if target_counts is None:
        majority_count = max(class_counts.values())
        target_counts = {cls: majority_count for cls in class_counts}

    # Group indices by class
    class_indices = {}
    for cls in class_counts:
        class_indices[cls] = np.where(labels_type == cls)[0]

    aug_v, aug_c, aug_labels = list(voltage_seqs), list(current_seqs), list(labels_type)
    aug_src_indices = []  # track which original index each augmented sample came from

    for cls, target in target_counts.items():
        if cls not in class_counts:
            continue
        n_existing = class_counts[cls]
        n_needed = target - n_existing
        if n_needed <= 0:
            continue

        indices = class_indices[cls]
        for _ in range(n_needed):
            # Pick a random existing sample from this class
            src_idx = rng.choice(indices)
            v_aug, c_aug = augment_sample(
                voltage_seqs[src_idx], current_seqs[src_idx], rng)
            aug_v.append(v_aug)
            aug_c.append(c_aug)
            aug_labels.append(cls)
            aug_src_indices.append(src_idx)

    aug_labels = np.array(aug_labels, dtype=labels_type.dtype)
    aug_src_indices = np.array(aug_src_indices, dtype=np.int64)
    new_counts = dict(zip(*np.unique(aug_labels, return_counts=True)))
    logger.info(f"Class distribution after augmentation: {new_counts}")

    return aug_v, aug_c, aug_labels, aug_src_indices


def _add_gaussian_noise(v: np.ndarray, c: np.ndarray,
                        rng: np.random.Generator) -> tuple:
    """Add Gaussian noise at SNR between 20-40 dB."""
    snr_db = rng.uniform(20.0, 40.0)
    snr_linear = 10 ** (snr_db / 10.0)

    for arr in [v, c]:
        # Only add noise to magnitude columns (0, 2, 4)
        for col in [0, 2, 4]:
            signal = arr[:, :, col]
            signal_power = np.mean(signal ** 2) + 1e-10
            noise_power = signal_power / snr_linear
            noise = rng.normal(0, np.sqrt(noise_power), signal.shape)
            arr[:, :, col] = signal + noise.astype(arr.dtype)

    return v, c


def _magnitude_scaling(v: np.ndarray, c: np.ndarray,
                       rng: np.random.Generator) -> tuple:
    """Random magnitude scaling ±5% on all magnitude columns."""
    scale = rng.uniform(0.95, 1.05)
    for arr in [v, c]:
        for col in [0, 2, 4]:
            arr[:, :, col] *= scale
    return v, c


def _phase_rotation(v: np.ndarray, c: np.ndarray,
                    rng: np.random.Generator) -> tuple:
    """
    Random phase rotation: add uniform offset θ ~ U(-5°, 5°) to all angle
    columns. This is equivalent to multiplying phasors by e^{jθ}.
    """
    theta_deg = rng.uniform(-5.0, 5.0)
    for arr in [v, c]:
        for col in [1, 3, 5]:  # angle columns
            arr[:, :, col] += theta_deg
    return v, c


def _channel_dropout(v: np.ndarray, c: np.ndarray,
                     rng: np.random.Generator) -> tuple:
    """Zero out 1-2 random measurement channels (both mag and angle)."""
    n_drop = rng.choice([1, 2])
    # Choose which array and which phase to drop
    for _ in range(n_drop):
        target = rng.choice([0, 1])  # 0=voltage, 1=current
        phase = rng.integers(0, 3)    # which phase (0, 1, 2)
        entity = rng.integers(0, v.shape[1] if target == 0 else c.shape[1])
        arr = v if target == 0 else c
        arr[:, entity, phase * 2] = 0.0      # mag
        arr[:, entity, phase * 2 + 1] = 0.0  # ang
    return v, c


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

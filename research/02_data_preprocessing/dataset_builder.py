"""
Dataset Builder — Build train/val/test splits from validated NPZ files.
Produces:
  - Plain NumPy bundles for CNN-Transformer (temporal branch)
  - PyG Data objects for R-GNN (spatial branch)

Updated: current_seq shape is now [T, N_branches, 6] (mag+ang per phase).

IT22577924 — Karunanayake K.P.A.W.
"""
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# PyG import is optional — fall back gracefully if not installed
try:
    import torch
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


def load_all_samples(valid_paths: list) -> dict:
    """
    Load all validated NPZ files into in-memory lists.

    Returns dict with lists: voltage_seqs, current_seqs, labels_*,
    edge_indices, edge_attrs, metadata
    """
    v_seqs, i_seqs = [], []
    l_det, l_type, l_phase, l_loc = [], [], [], []
    edge_indices, edge_attrs = [], []
    meta = []

    for p in valid_paths:
        d = np.load(p, allow_pickle=True)
        v_seqs.append(d["voltage_seq"].astype(np.float32))
        i_seqs.append(d["current_seq"].astype(np.float32))
        l_det.append(int(d["label_detection"]))
        l_type.append(int(d["label_type"]))
        l_phase.append(int(d["label_phase"]))
        l_loc.append(int(d["label_location"]))
        edge_indices.append(d["edge_index"])
        edge_attrs.append(d["edge_attr"].astype(np.float32))
        meta.append({
            "fault_type":  str(d["fault_type"]),
            "bus":         str(d["bus"]),
            "resistance":  float(d["resistance"]),
        })

    return {
        "voltage_seqs": v_seqs,
        "current_seqs": i_seqs,
        "label_detection": np.array(l_det, dtype=np.int64),
        "label_type":      np.array(l_type, dtype=np.int64),
        "label_phase":     np.array(l_phase, dtype=np.int64),
        "label_location":  np.array(l_loc, dtype=np.int64),
        "edge_indices":    edge_indices,
        "edge_attrs":      edge_attrs,
        "metadata":        meta,
    }


def split_indices(n: int, label_type: np.ndarray) -> tuple:
    """Stratified train/val/test index split."""
    idx = np.arange(n)
    idx_train, idx_temp, _, y_temp = train_test_split(
        idx, label_type, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=label_type, random_state=RANDOM_SEED
    )
    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=1 - val_frac, stratify=y_temp, random_state=RANDOM_SEED
    )
    logger.info(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return idx_train, idx_val, idx_test


def save_numpy_split(indices: np.ndarray, samples: dict,
                     normalizer, out_dir: Path, split_name: str) -> None:
    """Save a split as a single NPZ bundle (for CNN-Transformer)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    v_seqs = [samples["voltage_seqs"][i] for i in indices]
    i_seqs = [samples["current_seqs"][i] for i in indices]

    # Normalize
    v_norm = np.stack([normalizer.transform_v(v) for v in v_seqs], axis=0)
    i_norm = np.stack([normalizer.transform_i(i) for i in i_seqs], axis=0)

    np.savez_compressed(
        out_dir / f"{split_name}.npz",
        voltage_seq      = v_norm,
        current_seq      = i_norm,
        label_detection  = samples["label_detection"][indices],
        label_type       = samples["label_type"][indices],
        label_phase      = samples["label_phase"][indices],
        label_location   = samples["label_location"][indices],
    )
    logger.info(f"Saved {split_name}: {len(indices)} samples → {out_dir}/{split_name}.npz")


def save_pyg_split(indices: np.ndarray, samples: dict,
                   normalizer, out_dir: Path, split_name: str) -> None:
    """Save a split as list of PyG Data objects (for R-GNN)."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        logger.warning("torch_geometric not available — skipping PyG split")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    data_list = []

    for i in indices:
        V = torch.tensor(normalizer.transform_v(samples["voltage_seqs"][i]),
                         dtype=torch.float32)  # [T, N_buses, 6]
        edge_idx = torch.tensor(samples["edge_indices"][i], dtype=torch.long)
        edge_a   = torch.tensor(samples["edge_attrs"][i],   dtype=torch.float32)

        d = Data(
            x          = V,
            edge_index = edge_idx,
            edge_attr  = edge_a,
            y_detection = torch.tensor(samples["label_detection"][i], dtype=torch.long),
            y_type      = torch.tensor(samples["label_type"][i],      dtype=torch.long),
            y_phase     = torch.tensor(samples["label_phase"][i],     dtype=torch.long),
            y_location  = torch.tensor(samples["label_location"][i],  dtype=torch.long),
        )
        data_list.append(d)

    out_path = out_dir / f"{split_name}_pyg.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data_list, f)
    logger.info(f"Saved PyG {split_name}: {len(data_list)} graphs → {out_path}")

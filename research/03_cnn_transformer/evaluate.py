"""
Evaluation script for CNN-Transformer model (Stage 03).
Loads best checkpoint, runs on test set, prints per-task metrics.

Usage:
    python research/03_cnn_transformer/evaluate.py

IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import logging
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "06_model_evaluation"))

from model import CNNTransformerModel
from train import FaultDataset
from config import CKPT_DIR, TEST_NPZ
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("cnn_transformer.evaluate")


def main():
    ckpt_path = CKPT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        logger.error(f"No checkpoint found at {ckpt_path}. Run train.py first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = CNNTransformerModel(
        in_features=ckpt["in_features"],
        n_buses=ckpt["n_buses"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds     = FaultDataset(TEST_NPZ)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = {k: [] for k in ["detection", "type", "phase", "location"]}
    all_true  = {k: [] for k in ["detection", "type", "phase", "location"]}
    task_keys = ["detection", "type", "phase", "location"]

    with torch.no_grad():
        for X, y_det, y_type, y_ph, y_loc in test_loader:
            out = model(X)
            for key, ytrue in zip(task_keys, [y_det, y_type, y_ph, y_loc]):
                all_preds[key].extend(out[key].argmax(1).numpy().tolist())
                all_true[key].extend(ytrue.numpy().tolist())

    # Compute metrics using metrics.py
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "06_model_evaluation"))
        from metrics import compute_all_metrics
        results = {}
        for key in task_keys:
            results[key] = compute_all_metrics(
                np.array(all_true[key]), np.array(all_preds[key]), key
            )
        logger.info("\n=== CNN-Transformer Test Results ===")
        for key, m in results.items():
            logger.info(f"  {key:12s}: Acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f} MCC={m['mcc']:.4f}")
    except ImportError:
        # Fall back to basic accuracy
        for key in task_keys:
            arr_true = np.array(all_true[key])
            arr_pred = np.array(all_preds[key])
            acc = (arr_true == arr_pred).mean()
            logger.info(f"  {key:12s}: Accuracy={acc:.4f}")


if __name__ == "__main__":
    main()

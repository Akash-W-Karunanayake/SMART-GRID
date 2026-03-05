"""
Evaluation for R-GNN model (Stage 04).
IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model import RGNNModel
from train import GraphFaultDataset, collate_fn
from config import CKPT_DIR, TEST_PKL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("r_gnn.evaluate")


def main():
    ckpt_path = CKPT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        logger.error(f"No checkpoint at {ckpt_path}")
        sys.exit(1)

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    model  = RGNNModel(n_buses=ckpt["n_buses"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = GraphFaultDataset(TEST_PKL)
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    ei = test_ds.edge_index
    ea = test_ds.edge_attr

    preds = {k: [] for k in ["detection", "type", "phase", "location"]}
    trues = {k: [] for k in ["detection", "type", "phase", "location"]}
    task_keys = ["detection", "type", "phase", "location"]

    with torch.no_grad():
        for x, y_det, y_type, y_ph, y_loc in loader:
            out = model(x, ei, ea)
            for key, ytrue in zip(task_keys, [y_det, y_type, y_ph, y_loc]):
                preds[key].extend(out[key].argmax(1).numpy().tolist())
                trues[key].extend(ytrue.numpy().tolist())

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "06_model_evaluation"))
        from metrics import compute_all_metrics
        logger.info("\n=== R-GNN Test Results ===")
        for key in task_keys:
            m = compute_all_metrics(np.array(trues[key]), np.array(preds[key]), key)
            logger.info(f"  {key:12s}: Acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f} MCC={m['mcc']:.4f}")
    except ImportError:
        for key in task_keys:
            acc = (np.array(trues[key]) == np.array(preds[key])).mean()
            logger.info(f"  {key}: Accuracy={acc:.4f}")


if __name__ == "__main__":
    main()

"""
Training loop for R-GNN model (Stage 04).

Usage:
    python research/04_r_gnn/train.py
    python research/04_r_gnn/train.py --epochs 5

IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model import RGNNModel
from config import (
    TRAIN_PKL, VAL_PKL, CKPT_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    LAMBDA_DETECT, LAMBDA_TYPE, LAMBDA_PHASE, LAMBDA_LOCATION,
    T_STEPS, N_CLASSES_TYPE, WEIGHTS_PATH,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r_gnn.train")


class GraphFaultDataset(Dataset):
    """Wraps PyG Data list into a standard Dataset returning tensors."""

    def __init__(self, pkl_path: Path):
        if not pkl_path.exists():
            raise FileNotFoundError(f"PyG dataset not found: {pkl_path}\n"
                                    "Run Stage 02 preprocessing first.")
        with open(pkl_path, "rb") as f:
            self.data_list = pickle.load(f)
        # Extract shared edge_index from first sample (topology is constant)
        d0 = self.data_list[0]
        self.edge_index = d0.edge_index   # [2, E]
        self.edge_attr  = d0.edge_attr    # [E, 3]
        self.n_buses    = d0.x.shape[1] if d0.x.dim() == 3 else d0.x.shape[0]

    def __len__(self): return len(self.data_list)

    def __getitem__(self, i):
        d = self.data_list[i]
        # d.x: [T, N_buses, 6]
        return (
            d.x,
            d.y_detection,
            d.y_type,
            d.y_phase,
            d.y_location,
        )


def collate_fn(batch):
    x, y_det, y_type, y_ph, y_loc = zip(*batch)
    return (
        torch.stack(x),
        torch.stack(y_det),
        torch.stack(y_type),
        torch.stack(y_ph),
        torch.stack(y_loc),
    )


def make_loss_fns(class_weights_dict, device):
    w = torch.ones(N_CLASSES_TYPE, dtype=torch.float32)
    for cls, wt in class_weights_dict.items():
        if cls < N_CLASSES_TYPE:
            w[cls] = wt
    return (
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(weight=w.to(device)),
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
    )


def compute_loss(out, labels, loss_fns):
    ce_det, ce_type, ce_ph, ce_loc = loss_fns
    y_det, y_type, y_ph, y_loc = labels
    return (
        LAMBDA_DETECT   * ce_det(out["detection"], y_det)  +
        LAMBDA_TYPE     * ce_type(out["type"],     y_type) +
        LAMBDA_PHASE    * ce_ph(out["phase"],      y_ph)   +
        LAMBDA_LOCATION * ce_loc(out["location"],  y_loc)
    )


def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def train_epoch(model, loader, optimizer, loss_fns, device, edge_index, edge_attr):
    model.train()
    total = 0.0
    ei = edge_index.to(device)
    ea = edge_attr.to(device)
    for x, y_det, y_type, y_ph, y_loc in loader:
        x = x.to(device)
        labels = (y_det.to(device), y_type.to(device),
                  y_ph.to(device), torch.clamp(y_loc.to(device), min=0))
        optimizer.zero_grad()
        out = model(x, ei, ea)
        loss = compute_loss(out, labels, loss_fns)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fns, device, edge_index, edge_attr):
    model.eval()
    total = 0.0
    accs = [0.0] * 4
    ei = edge_index.to(device)
    ea = edge_attr.to(device)
    for x, y_det, y_type, y_ph, y_loc in loader:
        x = x.to(device)
        labels = (y_det.to(device), y_type.to(device),
                  y_ph.to(device), torch.clamp(y_loc.to(device), min=0))
        out = model(x, ei, ea)
        total += compute_loss(out, labels, loss_fns).item()
        for k, (key, lbl) in enumerate(
                zip(["detection","type","phase","location"], labels)):
            accs[k] += accuracy(out[key], lbl)
    n = len(loader)
    return total/n, *(a/n for a in accs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading PyG datasets...")
    train_ds = GraphFaultDataset(TRAIN_PKL)
    val_ds   = GraphFaultDataset(VAL_PKL)
    n_buses  = train_ds.n_buses
    edge_index = train_ds.edge_index
    edge_attr  = train_ds.edge_attr

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    class_weights = {}
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH, "rb") as f:
            class_weights = pickle.load(f)

    model = RGNNModel(n_buses=n_buses).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"R-GNN params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fns  = make_loss_fns(class_weights, device)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, loss_fns, device, edge_index, edge_attr)
        vl, a_det, a_type, a_ph, a_loc = eval_epoch(
            model, val_loader, loss_fns, device, edge_index, edge_attr)
        scheduler.step(vl)

        history.append({"epoch": epoch, "train_loss": tr, "val_loss": vl,
                         "acc_detect": a_det, "acc_type": a_type,
                         "acc_phase": a_ph,   "acc_location": a_loc})

        logger.info(f"Epoch {epoch:3d}/{args.epochs} | "
                    f"tr={tr:.4f} vl={vl:.4f} | "
                    f"det={a_det:.3f} type={a_type:.3f} phase={a_ph:.3f} loc={a_loc:.3f}")

        if vl < best_val:
            best_val = vl
            patience_count = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "n_buses": n_buses, "val_loss": vl,
            }, CKPT_DIR / "best_model.pt")
            logger.info(f"  ✓ Best model saved")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"R-GNN training complete. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()

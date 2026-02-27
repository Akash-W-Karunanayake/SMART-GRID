"""
End-to-end training for the Hybrid CNN-Transformer + R-GNN model (Stage 05).

Requires both NPZ (for current sequences) and PyG pkl (for voltage graphs).

Usage:
    python research/05_hybrid_model/train.py
    python research/05_hybrid_model/train.py --epochs 5

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
sys.path.insert(0, str(Path(__file__).parent.parent / "04_r_gnn"))

from model import HybridModel
from config import (
    TRAIN_NPZ, VAL_NPZ, TRAIN_PKL, VAL_PKL, WEIGHTS_PATH, CKPT_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    LAMBDA_DETECT, LAMBDA_TYPE, LAMBDA_PHASE, LAMBDA_LOCATION,
    N_CLASSES_TYPE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid.train")


class HybridDataset(Dataset):
    """Joint dataset: aligns NPZ current sequences with PyG voltage graphs."""

    def __init__(self, npz_path: Path, pkl_path: Path):
        # Load NPZ
        d = np.load(npz_path, allow_pickle=True)
        I = d["current_seq"]          # [N, T, N_branches, 3]
        N, T, B, F = I.shape
        self.X_current = torch.tensor(I.reshape(N, T, B * F), dtype=torch.float32)
        self.y_det  = torch.tensor(d["label_detection"], dtype=torch.long)
        self.y_type = torch.tensor(d["label_type"],      dtype=torch.long)
        self.y_ph   = torch.tensor(d["label_phase"],     dtype=torch.long)
        self.y_loc  = torch.clamp(
            torch.tensor(d["label_location"], dtype=torch.long), min=0)

        # Load PyG voltage graphs
        with open(pkl_path, "rb") as f:
            self.pyg_list = pickle.load(f)

        # Trim to same length (should match)
        n_min = min(N, len(self.pyg_list))
        self.X_current = self.X_current[:n_min]
        self.y_det  = self.y_det[:n_min]
        self.y_type = self.y_type[:n_min]
        self.y_ph   = self.y_ph[:n_min]
        self.y_loc  = self.y_loc[:n_min]
        self.pyg_list = self.pyg_list[:n_min]

        d0 = self.pyg_list[0]
        self.edge_index = d0.edge_index   # [2, E]
        self.edge_attr  = d0.edge_attr    # [E, 3]
        self.n_buses    = d0.x.shape[1] if d0.x.dim() == 3 else d0.x.shape[0]
        self.n_branches = B

    def __len__(self): return len(self.X_current)

    def __getitem__(self, i):
        pyg = self.pyg_list[i]
        x_v = pyg.x   # [T, N_buses, 6]
        return (
            self.X_current[i],
            x_v,
            self.y_det[i], self.y_type[i], self.y_ph[i], self.y_loc[i],
        )


def collate_fn(batch):
    x_curr, x_volt, y_det, y_type, y_ph, y_loc = zip(*batch)
    return (
        torch.stack(x_curr),
        torch.stack(x_volt),
        torch.stack(y_det), torch.stack(y_type),
        torch.stack(y_ph),  torch.stack(y_loc),
    )


def make_loss_fns(class_weights, device):
    w = torch.ones(N_CLASSES_TYPE)
    for cls, wt in class_weights.items():
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


def run_epoch(model, loader, optimizer, loss_fns, device, ei, ea, train=True):
    model.train(train)
    total = 0.0
    accs = [0.0] * 4
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_cur, x_vol, y_det, y_type, y_ph, y_loc in loader:
            x_cur  = x_cur.to(device)
            x_vol  = x_vol.to(device)
            labels = (y_det.to(device), y_type.to(device),
                      y_ph.to(device), y_loc.to(device))
            if train:
                optimizer.zero_grad()
            out  = model(x_cur, x_vol, ei.to(device), ea.to(device))
            loss = compute_loss(out, labels, loss_fns)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            for k, key in enumerate(["detection","type","phase","location"]):
                accs[k] += accuracy(out[key], labels[k])
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

    logger.info("Loading hybrid datasets...")
    train_ds = HybridDataset(TRAIN_NPZ, TRAIN_PKL)
    val_ds   = HybridDataset(VAL_NPZ,   VAL_PKL)

    ei = train_ds.edge_index
    ea = train_ds.edge_attr
    n_buses    = train_ds.n_buses
    n_branches = train_ds.n_branches
    in_feat    = n_branches * 3

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    class_weights = {}
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH, "rb") as f:
            class_weights = pickle.load(f)

    model = HybridModel(in_features_cnn=in_feat, n_buses=n_buses).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Hybrid model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fns  = make_loss_fns(class_weights, device)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr, *tr_accs = run_epoch(model, train_loader, optimizer, loss_fns, device, ei, ea, True)
        vl, a_det, a_type, a_ph, a_loc = run_epoch(
            model, val_loader, None, loss_fns, device, ei, ea, False)
        scheduler.step()

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
                "n_buses": n_buses, "in_features_cnn": in_feat, "val_loss": vl,
            }, CKPT_DIR / "best_model.pt")
            logger.info(f"  ✓ Best hybrid model saved")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Hybrid training complete. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()

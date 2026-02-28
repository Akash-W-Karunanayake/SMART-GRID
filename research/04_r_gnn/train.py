"""
Training loop for R-GNN model (Stage 04) — Phase 2 transfer learning.

Phase 2a: Train detection only (20 epochs)
Phase 2b: Freeze GCN+GRU, train new heads (15 epochs), then unfreeze + fine-tune (10 epochs)

Usage:
    python research/04_r_gnn/train.py --phase 2a
    python research/04_r_gnn/train.py --phase 2b

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
    PHASE2B_EPOCHS, PHASE2B_LR_HEADS, PHASE2B_LR_FINETUNE,
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
        d0 = self.data_list[0]
        self.edge_index = d0.edge_index
        self.edge_attr  = d0.edge_attr
        self.n_buses    = d0.x.shape[1] if d0.x.dim() == 3 else d0.x.shape[0]

    def __len__(self): return len(self.data_list)

    def __getitem__(self, i):
        d = self.data_list[i]
        return (d.x, d.y_detection, d.y_type, d.y_phase, d.y_location)


def collate_fn(batch):
    x, y_det, y_type, y_ph, y_loc = zip(*batch)
    return (
        torch.stack(x), torch.stack(y_det), torch.stack(y_type),
        torch.stack(y_ph), torch.stack(y_loc),
    )


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loss_fns(class_weights_dict, device):
    w = torch.ones(N_CLASSES_TYPE, dtype=torch.float32)
    for cls, wt in class_weights_dict.items():
        if cls < N_CLASSES_TYPE:
            w[cls] = wt
    return (
        nn.CrossEntropyLoss(),
        FocalLoss(gamma=2.0, weight=w.to(device)),
        FocalLoss(gamma=2.0),
        nn.CrossEntropyLoss(),
    )


def compute_loss(out, labels, loss_fns, lambdas):
    ce_det, fl_type, fl_ph, ce_loc = loss_fns
    y_det, y_type, y_ph, y_loc = labels
    ld, lt, lp, ll = lambdas
    loss = ld * ce_det(out["detection"], y_det)
    if lt > 0:
        loss = loss + lt * fl_type(out["type"], y_type)
    if lp > 0:
        loss = loss + lp * fl_ph(out["phase"], y_ph)
    if ll > 0:
        loss = loss + ll * ce_loc(out["location"], y_loc)
    return loss


def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def train_epoch(model, loader, optimizer, loss_fns, device, ei, ea, lambdas):
    model.train()
    total = 0.0
    for x, y_det, y_type, y_ph, y_loc in loader:
        x = x.to(device)
        labels = (y_det.to(device), y_type.to(device),
                  y_ph.to(device), torch.clamp(y_loc.to(device), min=0))
        optimizer.zero_grad()
        out = model(x, ei, ea)
        loss = compute_loss(out, labels, loss_fns, lambdas)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fns, device, ei, ea, lambdas):
    model.eval()
    total = 0.0
    accs = [0.0] * 4
    for x, y_det, y_type, y_ph, y_loc in loader:
        x = x.to(device)
        labels = (y_det.to(device), y_type.to(device),
                  y_ph.to(device), torch.clamp(y_loc.to(device), min=0))
        out = model(x, ei, ea)
        total += compute_loss(out, labels, loss_fns, lambdas).item()
        for k, (key, lbl) in enumerate(
                zip(["detection","type","phase","location"], labels)):
            accs[k] += accuracy(out[key], lbl)
    n = len(loader)
    return total/n, *(a/n for a in accs)


def _run_training_loop(model, train_loader, val_loader, optimizer, scheduler,
                       loss_fns, device, ei, ea, lambdas, epochs, patience,
                       ckpt_path, history):
    best_val = float("inf")
    patience_count = 0
    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, loss_fns, device, ei, ea, lambdas)
        vl, a_det, a_type, a_ph, a_loc = eval_epoch(
            model, val_loader, loss_fns, device, ei, ea, lambdas)
        if scheduler:
            scheduler.step(vl)

        history.append({"epoch": len(history)+1, "train_loss": tr, "val_loss": vl,
                         "acc_detect": a_det, "acc_type": a_type,
                         "acc_phase": a_ph,   "acc_location": a_loc})

        logger.info(f"Epoch {epoch:3d}/{epochs} | "
                    f"tr={tr:.4f} vl={vl:.4f} | "
                    f"det={a_det:.3f} type={a_type:.3f} phase={a_ph:.3f} loc={a_loc:.3f}")

        if vl < best_val:
            best_val = vl
            patience_count = 0
            torch.save({
                "epoch": len(history), "model_state": model.state_dict(),
                "n_buses": model.n_buses, "val_loss": vl,
            }, ckpt_path)
            logger.info(f"  ✓ Best model saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    return best_val


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, default="2a", choices=["2a", "2b"])
    p.add_argument("--epochs", type=int, default=None)
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
    ei = train_ds.edge_index.to(device)
    ea = train_ds.edge_attr.to(device)

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

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    loss_fns = make_loss_fns(class_weights, device)
    history = []

    if args.phase == "2a":
        # Phase 2a: detection only
        epochs = args.epochs or EPOCHS
        lambdas = (LAMBDA_DETECT, LAMBDA_TYPE, LAMBDA_PHASE, LAMBDA_LOCATION)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        best_val = _run_training_loop(
            model, train_loader, val_loader, optimizer, scheduler,
            loss_fns, device, ei, ea, lambdas, epochs, PATIENCE,
            CKPT_DIR / "best_model.pt", history)
        logger.info(f"Phase 2a complete. Best val_loss={best_val:.4f}")

    elif args.phase == "2b":
        # Phase 2b: load Phase 2a checkpoint, transfer learning
        ckpt_path = CKPT_DIR / "best_model.pt"
        if not ckpt_path.exists():
            logger.error("Phase 2a checkpoint not found. Run --phase 2a first.")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Loaded Phase 2a checkpoint for transfer learning")

        # All tasks enabled
        lambdas = (1.0, 1.0, 0.5, 1.5)

        # Step 1: Freeze GCN+GRU, train new heads (15 epochs)
        for p in model.gcn_layers.parameters():
            p.requires_grad = False
        for p in model.gcn_norms.parameters():
            p.requires_grad = False
        for p in model.gru.parameters():
            p.requires_grad = False

        head_params = list(model.head_detection.parameters()) + \
                     list(model.head_type.parameters()) + \
                     list(model.head_phase.parameters()) + \
                     list(model.head_location.parameters())
        optimizer = torch.optim.Adam(head_params, lr=PHASE2B_LR_HEADS, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        logger.info("Phase 2b Step 1: Training heads only (frozen backbone)...")
        _run_training_loop(
            model, train_loader, val_loader, optimizer, scheduler,
            loss_fns, device, ei, ea, lambdas, 15, PATIENCE,
            CKPT_DIR / "best_model_2b_heads.pt", history)

        # Step 2: Unfreeze all, fine-tune (10 epochs)
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2B_LR_FINETUNE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        logger.info("Phase 2b Step 2: Full fine-tuning...")
        best_val = _run_training_loop(
            model, train_loader, val_loader, optimizer, scheduler,
            loss_fns, device, ei, ea, lambdas, 10, PATIENCE,
            CKPT_DIR / "best_model.pt", history)
        logger.info(f"Phase 2b complete. Best val_loss={best_val:.4f}")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("R-GNN training complete.")


if __name__ == "__main__":
    main()

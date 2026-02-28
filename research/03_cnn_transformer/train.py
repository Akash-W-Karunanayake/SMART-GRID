"""
Training loop for CNN-Transformer model (Stage 03) — Phase 1 pre-training.
Trains on detection + type tasks only (phase/location disabled via config).
Supports focal loss, weighted CE, and early stopping.

Usage:
    python research/03_cnn_transformer/train.py
    python research/03_cnn_transformer/train.py --epochs 20

IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model import CNNTransformerModel
from config import (
    TRAIN_NPZ, VAL_NPZ, WEIGHTS_PATH, CKPT_DIR, DEVICE,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    LAMBDA_DETECT, LAMBDA_TYPE, LAMBDA_PHASE, LAMBDA_LOCATION,
    N_CLASSES_DETECTION, N_CLASSES_TYPE, N_CLASSES_PHASE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cnn_transformer.train")


class FaultDataset(Dataset):
    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=True)
        # current_seq: [N, T, N_branches, 6] (mag+ang per phase)
        I = d["current_seq"]
        N, T, B, F = I.shape
        self.X = torch.tensor(I.reshape(N, T, B * F), dtype=torch.float32)
        self.y_det  = torch.tensor(d["label_detection"], dtype=torch.long)
        self.y_type = torch.tensor(d["label_type"],      dtype=torch.long)
        self.y_ph   = torch.tensor(d["label_phase"],     dtype=torch.long)
        self.y_loc  = torch.tensor(d["label_location"],  dtype=torch.long)
        # Clamp negative location (-1 = normal) to 0 for CE loss
        self.y_loc = torch.clamp(self.y_loc, min=0)
        self.n_buses = int(self.y_loc.max().item()) + 1

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y_det[i], self.y_type[i], self.y_ph[i], self.y_loc[i]


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loss_fns(class_weights_dict: dict, device: torch.device):
    """Build loss functions with Focal Loss for type head."""
    # Type head: focal loss with class weights
    w = torch.ones(N_CLASSES_TYPE, dtype=torch.float32)
    for cls, wt in class_weights_dict.items():
        if cls < N_CLASSES_TYPE:
            w[cls] = wt
    ce_det  = nn.CrossEntropyLoss()
    fl_type = FocalLoss(gamma=2.0, weight=w.to(device))
    ce_ph   = nn.CrossEntropyLoss()
    ce_loc  = nn.CrossEntropyLoss()
    return ce_det, fl_type, ce_ph, ce_loc


def compute_loss(outputs, labels, loss_fns):
    ce_det, fl_type, ce_ph, ce_loc = loss_fns
    y_det, y_type, y_ph, y_loc = labels
    loss = LAMBDA_DETECT * ce_det(outputs["detection"], y_det)
    if LAMBDA_TYPE > 0:
        loss = loss + LAMBDA_TYPE * fl_type(outputs["type"], y_type)
    if LAMBDA_PHASE > 0:
        loss = loss + LAMBDA_PHASE * ce_ph(outputs["phase"], y_ph)
    if LAMBDA_LOCATION > 0:
        loss = loss + LAMBDA_LOCATION * ce_loc(outputs["location"], y_loc)
    return loss


def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def train_epoch(model, loader, optimizer, loss_fns, device):
    model.train()
    total_loss = 0.0
    for X, y_det, y_type, y_ph, y_loc in loader:
        X      = X.to(device)
        labels = (y_det.to(device), y_type.to(device), y_ph.to(device), y_loc.to(device))
        optimizer.zero_grad()
        out = model(X)
        loss = compute_loss(out, labels, loss_fns)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fns, device):
    model.eval()
    total_loss = 0.0
    acc_det = acc_type = acc_ph = acc_loc = 0.0
    for X, y_det, y_type, y_ph, y_loc in loader:
        X = X.to(device)
        labels = (y_det.to(device), y_type.to(device), y_ph.to(device), y_loc.to(device))
        out = model(X)
        total_loss += compute_loss(out, labels, loss_fns).item()
        acc_det  += accuracy(out["detection"], labels[0])
        acc_type += accuracy(out["type"],      labels[1])
        acc_ph   += accuracy(out["phase"],     labels[2])
        acc_loc  += accuracy(out["location"],  labels[3])
    n = len(loader)
    return total_loss/n, acc_det/n, acc_type/n, acc_ph/n, acc_loc/n


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

    # Load datasets
    logger.info("Loading datasets...")
    train_ds = FaultDataset(TRAIN_NPZ)
    val_ds   = FaultDataset(VAL_NPZ)

    n_buses  = max(train_ds.n_buses, val_ds.n_buses)
    in_feat  = train_ds.X.shape[-1]    # N_branches * 6

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weights
    class_weights = {}
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH, "rb") as f:
            class_weights = pickle.load(f)

    # Model
    model = CNNTransformerModel(in_features=in_feat, n_buses=n_buses).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fns  = make_loss_fns(class_weights, device)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fns, device)
        vl_loss, a_det, a_type, a_ph, a_loc = eval_epoch(model, val_loader, loss_fns, device)
        scheduler.step(vl_loss)

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss,
                         "acc_detect": a_det, "acc_type": a_type, "acc_phase": a_ph,
                         "acc_location": a_loc})

        logger.info(f"Epoch {epoch:3d}/{args.epochs} | "
                    f"tr_loss={tr_loss:.4f} vl_loss={vl_loss:.4f} | "
                    f"det={a_det:.3f} type={a_type:.3f} phase={a_ph:.3f} loc={a_loc:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": vl_loss,
                "in_features": in_feat,
                "n_buses": n_buses,
            }, CKPT_DIR / "best_model.pt")
            logger.info(f"  ✓ Best model saved (val_loss={vl_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Save training history
    import json
    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Checkpoint: {CKPT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()

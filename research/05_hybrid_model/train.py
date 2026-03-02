"""
3-Phase Curriculum Training for BHAF Hybrid Model (Stage 05).

Phase 3a: Freeze branches, train fusion + heads only (warm-up)
Phase 3b: Unfreeze all, end-to-end with differential learning rates

Requires pre-trained branch checkpoints from Stages 03 and 04.

Usage:
    python research/05_hybrid_model/train.py
    python research/05_hybrid_model/train.py --skip-warmup

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "04_r_gnn"))
sys.path.insert(0, str(Path(__file__).parent))

from model import HybridModel, FocalLoss
from config import (
    TRAIN_NPZ, VAL_NPZ, TRAIN_PKL, VAL_PKL, WEIGHTS_PATH, CKPT_DIR,
    CNN_T_CKPT, RGNN_CKPT,
    BATCH_SIZE, WEIGHT_DECAY, PATIENCE, MAX_GRAD_NORM,
    PHASE3A_EPOCHS, PHASE3A_LR,
    PHASE3B_EPOCHS, PHASE3B_LR_BRANCHES, PHASE3B_LR_FUSION,
    WARMUP_EPOCHS,
    N_CLASSES_TYPE, N_CLASSES_PHASE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid.train")


class HybridDataset(Dataset):
    """Joint dataset: aligns NPZ current sequences with PyG voltage graphs."""

    def __init__(self, npz_path: Path, pkl_path: Path):
        d = np.load(npz_path, allow_pickle=True)
        I = d["current_seq"]          # [N, T, N_branches, 6]
        N, T, B, Feat = I.shape
        self.X_current = torch.tensor(I.reshape(N, T, B * Feat), dtype=torch.float32)
        self.y_det  = torch.tensor(d["label_detection"], dtype=torch.long)
        self.y_type = torch.tensor(d["label_type"],      dtype=torch.long)
        self.y_ph   = torch.tensor(d["label_phase"],     dtype=torch.long)
        raw_loc     = torch.tensor(d["label_location"],  dtype=torch.long)

        with open(pkl_path, "rb") as f:
            self.pyg_list = pickle.load(f)

        n_min = min(N, len(self.pyg_list))
        self.X_current = self.X_current[:n_min]
        self.y_det  = self.y_det[:n_min]
        self.y_type = self.y_type[:n_min]
        self.y_ph   = self.y_ph[:n_min]
        raw_loc     = raw_loc[:n_min]
        self.pyg_list = self.pyg_list[:n_min]

        d0 = self.pyg_list[0]
        self.edge_index = d0.edge_index
        self.edge_attr  = d0.edge_attr
        self.n_buses    = d0.x.shape[1] if d0.x.dim() == 3 else d0.x.shape[0]
        self.n_branches = B

        # Fix: map -1 (no-fault) → n_buses (last class) instead of 0
        self.y_loc = raw_loc.clone()
        self.y_loc[self.y_loc < 0] = self.n_buses

    def __len__(self): return len(self.X_current)

    def __getitem__(self, i):
        pyg = self.pyg_list[i]
        return (
            self.X_current[i], pyg.x,
            self.y_det[i], self.y_type[i], self.y_ph[i], self.y_loc[i],
        )


def collate_fn(batch):
    x_curr, x_volt, y_det, y_type, y_ph, y_loc = zip(*batch)
    return (
        torch.stack(x_curr), torch.stack(x_volt),
        torch.stack(y_det), torch.stack(y_type),
        torch.stack(y_ph),  torch.stack(y_loc),
    )


def _inverse_freq_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from label tensor."""
    counts = torch.bincount(labels, minlength=n_classes).float()
    counts = counts.clamp(min=1.0)  # avoid div-by-zero
    weights = counts.sum() / (n_classes * counts)
    return weights


def make_loss_fns(class_weights, device, n_buses: int,
                  loc_labels: torch.Tensor, phase_labels: torch.Tensor):
    # Type weights from pre-computed class_weights dict
    w_type = torch.ones(N_CLASSES_TYPE)
    for cls, wt in class_weights.items():
        if cls < N_CLASSES_TYPE:
            w_type[cls] = wt

    # Phase weights from training labels
    w_phase = _inverse_freq_weights(phase_labels, N_CLASSES_PHASE)

    # Location weights from training labels (N+1 classes: N buses + no-fault)
    n_loc_classes = n_buses + 1
    w_loc = _inverse_freq_weights(loc_labels, n_loc_classes)

    return (
        nn.CrossEntropyLoss(),                                    # detection
        FocalLoss(gamma=2.0, weight=w_type.to(device)),           # type
        FocalLoss(gamma=2.0, weight=w_phase.to(device)),          # phase
        FocalLoss(gamma=1.0, weight=w_loc.to(device)),            # location
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
            x_cur = x_cur.to(device)
            x_vol = x_vol.to(device)
            labels = (y_det.to(device), y_type.to(device),
                      y_ph.to(device), y_loc.to(device))
            if train:
                optimizer.zero_grad()
            out = model(x_cur, x_vol, ei, ea)
            loss = model.compute_multitask_loss(out, labels, loss_fns)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            total += loss.item()
            for k, key in enumerate(["detection","type","phase","location"]):
                accs[k] += accuracy(out[key], labels[k])
    n = len(loader)
    return total/n, *(a/n for a in accs)


def _training_loop(model, train_loader, val_loader, optimizer, scheduler,
                   loss_fns, device, ei, ea, epochs, patience, history, tag=""):
    best_val = float("inf")
    patience_count = 0
    for epoch in range(1, epochs + 1):
        tr, *_ = run_epoch(model, train_loader, optimizer, loss_fns, device, ei, ea, True)
        vl, a_det, a_type, a_ph, a_loc = run_epoch(
            model, val_loader, None, loss_fns, device, ei, ea, False)
        if scheduler:
            scheduler.step()

        history.append({"epoch": len(history)+1, "phase": tag,
                         "train_loss": tr, "val_loss": vl,
                         "acc_detect": a_det, "acc_type": a_type,
                         "acc_phase": a_ph,   "acc_location": a_loc})

        logger.info(f"[{tag}] Epoch {epoch:3d}/{epochs} | "
                    f"tr={tr:.4f} vl={vl:.4f} | "
                    f"det={a_det:.3f} type={a_type:.3f} "
                    f"phase={a_ph:.3f} loc={a_loc:.3f}")

        if vl < best_val:
            best_val = vl
            patience_count = 0
            torch.save({
                "epoch": len(history),
                "model_state": model.state_dict(),
                "n_buses": model.n_buses,
                "in_features_cnn": train_loader.dataset.X_current.shape[-1],
                "val_loss": vl,
            }, CKPT_DIR / "best_model.pt")
            logger.info(f"  ✓ Best hybrid model saved (vl={vl:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    return best_val


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-warmup", action="store_true",
                   help="Skip Phase 3a warm-up (use for resumed training)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading hybrid datasets...")
    train_ds = HybridDataset(TRAIN_NPZ, TRAIN_PKL)
    val_ds   = HybridDataset(VAL_NPZ,   VAL_PKL)

    ei = train_ds.edge_index.to(device)
    ea = train_ds.edge_attr.to(device)
    n_buses    = train_ds.n_buses
    in_feat    = train_ds.X_current.shape[-1]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    class_weights = {}
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH, "rb") as f:
            class_weights = pickle.load(f)

    # Build model and load pre-trained branches
    model = HybridModel(in_features_cnn=in_feat, n_buses=n_buses).to(device)
    model.load_pretrained_branches(CNN_T_CKPT, RGNN_CKPT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Hybrid model params: {n_params:,}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    loss_fns = make_loss_fns(class_weights, device, n_buses=n_buses,
                             loc_labels=train_ds.y_loc,
                             phase_labels=train_ds.y_ph)
    history = []

    # ========== Phase 3a: Warm-up (frozen branches) ==========
    if not args.skip_warmup:
        logger.info("=" * 60)
        logger.info("Phase 3a: Warm-up — training fusion + heads only")
        logger.info("=" * 60)

        model.freeze_branches(freeze_cnn=True, freeze_gnn=True)

        # Only optimize fusion, heads, and uncertainty params
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=PHASE3A_LR, weight_decay=WEIGHT_DECAY)

        _training_loop(model, train_loader, val_loader, optimizer, None,
                       loss_fns, device, ei, ea,
                       PHASE3A_EPOCHS, PATIENCE, history, "3a-warmup")

    # ========== Phase 3b: End-to-end fine-tuning ==========
    logger.info("=" * 60)
    logger.info("Phase 3b: End-to-end fine-tuning with differential LR")
    logger.info("=" * 60)

    model.freeze_branches(freeze_cnn=False, freeze_gnn=False)

    # Differential learning rates
    branch_params = list(model.cnn_transformer.parameters()) + \
                   list(model.r_gnn.parameters())
    fusion_params = list(model.fusion.parameters()) + \
                   list(model.head_detection.parameters()) + \
                   list(model.head_type.parameters()) + \
                   list(model.head_phase.parameters()) + \
                   list(model.loc_node_proj.parameters()) + \
                   list(model.loc_nofault.parameters()) + \
                   list(model.uncertainty_loss.parameters())

    optimizer = torch.optim.Adam([
        {"params": branch_params, "lr": PHASE3B_LR_BRANCHES},
        {"params": fusion_params, "lr": PHASE3B_LR_FUSION},
    ], weight_decay=WEIGHT_DECAY)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE3B_EPOCHS - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS])

    best_val = _training_loop(model, train_loader, val_loader, optimizer, scheduler,
                              loss_fns, device, ei, ea,
                              PHASE3B_EPOCHS, PATIENCE, history, "3b-finetune")

    # Save history
    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Hybrid training complete. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()

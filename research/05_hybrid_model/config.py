"""
Hybrid CNN-Transformer + R-GNN Hyperparameters — v2 BHAF.
IT22577924 — Karunanayake K.P.A.W.
"""
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent.parent

# Data paths
TRAIN_NPZ  = RESEARCH_DIR / "datasets" / "processed" / "train" / "train.npz"
VAL_NPZ    = RESEARCH_DIR / "datasets" / "processed" / "val"   / "val.npz"
TEST_NPZ   = RESEARCH_DIR / "datasets" / "processed" / "test"  / "test.npz"
TRAIN_PKL  = RESEARCH_DIR / "datasets" / "processed" / "train" / "train_pyg.pkl"
VAL_PKL    = RESEARCH_DIR / "datasets" / "processed" / "val"   / "val_pyg.pkl"
TEST_PKL   = RESEARCH_DIR / "datasets" / "processed" / "test"  / "test_pyg.pkl"
WEIGHTS_PATH = RESEARCH_DIR / "datasets" / "processed" / "class_weights.pkl"

CKPT_DIR = RESEARCH_DIR / "models" / "hybrid"

# Pre-trained branch checkpoints
CNN_T_CKPT = RESEARCH_DIR / "models" / "cnn_transformer" / "best_model.pt"
RGNN_CKPT  = RESEARCH_DIR / "models" / "r_gnn" / "best_model.pt"

# Branch output dimensions (must match sub-model configs)
CNN_T_DIM   = 256    # CNN-Transformer output (D_MODEL)
RGNN_DIM    = 128    # R-GNN graph-level feature (GRU_HIDDEN)

# Multi-scale DenseNet tap dimensions
F_EARLY_DIM = 208    # DenseBlock1: 64 + 6*24
F_MID_DIM   = 392    # DenseBlock2: 104 + 12*24

# Fusion
FUSED_DIM       = 192     # cross-attention dimension
FUSION_HEADS    = 4
FUSION_DROPOUT  = 0.1

# Output heads
N_CLASSES_DETECTION = 2
N_CLASSES_TYPE      = 6
N_CLASSES_PHASE     = 8
# N_CLASSES_LOCATION: inferred at runtime

# Training — Phase 3: 3-phase curriculum
BATCH_SIZE    = 16
WEIGHT_DECAY  = 1e-4

# Phase 3a: Freeze branches, train fusion + heads (5 epochs warm-up)
PHASE3A_EPOCHS = 5
PHASE3A_LR     = 5e-4

# Phase 3b: Unfreeze all, end-to-end fine-tuning (40 epochs)
PHASE3B_EPOCHS = 40
PHASE3B_LR_BRANCHES = 1e-4   # lower LR for pre-trained branches
PHASE3B_LR_FUSION   = 5e-4   # higher LR for fusion

# CosineAnnealingWarmRestarts
COSINE_T0    = 10
COSINE_TMULT = 2

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Early stopping
PATIENCE = 12

# Legacy compatibility
EPOCHS        = PHASE3A_EPOCHS + PHASE3B_EPOCHS
LEARNING_RATE = PHASE3A_LR
LAMBDA_DETECT   = 1.0
LAMBDA_TYPE     = 1.5
LAMBDA_PHASE    = 0.5
LAMBDA_LOCATION = 1.0

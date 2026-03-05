"""
R-GNN Hyperparameters — v2 with upgraded hidden dimensions (128).
IT22577924 — Karunanayake K.P.A.W.
"""
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent.parent

# Data paths (PyG pkl files)
TRAIN_PKL = RESEARCH_DIR / "datasets" / "processed" / "train" / "train_pyg.pkl"
VAL_PKL   = RESEARCH_DIR / "datasets" / "processed" / "val"   / "val_pyg.pkl"
TEST_PKL  = RESEARCH_DIR / "datasets" / "processed" / "test"  / "test_pyg.pkl"
WEIGHTS_PATH = RESEARCH_DIR / "datasets" / "processed" / "class_weights.pkl"

CKPT_DIR = RESEARCH_DIR / "models" / "r_gnn"

# Graph sequence shape (must match Stage 01)
T_STEPS      = 20
NODE_FEAT    = 6           # V_A_mag, V_A_ang, V_B_mag, V_B_ang, V_C_mag, V_C_ang
EDGE_FEAT    = 3           # R_pu, X_pu, rated_kV

# GCN layers — upgraded from 64 to 128
GCN_HIDDEN   = 128
GCN_LAYERS   = 2
GCN_DROPOUT  = 0.1

# GRU recurrence — upgraded from 64 to 128
GRU_HIDDEN   = 128

# Output (multi-task) — same label set as CNN-T
N_CLASSES_DETECTION = 2
N_CLASSES_TYPE      = 6
N_CLASSES_PHASE     = 8
# N_CLASSES_LOCATION: inferred at runtime (N_buses + 1)

# Training — Phase 2 transfer learning
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 20           # Phase 2a: detection only (20 epochs)
PATIENCE      = 10

# Phase 2a: detection only
LAMBDA_DETECT   = 1.0
LAMBDA_TYPE     = 0.0
LAMBDA_PHASE    = 0.0
LAMBDA_LOCATION = 0.0

# Phase 2b (transfer learning): these are used when --phase 2b is passed
PHASE2B_EPOCHS = 25
PHASE2B_LR_HEADS = 5e-4      # new heads lr (15 epochs)
PHASE2B_LR_FINETUNE = 1e-4   # full fine-tune lr (10 epochs)

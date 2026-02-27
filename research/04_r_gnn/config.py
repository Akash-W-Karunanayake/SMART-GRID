"""
R-GNN Hyperparameters.
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

# GCN layers
GCN_HIDDEN   = 64
GCN_LAYERS   = 2
GCN_DROPOUT  = 0.1

# GRU recurrence
GRU_HIDDEN   = 64

# Output (multi-task) — same label set as CNN-T
N_CLASSES_DETECTION = 2
N_CLASSES_TYPE      = 6
N_CLASSES_PHASE     = 8
# N_CLASSES_LOCATION: inferred at runtime (N_buses + 1)

# Training
BATCH_SIZE    = 16          # smaller for graph batches
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
PATIENCE      = 10

LAMBDA_DETECT   = 1.0
LAMBDA_TYPE     = 1.0
LAMBDA_PHASE    = 0.5
LAMBDA_LOCATION = 1.5

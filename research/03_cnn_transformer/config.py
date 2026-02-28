"""
CNN-Transformer Hyperparameters — v2 with DenseNet-121-1D backbone.
IT22577924 — Karunanayake K.P.A.W.
"""
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent.parent

# Data paths
TRAIN_NPZ = RESEARCH_DIR / "datasets" / "processed" / "train" / "train.npz"
VAL_NPZ   = RESEARCH_DIR / "datasets" / "processed" / "val"   / "val.npz"
TEST_NPZ  = RESEARCH_DIR / "datasets" / "processed" / "test"  / "test.npz"
WEIGHTS_PATH = RESEARCH_DIR / "datasets" / "processed" / "class_weights.pkl"

# Checkpoint dir
CKPT_DIR = RESEARCH_DIR / "models" / "cnn_transformer"

# Sequence shape (must match Stage 01 output)
T_STEPS      = 20           # timesteps
N_BRANCHES   = None         # inferred from data at runtime
INPUT_FEAT   = 6            # I_A_mag, I_A_ang, I_B_mag, I_B_ang, I_C_mag, I_C_ang

# DenseNet-121-1D backbone
DENSE_GROWTH_RATE  = 24
DENSE_BLOCK_LAYERS = [6, 12, 24, 16]  # DenseNet-121 configuration
DENSE_DROP_RATE    = 0.1

# Transformer encoder
D_MODEL       = 256
N_HEAD        = 4
N_LAYERS      = 4
DIM_FF        = 512
TRANS_DROPOUT = 0.1

# Output (multi-task)
N_CLASSES_DETECTION = 2
N_CLASSES_TYPE      = 6
N_CLASSES_PHASE     = 8
# N_CLASSES_LOCATION: inferred from data (N_buses + 1)

# Training — Phase 1 pre-training (detection + type only)
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 20           # Phase 1: 20 epochs
PATIENCE      = 10

# Phase 1 task weights (detection + type only; phase/location disabled)
LAMBDA_DETECT   = 1.0
LAMBDA_TYPE     = 1.0
LAMBDA_PHASE    = 0.0        # disabled in Phase 1
LAMBDA_LOCATION = 0.0        # disabled in Phase 1

# Device
DEVICE = "cuda"              # will fall back to cpu automatically

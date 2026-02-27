"""
CNN-Transformer Hyperparameters.
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
INPUT_FEAT   = 3            # I_A, I_B, I_C per branch

# CNN encoder
CNN_CHANNELS  = [64, 128, 128]
CNN_KERNEL    = 3
CNN_DROPOUT   = 0.1

# Transformer encoder
D_MODEL       = 128
N_HEAD        = 4
N_LAYERS      = 2
DIM_FF        = 256
TRANS_DROPOUT = 0.1

# Output (multi-task)
N_CLASSES_DETECTION = 2
N_CLASSES_TYPE      = 6
N_CLASSES_PHASE     = 8
# N_CLASSES_LOCATION: inferred from data (N_buses + 1)

# Training
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
PATIENCE      = 10           # Early stopping patience

# Multi-task loss weights
LAMBDA_DETECT   = 1.0
LAMBDA_TYPE     = 1.0
LAMBDA_PHASE    = 0.5
LAMBDA_LOCATION = 0.5

# Device
DEVICE = "cuda"              # will fall back to cpu automatically

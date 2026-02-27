"""
Hybrid CNN-Transformer + R-GNN Hyperparameters.
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

# Inherited from sub-models
CNN_T_DIM   = 128    # CNN-Transformer output feature dim
RGNN_DIM    = 64     # R-GNN graph-level feature dim
FUSED_DIM   = CNN_T_DIM + RGNN_DIM   # 192

# Fusion
FUSION_HEADS    = 4
FUSION_DROPOUT  = 0.1

# Output heads
N_CLASSES_DETECTION = 2
N_CLASSES_TYPE      = 6
N_CLASSES_PHASE     = 8
# N_CLASSES_LOCATION: inferred at runtime

# Training
BATCH_SIZE    = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 60
PATIENCE      = 12

LAMBDA_DETECT   = 1.0
LAMBDA_TYPE     = 1.5
LAMBDA_PHASE    = 0.5
LAMBDA_LOCATION = 1.0

"""
Dataset Generation Configuration
All tunable constants for the Chunnakam GSS fault simulation pipeline.
IT22577924 — Karunanayake K.P.A.W.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
RESEARCH_DIR = Path(__file__).resolve().parent.parent          # research/
PROJECT_ROOT = RESEARCH_DIR.parent                             # GRID_SIMULATION/
MASTER_DSS   = PROJECT_ROOT / "Master.dss"

DATASET_RAW_DIR   = RESEARCH_DIR / "datasets" / "raw"
DATASET_PROC_DIR  = RESEARCH_DIR / "datasets" / "processed"
GRAPH_DIR         = RESEARCH_DIR / "datasets" / "graph"
RESULTS_DIR       = RESEARCH_DIR / "results"

# ---------------------------------------------------------------------------
# Sri Lankan grid constants
# ---------------------------------------------------------------------------
GRID_FREQUENCY_HZ  = 50          # CEB standard
CYCLE_DURATION_S   = 1.0 / GRID_FREQUENCY_HZ   # 20 ms per cycle
BASE_KV_HV         = 132.0       # Transmission voltage (Kilinochchi link)
BASE_KV_MV         = 33.0        # Distribution voltage (Chunnakam feeders)
BASE_KV_LV         = 0.4         # LV consumer voltage
SCC_MVA            = 1500.0      # 3-phase short-circuit capacity (weak grid)

# ---------------------------------------------------------------------------
# Simulation time window
# ---------------------------------------------------------------------------
PRE_FAULT_CYCLES   = 5           # Steady-state capture before fault
FAULT_CYCLES       = 10          # Fault active window
POST_FAULT_CYCLES  = 5           # Post-clearing recovery
TOTAL_CYCLES       = PRE_FAULT_CYCLES + FAULT_CYCLES + POST_FAULT_CYCLES  # 20

# OpenDSS Dynamic step = 1 cycle
DYN_STEPSIZE_S     = CYCLE_DURATION_S   # 0.02 s

# ---------------------------------------------------------------------------
# Dataset size targets
# ---------------------------------------------------------------------------
TARGET_SAMPLES = {
    "normal": 8000,
    "LG":     7000,
    "LL":     5000,
    "LLG":    5000,
    "LLL":    5000,
    "HIF":   10000,
}
TOTAL_TARGET = sum(TARGET_SAMPLES.values())   # 40,000

# ---------------------------------------------------------------------------
# Fault parameters
# ---------------------------------------------------------------------------
FAULT_IMPEDANCES = {
    "normal": [0.0],
    "LG":     [0.1, 1.0, 5.0, 10.0],       # Ohms
    "LL":     [0.1, 1.0, 5.0, 10.0],
    "LLG":    [0.1, 1.0, 10.0],
    "LLL":    [0.01, 0.1, 1.0],
    "HIF":    [],                            # Variable — see hif_model.py
}

# Phase combinations per fault type
FAULT_PHASES = {
    "normal": ["none"],
    "LG":     ["A", "B", "C"],
    "LL":     ["AB", "BC", "CA"],
    "LLG":    ["ABG", "BCG", "CAG"],
    "LLL":    ["ABC"],
    "HIF":    ["A", "B", "C"],
}

# OpenDSS phase count for Fault element
PHASE_COUNT = {
    "A": 1, "B": 1, "C": 1,
    "AB": 2, "BC": 2, "CA": 2,
    "ABG": 2, "BCG": 2, "CAG": 2,
    "ABC": 3,
    "none": 3,
}

# OpenDSS bus node suffix for each phase specification
PHASE_NODES = {
    "A":   ".1",
    "B":   ".2",
    "C":   ".3",
    "AB":  ".1.2",
    "BC":  ".2.3",
    "CA":  ".3.1",
    "ABG": ".1.2.4",
    "BCG": ".2.3.4",
    "CAG": ".3.1.4",
    "ABC": ".1.2.3",
    "none": "",
}

# Label maps (multi-task)
FAULT_TYPE_LABEL = {
    "normal": 0,
    "LG":     1,
    "LL":     2,
    "LLG":    3,
    "LLL":    4,
    "HIF":    5,
}
FAULT_PHASE_LABEL = {
    "none": 0,
    "A":    1, "B": 2, "C": 3,
    "AB":   4, "BC": 5, "CA": 6,
    "ABG":  4, "BCG": 5, "CAG": 6,   # same as LL phase labels
    "ABC":  7,
}

# ---------------------------------------------------------------------------
# Operating condition variations
# ---------------------------------------------------------------------------
LOAD_MULTIPLIERS  = [0.70, 0.85, 1.00, 1.15, 1.30]  # × rated
DER_PENETRATIONS  = [0.20, 0.50, 0.80, 1.00]          # fraction of rated PV/Wind
START_TIMES_HOUR  = {
    "morning_peak":   7.0,
    "noon_trough":   12.0,
    "evening_peak":  20.0,
}

# ---------------------------------------------------------------------------
# HIF (Emanuel arc model) parameters
# ---------------------------------------------------------------------------
HIF_R_MIN  = 100.0    # Ohms — minimum arc resistance
HIF_R_MAX  = 500.0    # Ohms — maximum arc resistance
# HIF buses: inject on ~30% of total buses (those on MV feeders)
HIF_BUS_FRACTION = 0.30

# ---------------------------------------------------------------------------
# OpenDSS snapshot/dynamic solver settings
# ---------------------------------------------------------------------------
SNAPSHOT_MAXITER   = 300
SNAPSHOT_TOLERANCE = 0.0001
DYNAMIC_MAXITER    = 100

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

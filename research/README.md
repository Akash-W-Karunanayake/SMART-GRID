# Hybrid Deep Learning Research Pipeline for Fault Diagnostics
## Project: IT22577924 — Chunnakam GSS Fault Dataset & Model Pipeline
### Researcher: Karunanayake K.P.A.W.

---

## Overview

This research develops a novel **Hybrid CNN-Transformer + Recurrent GNN (R-GNN)** model for
power system fault diagnostics at Chunnakam Grid Substation (Jaffna, Northern Sri Lanka).

The pipeline generates a physics-valid fault dataset from the existing Chunnakam OpenDSS model,
then trains and evaluates the proposed hybrid architecture against baseline methods.

---

## Pipeline Stages

| Stage | Directory | Purpose |
|-------|-----------|---------|
| 01 | `01_dataset_generation/` | OpenDSS co-simulation → labeled NPZ dataset |
| 02 | `02_data_preprocessing/` | Physics validation, normalization, SMOTE, splits |
| 03 | `03_cnn_transformer/` | 1D-CNN + Transformer Encoder (temporal branch) |
| 04 | `04_r_gnn/` | GCN + GRU recurrent message passing (spatial branch) |
| 05 | `05_hybrid_model/` | Integrated CNN-T + R-GNN with attention fusion |
| 06 | `06_model_evaluation/` | Metrics, benchmarks, plots, comparison report |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r research/requirements.txt

# 2. Generate dataset (from GRID_SIMULATION root)
python research/01_dataset_generation/run_generation.py --samples 15000

# 3. Preprocess
python research/02_data_preprocessing/run_preprocessing.py

# 4. Train CNN-Transformer
python research/03_cnn_transformer/train.py

# 5. Train R-GNN
python research/04_r_gnn/train.py

# 6. Train Hybrid Model
python research/05_hybrid_model/train.py

# 7. Evaluate and generate report
python research/06_model_evaluation/benchmark.py
```

---

## Grid Alignment (Sri Lankan Grid)

- **Frequency**: 50 Hz (CEB standard)
- **Feeder voltage**: 33 kV (Chunnakam GSS distribution)
- **Source SCC**: 1500 MVA (weak grid — Kilinochchi–Chunnakam 132 kV link)
- **DER**: 47.32 MW rooftop solar + 20 MW wind + 24 MW UJPS thermal
- **Feeders**: F06–F12 (Jaffna peninsula loads)
- **Simulation**: OpenDSS Dynamic mode, 20 ms (1-cycle) timesteps

---

## Dataset Summary

| Fault Type | Samples | Notes |
|------------|---------|-------|
| Normal | 4,000 | Varied load/DER levels |
| LG (Line-to-Ground) | 3,000 | All buses, Rf = 0.1–10 Ω |
| LL (Line-to-Line) | 2,000 | All buses, Rf = 0.1–10 Ω |
| LLG | 2,000 | All buses, Rf = 0.1–10 Ω |
| LLL (3-phase bolted) | 1,500 | All buses |
| HIF (Emanuel arc) | 2,500 | 20–40% buses, variable R |
| **Total** | **~15,000** | Multi-task labels |

---

## Labels (Multi-task)

Each sample carries 4 labels:
- `label_detection`: 0=normal, 1=fault
- `label_type`: 0=normal, 1=LG, 2=LL, 3=LLG, 4=LLL, 5=HIF
- `label_phase`: 0=none, 1=A, 2=B, 3=C, 4=AB, 5=BC, 6=CA, 7=ABC
- `label_location`: bus index (int), -1 for normal

---

## Directory Structure

```
research/
  01_dataset_generation/   ← Simulation engine + fault injection
  02_data_preprocessing/   ← Validation, normalization, splits
  03_cnn_transformer/      ← Temporal branch model
  04_r_gnn/                ← Spatial branch model
  05_hybrid_model/         ← Fusion of both branches
  06_model_evaluation/     ← Metrics, benchmarks, reports
  datasets/
    raw/                   ← NPZ files from simulation
    processed/train/val/test/
    graph/                 ← grid topology pickle
  models/                  ← Saved .pt checkpoints
  results/metrics/plots/reports/

---------------------------------------------------------------------------------- NEW FLOW 
● Execution Workflow                                                                                                         
  Stage 1: Generate Dataset (40K samples)                                                                                    
  # Fix HIF + current angles are already in code                                                                          
  python research/01_dataset_generation/run_generation.py --samples 40000

  Stage 2: Preprocess

  python research/02_data_preprocessing/run_preprocessing.py
  Uses physics-aware augmentation (not SMOTE), normalizes 6-feature currents.

  Stage 3: Pre-train CNN-Transformer (Phase 1)

  python research/03_cnn_transformer/train.py --epochs 20
  Detection + type only (phase/location lambdas = 0).

  Stage 4: Pre-train R-GNN (Phase 2)

  # Phase 2a: detection only
  python research/04_r_gnn/train.py --phase 2a --epochs 20

  # Phase 2b: transfer learning — freeze backbone, train all heads, then fine-tune
  python research/04_r_gnn/train.py --phase 2b

  Stage 5: Train Hybrid BHAF (Phase 3)

  python research/05_hybrid_model/train.py
  Loads pre-trained branches automatically from models/cnn_transformer/best_model.pt and models/r_gnn/best_model.pt. Runs:
  - 3a: Frozen branches, train fusion+heads (5 epochs)
  - 3b: Unfreeze all, differential LR (40 epochs, cosine restarts)

  Stage 6: Benchmark

  python research/06_model_evaluation/benchmark.py
  Compares SVM → CNN-T v2 → R-GNN v2 → Hybrid BHAF v2 across all 4 tasks.

  ---
  Key dependency chain: Stage 1 → 2 → 3 & 4 (parallel) → 5 → 6

  Stages 3 and 4 can run independently since they use different data formats (NPZ vs PyG pkl). Stage 5 requires both      
  branch checkpoints.
```

# Self-Healing Grid Backend (v0.2)

**Component 4 — IT22053350**
Fault isolation (rule-based) + MARL restoration (5 tie-switch agents)
for the Chunnakam GSS OpenDSS model.

## What Changed from v0.1

| Issue | v0.1 (broken) | v0.2 (fixed) |
|-------|---------------|--------------|
| Action space | 21 switches (all agents) | **5 tie switches only** |
| Bus name casing | Mixed (caused lookup bugs) | **All lowercase everywhere** |
| GNN edge_index | Casing mismatch → empty graph | **Consistent lowercase** |
| Isolation zone detection | Name parsing heuristic | **Graph-based reachability** |
| Reward function | No switching cost | **Penalty per switch toggled** |
| Dataset generation | Post-restoration state in input (leakage) | **Input = post-isolation only** |
| Old script compatibility | N/A | Old script was isolation-only, not restoration |

## Key Design Decision

```
Isolation: CBs + Sectionalizers  →  RULE-BASED  (not RL)
Restoration: Tie Switches only   →  MARL AGENTS (5 agents × Discrete(2))
```

This is correct because:
- Isolation is safety-critical and well-understood → rules are appropriate
- Restoration is the optimization problem → RL adds value
- Mixing all 21 switches as agents made the action space needlessly large (2^21 = 2M joint actions vs 2^5 = 32)

## Project Structure

```
self-healing-backend/
├── config/
│   ├── settings.py          # Env vars, paths
│   └── grid_config.py       # 21 switches, 22 loads, critical loads, ALL LOWERCASE
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   ├── routes/          # /isolate, /restore, /grid-state, /switch/{name}
│   │   └── schemas/         # FaultReport, IsolationResult, RestorationResult
│   ├── services/
│   │   ├── opendss_engine.py    # OpenDSS wrapper (all buses lowercase)
│   │   ├── grid_graph.py        # NetworkX graph (graph-based zone detection)
│   │   ├── isolation_service.py # Rule-based: opens CB + SEC
│   │   └── restoration_service.py # MARL or heuristic: closes tie switches
│   └── ml/
│       ├── environment/grid_env.py   # Gym env (5 tie-switch agents)
│       ├── agents/marl_agent.py      # Multi-Agent DQN (shared Q-network)
│       ├── models/gnn_encoder.py     # GCN topology encoder
│       └── reward.py                 # Multi-objective reward
├── scripts/
│   ├── generate_dataset.py  # Leakage-free dataset generation
│   ├── train_marl.py        # Online RL training (no pre-generated dataset)
│   └── evaluate.py          # Test on held-out fault buses
├── tests/                   # pytest tests for invariants
├── data/                    # Generated datasets
├── models/                  # Saved checkpoints
└── requirements.txt
```

## Setup

```bash
cd self-healing-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set DSS_MODEL_DIR to your SMART-GRID repo root
```

## Run the API

```bash
uvicorn app.main:app --reload --port 8100
# Open http://localhost:8100/docs
```

## API Endpoints

All under `/api/v1/self-healing/`:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/load-model` | Load/reload OpenDSS model |
| POST | `/isolate` | Rule-based fault isolation |
| POST | `/restore?strategy=auto` | Full FLISR (isolate + restore) |
| GET | `/grid-state` | Full grid snapshot |
| POST | `/switch/{name}?action=open` | Manual switch control |
| GET | `/switches` | All 21 switch states |

### Example: Full FLISR Request

```json
POST /api/v1/self-healing/restore?strategy=auto

{
  "fault_type": "lg",
  "fault_location": "F09_Node3",
  "fault_resistance_ohms": 0.01,
  "timestamp_step": 42
}
```

## Training Pipeline

### Step 1: Generate dataset (for EDA and analysis, NOT for RL training)

```bash
python -m scripts.generate_dataset --num-scenarios 500
```

This is leakage-free: input = post-isolation state, label = tie switch actions.

### Step 2: Train MARL agent (online RL — no pre-generated dataset needed)

```bash
python -m scripts.train_marl --episodes 2000 --max-steps 10
```

### Step 3: Evaluate on held-out scenarios

```bash
python -m scripts.evaluate --model models/marl_checkpoint.pt --scenarios 100
```

## How Data Leakage is Prevented

1. **RL training is online**: Agent interacts with the environment step-by-step. It only sees the *current* state before acting. No future state is ever in the observation.

2. **Dataset generation (if used for supervised pre-training)**: Input features are captured AFTER isolation but BEFORE restoration. The restoration result is the label, never the input.

3. **Train/test split**: Evaluation uses *different fault bus locations* than training, so the agent can't memorise specific scenarios.

## Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Load restored | +1.0 | Per load re-energized |
| Hospital bonus | +5.0 | Extra for F11_Hospital |
| Voltage violation | -2.0 | Per bus outside [0.94, 1.06] pu |
| Line overload | -3.0 | Per line exceeding NormalAmps |
| Loop penalty | -10.0 | If topology has cycles |
| Non-convergence | -20.0 | Power flow didn't converge |
| Switching cost | -0.5 | Per tie switch toggled |
| Step penalty | -0.1 | Per timestep (encourages speed) |

## Enforced Operational Constraints

- **Voltage bounds**: 0.94 – 1.06 pu (CEB Sri Lankan standard)
- **Line ampacity**: NormalAmps from OpenDSS model
- **Radial topology**: No loops allowed (validated via NetworkX cycle detection)
- **Convergence**: Power flow must converge
- **Switching cost**: Penalised to prevent unnecessary operations

## Chunnakam Grid Reference

| Item | Count | Details |
|------|-------|---------|
| Feeders | 8 | F05–F12 |
| Circuit Breakers | 8 | CB_F05–CB_F12 (isolation, normally closed) |
| Sectionalizers | 8 | SEC_F05–SEC_F12 (isolation, normally closed) |
| **Tie Switches** | **5** | **4 inter-feeder + 1 bus coupler (MARL agents, normally open)** |
| Loads | 22 | Residential, commercial, industrial, hospital |
| Critical | 1 | F11_Hospital (Jaffna Teaching Hospital) |
| Generators | 4 | 3× UJPS 8MW thermal + 1× 20MW wind |

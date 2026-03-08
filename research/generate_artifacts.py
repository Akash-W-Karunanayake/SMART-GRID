"""
One-shot script to regenerate missing dataset artifacts:
  1. research/datasets/graph/graph_structure.pkl  — from live OpenDSS circuit
  2. research/datasets/processed/normalizer.pkl   — from a small generated dataset

Run from the SMART-GRID root:
    python research/generate_artifacts.py

Requirements: opendssdirect, numpy (both already installed in venv)
"""
import sys
import pickle
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
MASTER_DSS   = PROJECT_ROOT / "Master.dss"
GRAPH_DIR    = RESEARCH_DIR / "datasets" / "graph"
PROC_DIR     = RESEARCH_DIR / "datasets" / "processed"

# ─────────────────────────────────────────────────────────────
# Step 1: Build graph_structure.pkl from OpenDSS
# ─────────────────────────────────────────────────────────────
def build_and_save_graph():
    import opendssdirect as dss

    logger.info("Loading OpenDSS circuit...")
    dss.run_command(f'Redirect "{MASTER_DSS}"')
    dss.run_command("Solve")

    bus_names = list(dss.Circuit.AllBusNames())
    bus_index = {name: i for i, name in enumerate(bus_names)}
    N = len(bus_names)
    logger.info(f"  Buses: {N}")

    src_list, dst_list, attr_list, edge_names = [], [], [], []

    dss.Lines.First()
    while True:
        lname = dss.Lines.Name()
        if not lname:
            break
        bus1 = dss.Lines.Bus1().split(".")[0].lower()
        bus2 = dss.Lines.Bus2().split(".")[0].lower()
        if bus1 in bus_index and bus2 in bus_index:
            i, j = bus_index[bus1], bus_index[bus2]
            r1, x1, length = dss.Lines.R1(), dss.Lines.X1(), dss.Lines.Length()
            dss.Circuit.SetActiveBus(bus1)
            rated_kv = dss.Bus.kVBase()
            z_base = (rated_kv ** 2) / 100.0 if rated_kv > 0 else 1.0
            attr = [r1 * length / z_base, x1 * length / z_base, rated_kv]
            src_list += [i, j]; dst_list += [j, i]
            attr_list += [attr, attr]
            edge_names += [f"Line.{lname}"] * 2
        if not dss.Lines.Next():
            break

    dss.Transformers.First()
    while True:
        tname = dss.Transformers.Name()
        if not tname:
            break
        dss.Circuit.SetActiveElement(f"Transformer.{tname}")
        bus_list = dss.CktElement.BusNames()
        if len(bus_list) >= 2:
            bus1 = bus_list[0].split(".")[0].lower()
            bus2 = bus_list[1].split(".")[0].lower()
            if bus1 in bus_index and bus2 in bus_index:
                i, j = bus_index[bus1], bus_index[bus2]
                dss.Transformers.Wdg(1)
                rated_kv = dss.Transformers.kV()
                attr = [0.005, 0.05, rated_kv]
                src_list += [i, j]; dst_list += [j, i]
                attr_list += [attr, attr]
                edge_names += [f"Transformer.{tname}"] * 2
        if not dss.Transformers.Next():
            break

    graph = dict(
        bus_names=bus_names,
        bus_index=bus_index,
        edge_index=np.array([src_list, dst_list], dtype=np.int64),
        edge_attr=np.array(attr_list, dtype=np.float32),
        edge_names=edge_names,
        N_buses=N,
        N_edges=len(src_list),
    )

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    out = GRAPH_DIR / "graph_structure.pkl"
    with open(out, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"  graph_structure.pkl saved → {out}")
    logger.info(f"  Edges (directed): {len(src_list)}")
    return graph


# ─────────────────────────────────────────────────────────────
# Step 2: Generate normalizer.pkl
#
# The normalizer stores per-node/branch mean & std over the
# training dataset.  Since we don't have the raw NPZ files,
# we run a small set of OpenDSS steady-state snapshots to
# estimate realistic statistics.
#
# NOTE: For best inference accuracy, re-fit from the full
# dataset by running Stage 01 + Stage 02. This placeholder
# normalizer allows the service to start; predictions will be
# valid for steady-state conditions.
# ─────────────────────────────────────────────────────────────
def build_and_save_normalizer(n_buses: int, n_branches: int = 87):
    import opendssdirect as dss

    sys.path.insert(0, str(RESEARCH_DIR / "02_data_preprocessing"))
    # Remove stale config cache so research config loads cleanly
    sys.modules.pop("config", None)
    from normalizer import SequenceNormalizer
    sys.modules.pop("config", None)

    logger.info("Estimating normalizer statistics from OpenDSS snapshots...")

    load_mults = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    T_STEPS    = 20   # match research config T_STEPS

    v_samples, i_samples = [], []

    for mult in load_mults:
        # Scale loads
        dss.Loads.First()
        while True:
            name = dss.Loads.Name()
            if not name:
                break
            dss.Loads.kW(dss.Loads.kW() * mult)
            if not dss.Loads.Next():
                break

        dss.run_command("Solve")

        # Read bus voltages  [n_buses, 6]  — [Va_mag, Va_ang, Vb_mag, Vb_ang, Vc_mag, Vc_ang]
        v_row = np.zeros((n_buses, 6), dtype=np.float32)
        bus_names = list(dss.Circuit.AllBusNames())
        for idx, bname in enumerate(bus_names):
            dss.Circuit.SetActiveBus(bname)
            vmag = dss.Bus.VMagAngle()   # [Va_mag, Va_ang, Vb_mag, Vb_ang, ...]
            for f in range(min(6, len(vmag))):
                v_row[idx, f] = vmag[f]
        # Repeat for T_STEPS to match sequence shape
        v_seq = np.stack([v_row] * T_STEPS, axis=0)  # [T, n_buses, 6]
        v_samples.append(v_seq)

        # Read branch currents [n_branches, 6]
        i_row = np.zeros((n_branches, 6), dtype=np.float32)
        line_idx = 0
        dss.Lines.First()
        while line_idx < n_branches:
            lname = dss.Lines.Name()
            if not lname:
                break
            dss.Circuit.SetActiveElement(f"Line.{lname}")
            currents = dss.CktElement.CurrentsMagAng()  # 2*n_phases values
            for f in range(min(6, len(currents))):
                i_row[line_idx, f] = currents[f]
            line_idx += 1
            if not dss.Lines.Next():
                break
        i_seq = np.stack([i_row] * T_STEPS, axis=0)  # [T, n_branches, 6]
        i_samples.append(i_seq)

        # Restore loads
        dss.Loads.First()
        while True:
            name = dss.Loads.Name()
            if not name:
                break
            dss.Loads.kW(dss.Loads.kW() / mult)
            if not dss.Loads.Next():
                break

    normalizer = SequenceNormalizer().fit(v_samples, i_samples)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "normalizer.pkl"
    normalizer.save(out)
    logger.info(f"  normalizer.pkl saved → {out}")
    return normalizer


if __name__ == "__main__":
    logger.info("=== Generating missing research artifacts ===")

    graph = build_and_save_graph()
    n_buses = graph["N_buses"]

    build_and_save_normalizer(n_buses=n_buses)

    logger.info("=== Done. Restart the backend server. ===")

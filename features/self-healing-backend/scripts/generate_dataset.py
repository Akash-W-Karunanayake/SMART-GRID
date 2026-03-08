"""
generate_dataset.py
====================
Generates a labeled dataset of fault scenarios for analysis and
optional imitation pre-training.

ANTI-LEAKAGE DESIGN:
  - INPUT features  = grid state AFTER isolation, BEFORE restoration
  - LABEL           = which tie switches the heuristic closed
  - Post-restoration state is NEVER included in input features

Output: data/restoration_dataset.csv

Usage:
  python -m scripts.generate_dataset --num-scenarios 500
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from config import settings, TIE_SWITCH_NAMES, ALL_LOAD_NAMES, CRITICAL_LOADS
from app.services.opendss_engine import OpenDSSEngine
from app.services.isolation_service import isolate_fault
from app.services.restoration_service import restore_service
from app.api.schemas.fault_input import FaultReport


FAULT_BUSES = [
    "F06_Node2", "F06_Node3", "F06_Node4",
    "F07_Node2", "F07_Node3", "F07_Node4",
    "F08_Node2", "F08_Node3", "F08_Node4",
    "F09_Node2", "F09_Node3", "F09_Node4", "F09_Node5",
    "F10_Node2", "F10_Node3", "F10_Node4",
    "F11_Node2", "F11_Node3", "F11_Node4", "F11_Node5",
    "F12_Node2", "F12_Node3",
]

FAULT_TYPES = ["lg", "ll", "3phase", "llg"]


def generate_one(eng: OpenDSSEngine, sid: int) -> dict:
    fault_bus = random.choice(FAULT_BUSES)
    fault_type = random.choice(FAULT_TYPES)
    timestep = random.randint(0, 95)

    eng.load_model()
    eng.solve_at_step(timestep)
    pre_load_kw = eng.get_total_load_kw()

    report = FaultReport(fault_type=fault_type, fault_location=fault_bus,
                         timestamp_step=timestep)
    isolation = isolate_fault(report)
    if not isolation.success:
        return None

    # ── SNAPSHOT: post-isolation, pre-restoration (this is the INPUT) ──
    eng.solve()
    post_iso_state = eng.get_grid_state()
    post_iso_tie_states = {k: v for k, v in post_iso_state["tie_switch_states"].items()}
    post_iso_de_energized = list(post_iso_state["de_energized_loads"])
    post_iso_voltages = {k: round(v, 4) for k, v in post_iso_state["bus_voltages_pu"].items()}

    # ── RUN RESTORATION (this produces the LABEL) ──
    result = restore_service(isolation, strategy="heuristic")

    # Label = which tie switches were closed during restoration
    tie_actions = {}
    for step in result.restoration_steps:
        if step.switch_name in TIE_SWITCH_NAMES:
            tie_actions[step.switch_name] = step.action

    return {
        "scenario_id": sid,
        "fault_bus": fault_bus,
        "fault_type": fault_type,
        "timestep": timestep,
        "feeder": isolation.feeder,
        "pre_fault_load_kw": round(pre_load_kw, 2),
        # INPUT: post-isolation state
        "post_iso_tie_states": json.dumps(post_iso_tie_states),
        "post_iso_de_energized_loads": json.dumps(post_iso_de_energized),
        "post_iso_bus_voltages": json.dumps(post_iso_voltages),
        "num_de_energized_after_iso": isolation.num_de_energized_loads,
        "critical_loads_affected": json.dumps(isolation.critical_loads_affected),
        # LABEL: restoration actions
        "tie_actions_taken": json.dumps(tie_actions),
        "num_ties_closed": len(tie_actions),
        # OUTCOME: for evaluation only (NOT input to model)
        "loads_restored": json.dumps(result.loads_restored),
        "loads_still_offline": json.dumps(result.loads_still_offline),
        "pct_load_restored": result.pct_load_restored,
        "voltage_violations_after": json.dumps(result.voltage_violations),
        "is_radial_after": result.is_radial,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenarios", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/restoration_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    eng = OpenDSSEngine()
    if not eng.load_model():
        logger.error("Failed to load model")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(args.num_scenarios):
        if (i + 1) % 50 == 0:
            logger.info(f"Scenario {i+1}/{args.num_scenarios}")
        row = generate_one(eng, i)
        if row:
            rows.append(row)

    if rows:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Saved {len(rows)} scenarios → {args.output}")


if __name__ == "__main__":
    main()

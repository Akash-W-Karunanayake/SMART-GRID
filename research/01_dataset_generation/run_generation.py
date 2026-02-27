"""
CLI entry point for Stage 01 — Dataset Generation.
Generates labeled NPZ fault samples using the OpenDSS Dynamic simulation engine.

Usage (from GRID_SIMULATION root):
    python research/01_dataset_generation/run_generation.py
    python research/01_dataset_generation/run_generation.py --samples 100 --dry-run
    python research/01_dataset_generation/run_generation.py --samples 15000 --workers 1

IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import logging
import argparse
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make stage-local imports work
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASET_RAW_DIR, TOTAL_TARGET
from sim_engine import SimEngine
from scenario_generator import generate_scenarios

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_generation")


def parse_args():
    p = argparse.ArgumentParser(description="Generate Chunnakam GSS fault dataset")
    p.add_argument("--samples", type=int, default=TOTAL_TARGET,
                   help=f"Total samples to generate (default {TOTAL_TARGET})")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate small batch of 100 samples to verify pipeline")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-generated NPZ files")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory (default: research/datasets/raw/)")
    return p.parse_args()


def save_sample(sample: dict, path: Path) -> None:
    """Save one simulation sample to NPZ."""
    np.savez_compressed(
        path,
        voltage_seq   = sample["voltage_seq"],
        current_seq   = sample["current_seq"],
        # labels
        label_detection = np.array(sample["labels"]["label_detection"], dtype=np.int32),
        label_type      = np.array(sample["labels"]["label_type"],      dtype=np.int32),
        label_phase     = np.array(sample["labels"]["label_phase"],     dtype=np.int32),
        label_location  = np.array(sample["labels"]["label_location"],  dtype=np.int32),
        # graph
        edge_index      = sample["graph_snapshot"]["edge_index"],
        edge_attr       = sample["graph_snapshot"]["edge_attr"],
        # metadata stored as object array for flexibility
        fault_type      = np.array(sample["metadata"]["fault_type"]),
        phase           = np.array(sample["metadata"]["phase"]),
        bus             = np.array(sample["metadata"]["bus"]),
        resistance      = np.array(sample["metadata"]["resistance"],    dtype=np.float32),
        load_mult       = np.array(sample["metadata"]["load_mult"],     dtype=np.float32),
        der_fraction    = np.array(sample["metadata"]["der_fraction"],  dtype=np.float32),
        start_time_hr   = np.array(sample["metadata"]["start_time_hr"],dtype=np.float32),
    )


def main():
    args = parse_args()
    n_samples = 100 if args.dry_run else args.samples
    out_dir   = Path(args.output_dir) if args.output_dir else DATASET_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Stage 01: Dataset Generation ===")
    logger.info(f"Target samples : {n_samples}")
    logger.info(f"Output dir     : {out_dir}")
    logger.info(f"Dry run        : {args.dry_run}")

    # --- Initialise simulation engine ---
    engine = SimEngine()
    logger.info("Compiling OpenDSS model...")
    engine.compile()

    logger.info("Building/loading graph topology...")
    engine.build_graph_once()

    bus_names = engine.bus_names
    logger.info(f"Circuit: {engine.n_buses} buses, {engine.n_branches} branches")

    # --- Generate scenario list ---
    scenarios = generate_scenarios(bus_names, n_total=n_samples)

    # --- Simulation loop ---
    n_success = 0
    n_failed  = 0
    stats = {k: 0 for k in ["normal","LG","LL","LLG","LLL","HIF"]}

    for idx, scenario in enumerate(tqdm(scenarios, desc="Simulating")):
        npz_path = out_dir / f"sample_{idx:06d}.npz"

        if args.resume and npz_path.exists():
            n_success += 1
            continue

        try:
            sample = engine.run_sample(
                fault_type    = scenario["fault_type"],
                phase         = scenario["phase"],
                bus           = scenario["bus"],
                resistance    = scenario["resistance"],
                load_mult     = scenario["load_mult"],
                der_fraction  = scenario["der_fraction"],
                start_time_hr = scenario["start_time_hr"],
            )
            save_sample(sample, npz_path)
            n_success += 1
            stats[scenario["fault_type"]] += 1

        except Exception as e:
            n_failed += 1
            logger.error(f"Sample {idx} failed ({scenario['fault_type']} @ "
                         f"{scenario['bus']}): {e}")
            if args.dry_run:
                traceback.print_exc()

        # Progress report every 500 samples
        if (idx + 1) % 500 == 0:
            logger.info(f"Progress: {idx+1}/{n_samples} | "
                        f"Success={n_success}, Failed={n_failed}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info(f"Dataset generation complete")
    logger.info(f"  Total scenarios : {len(scenarios)}")
    logger.info(f"  Succeeded       : {n_success}")
    logger.info(f"  Failed          : {n_failed}")
    logger.info(f"  Breakdown: {stats}")
    logger.info(f"  Output dir: {out_dir}")

    if args.dry_run:
        # Verify one saved file
        first_npz = sorted(out_dir.glob("sample_*.npz"))
        if first_npz:
            data = np.load(first_npz[0], allow_pickle=True)
            logger.info(f"\nDry-run verification — {first_npz[0].name}:")
            logger.info(f"  voltage_seq shape : {data['voltage_seq'].shape}")
            logger.info(f"  current_seq shape : {data['current_seq'].shape}")
            logger.info(f"  label_type        : {int(data['label_type'])}")
            logger.info(f"  label_detection   : {int(data['label_detection'])}")
            logger.info(f"  fault_type        : {str(data['fault_type'])}")


if __name__ == "__main__":
    main()

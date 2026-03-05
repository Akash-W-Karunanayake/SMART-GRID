"""
CLI entry point for Stage 02 — Data Preprocessing.
1. Physics validation of raw NPZ files
2. Z-score normalization (fit on training set)
3. SMOTE for HIF class imbalance
4. Train/val/test split → NPZ bundles + PyG Data objects

Usage (from GRID_SIMULATION root):
    python research/02_data_preprocessing/run_preprocessing.py
    python research/02_data_preprocessing/run_preprocessing.py --skip-smote

IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import logging
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "01_dataset_generation"))

from validate_physics import run_validation
from normalizer import SequenceNormalizer
from augmentor import augment_dataset, compute_class_weights
from dataset_builder import (load_all_samples, split_indices,
                              save_numpy_split, save_pyg_split)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_preprocessing")

# Paths (relative to research/)
RESEARCH_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR       = RESEARCH_DIR / "datasets" / "raw"
PROC_DIR      = RESEARCH_DIR / "datasets" / "processed"
GRAPH_DIR     = RESEARCH_DIR / "datasets" / "graph"
RESULTS_DIR   = RESEARCH_DIR / "results"
REPORT_PATH   = RESULTS_DIR / "reports" / "physics_validation_report.csv"
NORM_PATH     = PROC_DIR / "normalizer.pkl"
WEIGHTS_PATH  = PROC_DIR / "class_weights.pkl"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 02: Data Preprocessing")
    p.add_argument("--skip-smote",    action="store_true", help="Skip SMOTE augmentation")
    p.add_argument("--skip-pyg",      action="store_true", help="Skip PyG Data object creation")
    p.add_argument("--raw-dir",       type=str, default=None, help="Override raw dataset dir")
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR

    logger.info("=== Stage 02: Data Preprocessing ===")

    # ------------------------------------------------------------------
    # Step 1: Physics validation
    # ------------------------------------------------------------------
    logger.info("Step 1: Physics validation...")
    valid_paths, flagged = run_validation(raw_dir, REPORT_PATH)
    logger.info(f"  Valid: {len(valid_paths)}, Flagged: {len(flagged)}")

    if len(valid_paths) == 0:
        logger.error("No valid samples found. Run Stage 01 first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Load all valid samples
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading valid samples...")
    samples = load_all_samples(valid_paths)
    N = len(samples["voltage_seqs"])
    logger.info(f"  Loaded {N} samples")

    # ------------------------------------------------------------------
    # Step 3: Train/val/test split
    # ------------------------------------------------------------------
    logger.info("Step 3: Splitting dataset...")
    idx_train, idx_val, idx_test = split_indices(N, samples["label_type"])

    # ------------------------------------------------------------------
    # Step 4: Augment RAW training data (before normalization)
    # ------------------------------------------------------------------
    if not args.skip_smote and len(idx_train) > 10:
        logger.info("Step 4: Physics-aware augmentation on raw training data...")

        train_v = [samples["voltage_seqs"][i] for i in idx_train]
        train_i = [samples["current_seqs"][i]  for i in idx_train]
        y_type  = samples["label_type"][idx_train]

        # Augment raw (unnormalized) data so transforms are physically meaningful
        aug_v, aug_c, aug_labels, src_indices = augment_dataset(
            train_v, train_i, y_type, random_state=42
        )

        n_orig = len(idx_train)
        n_new  = len(aug_labels) - n_orig
        logger.info(f"  Added {n_new} synthetic samples via physics-aware augmentation")

        # Copy ALL labels from source samples (not hardcoded values)
        l_det_aug = np.concatenate([
            samples["label_detection"][idx_train],
            samples["label_detection"][idx_train[src_indices]]
        ])
        l_type_aug  = aug_labels
        l_phase_aug = np.concatenate([
            samples["label_phase"][idx_train],
            samples["label_phase"][idx_train[src_indices]]
        ])
        l_loc_aug = np.concatenate([
            samples["label_location"][idx_train],
            samples["label_location"][idx_train[src_indices]]
        ])

        # ------------------------------------------------------------------
        # Step 5: Fit normalizer on augmented training set
        # ------------------------------------------------------------------
        logger.info("Step 5: Fitting normalizer on augmented training set...")
        normalizer = SequenceNormalizer().fit(aug_v, aug_c)
        normalizer.save(NORM_PATH)

        # Normalize augmented training data
        aug_v_norm = [normalizer.transform_v(v) for v in aug_v]
        aug_c_norm = [normalizer.transform_i(c) for c in aug_c]

        # Save augmented + normalized training split
        (PROC_DIR / "train").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            PROC_DIR / "train" / "train.npz",
            voltage_seq     = np.stack(aug_v_norm, axis=0),
            current_seq     = np.stack(aug_c_norm, axis=0),
            label_detection = l_det_aug,
            label_type      = l_type_aug,
            label_phase     = l_phase_aug,
            label_location  = l_loc_aug,
        )
        logger.info(f"  Augmented training set saved ({len(aug_labels)} samples)")
    else:
        logger.info("Step 4: Fitting normalizer on training set...")
        v_train = [samples["voltage_seqs"][i] for i in idx_train]
        i_train = [samples["current_seqs"][i]  for i in idx_train]
        normalizer = SequenceNormalizer().fit(v_train, i_train)
        normalizer.save(NORM_PATH)

        logger.info("Step 5: Saving training split without augmentation...")
        save_numpy_split(idx_train, samples, normalizer,
                         PROC_DIR / "train", "train")

    # ------------------------------------------------------------------
    # Step 6: Save val and test splits
    # ------------------------------------------------------------------
    logger.info("Step 6: Saving val/test splits...")
    save_numpy_split(idx_val,  samples, normalizer, PROC_DIR / "val",  "val")
    save_numpy_split(idx_test, samples, normalizer, PROC_DIR / "test", "test")

    # ------------------------------------------------------------------
    # Step 7: PyG Data objects (for R-GNN)
    # ------------------------------------------------------------------
    if not args.skip_pyg:
        logger.info("Step 7: Creating PyG Data objects...")
        save_pyg_split(idx_train, samples, normalizer, PROC_DIR / "train", "train")
        save_pyg_split(idx_val,   samples, normalizer, PROC_DIR / "val",   "val")
        save_pyg_split(idx_test,  samples, normalizer, PROC_DIR / "test",  "test")

    # ------------------------------------------------------------------
    # Step 8: Class weights for weighted CE loss
    # ------------------------------------------------------------------
    logger.info("Step 8: Computing class weights...")
    import pickle
    weights = compute_class_weights(samples["label_type"][idx_train])
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump(weights, f)
    logger.info(f"  Class weights saved → {WEIGHTS_PATH}")

    logger.info("=== Stage 02 complete ===")
    logger.info(f"  Processed splits in: {PROC_DIR}")
    logger.info(f"  Physics report:      {REPORT_PATH}")


if __name__ == "__main__":
    main()

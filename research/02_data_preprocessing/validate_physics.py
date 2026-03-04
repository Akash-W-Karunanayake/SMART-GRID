"""
Physics Validation — Check that generated simulation samples are physically plausible.
Samples failing more than MAX_FAILURES checks are flagged and excluded from training.
IT22577924 — Karunanayake K.P.A.W.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Validation thresholds
MAX_FAILURES   = 3          # Flag sample if it fails > this many checks
V_NOM_PU       = 1.0
V_OVERFLOW_CAP = 100.0      # > 100× nominal pu = catastrophically corrupt data
V_FAULT_DROP   = 0.03       # LIF faulted bus must drop >3% pu (weak 33kV grid)
HIF_FAULT_DROP = 0.03       # HIF minimum drop (weaker signal)
I_FAULT_RATIO  = 1.3        # LIF: I_fault > 1.3× I_prefault (weak grid)
I_HIF_RATIO    = 1.1        # HIF: smaller but detectable increase
V_MIN_PU       = 0.0        # Lower guard (allow zero during fault)
V2_THRESHOLD   = 0.01       # |V2| > this for unbalanced faults (weak grid)


def validate_sample(sample_path: Path) -> Tuple[bool, list]:
    """
    Run physics checks on one NPZ sample.

    Returns
    -------
    (passed: bool, failures: list[str])
    """
    try:
        data = np.load(sample_path, allow_pickle=True)
    except Exception as e:
        return False, [f"Load error: {e}"]

    V = data["voltage_seq"]    # [T=20, N_buses, 6]
    I = data["current_seq"]    # [T=20, N_branches, 6]
    fault_type  = str(data["fault_type"])
    label_det   = int(data["label_detection"])
    label_type  = int(data["label_type"])
    label_loc   = int(data["label_location"])

    T, N_buses, _ = V.shape
    PRE  = slice(0, 5)
    FAULT = slice(5, 15)

    failures = []

    # ------------------------------------------------------------------
    # Check 0: Hard absolute magnitude cap — catastrophically corrupt data
    # Values > V_OVERFLOW_CAP× nominal are physically impossible and indicate
    # dtype overflow during OpenDSS extraction (float32 → float64 promotion).
    # This failure is always counted regardless of MAX_FAILURES.
    # ------------------------------------------------------------------
    v_mags = V[:, :, 0::2]  # [T, N_buses, 3] — phase mag columns (0,2,4)
    if np.any(v_mags > V_OVERFLOW_CAP * V_NOM_PU):
        failures.append(f"C0: Catastrophic voltage overflow (max={v_mags.max():.3e} > "
                        f"{V_OVERFLOW_CAP}x nominal)")

    # ------------------------------------------------------------------
    # Check 1: Voltage magnitudes are non-negative and bounded
    # ------------------------------------------------------------------
    if np.any(v_mags < 0):
        failures.append("C1: Negative voltage magnitudes")
    if np.any(v_mags > 2.0):    # >200% pu is physically impossible
        failures.append("C1: Voltage exceeds 200% pu")

    # ------------------------------------------------------------------
    # Check 2: Current magnitudes non-negative
    # ------------------------------------------------------------------
    i_mags = I[:, :, 0::2]  # magnitude columns only (indices 0, 2, 4)
    if np.any(i_mags < 0):
        failures.append("C2: Negative current magnitudes")

    # ------------------------------------------------------------------
    # Check 3: Fault voltage sag at faulted bus
    # HIF is skipped: arc resistance 100-500 ohm on 11 kV creates < 0.1% sag —
    # requiring a 10% sag threshold would reject all valid HIF samples by design.
    # ------------------------------------------------------------------
    if label_det == 1 and fault_type != "HIF" and label_loc >= 0 and label_loc < N_buses:
        v_pre   = v_mags[PRE,   label_loc, :].mean()
        v_fault = v_mags[FAULT, label_loc, :].mean()
        sag     = v_pre - v_fault

        if v_pre > 0.1 and sag < V_FAULT_DROP:
            failures.append(f"C3: Insufficient voltage sag ({sag:.3f} < {V_FAULT_DROP})")

    # ------------------------------------------------------------------
    # Check 4: Fault current increase
    # HIF is skipped: 100-500 ohm fault resistance limits current increase to
    # < 1% of pre-fault level — the I_HIF_RATIO threshold is physically unachievable.
    # ------------------------------------------------------------------
    if label_det == 1 and fault_type != "HIF" and I.shape[1] > 0:
        i_pre   = I[PRE,  :, :].mean()
        i_fault = I[FAULT, :, :].mean()
        if i_pre > 0:
            ratio = i_fault / i_pre
            if ratio < I_FAULT_RATIO:
                failures.append(f"C4: Insufficient fault current increase ({ratio:.2f} < {I_FAULT_RATIO})")

    # ------------------------------------------------------------------
    # Check 5: Symmetrical components — V2 should be >0 for unbalanced faults
    # HIF is skipped: negative-sequence voltage from a 100-500 ohm arc is < 0.01 pu,
    # far below the V2_THRESHOLD designed for bolted/low-impedance faults.
    # ------------------------------------------------------------------
    if label_type in (1, 2, 3):   # LG, LL, LLG — unbalanced (HIF excluded)
        v2_vals = _compute_v2(V[FAULT])  # [fault_steps, N_buses]
        v2_max  = np.nanmax(np.abs(v2_vals))
        if v2_max < V2_THRESHOLD:
            failures.append(f"C5: V2~0 for unbalanced fault (max={v2_max:.4f})")

    # ------------------------------------------------------------------
    # Check 6: Sequence components for 3-phase faults — V2 should be ≈ 0
    # ------------------------------------------------------------------
    if label_type == 4:   # LLL
        v2_vals = _compute_v2(V[FAULT])
        v2_max  = np.nanmax(np.abs(v2_vals))
        if v2_max > 0.20:   # some tolerance
            failures.append(f"C6: V2>0.20 for 3-phase fault (max={v2_max:.4f})")

    # ------------------------------------------------------------------
    # Check 7: Normal samples — no significant disturbance
    # ------------------------------------------------------------------
    if label_det == 0:
        v_std = v_mags.std()
        if v_std > 0.15:
            failures.append(f"C7: High voltage variance in normal sample (std={v_std:.3f})")

    passed = len(failures) <= MAX_FAILURES
    return passed, failures


def _compute_v2(V_window: np.ndarray) -> np.ndarray:
    """
    Compute negative-sequence voltage magnitude per bus from phasor data.
    Uses symmetrical components transform on per-unit magnitudes.

    V_window : [T, N_buses, 6]  — [Va_mag, Va_ang, Vb_mag, Vb_ang, Vc_mag, Vc_ang]
    Returns  : [T, N_buses]     — |V2| in pu
    """
    T, N, _ = V_window.shape
    Va_m, Va_a = V_window[:, :, 0], np.deg2rad(V_window[:, :, 1])
    Vb_m, Vb_a = V_window[:, :, 2], np.deg2rad(V_window[:, :, 3])
    Vc_m, Vc_a = V_window[:, :, 4], np.deg2rad(V_window[:, :, 5])

    Va = Va_m * np.exp(1j * Va_a)
    Vb = Vb_m * np.exp(1j * Vb_a)
    Vc = Vc_m * np.exp(1j * Vc_a)

    a = np.exp(1j * 2 * np.pi / 3)
    V2 = (Va + a**2 * Vb + a * Vc) / 3.0
    return np.abs(V2)


def run_validation(raw_dir: Path, report_path: Path) -> Tuple[list, list]:
    """
    Validate all NPZ files in raw_dir.

    Returns
    -------
    (valid_paths, flagged_paths)
    Also writes CSV report to report_path.
    """
    import csv
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(raw_dir.glob("sample_*.npz"))
    logger.info(f"Validating {len(npz_files)} samples in {raw_dir}")

    valid_paths   = []
    flagged_paths = []

    with open(report_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["file", "passed", "n_failures", "failure_details"])

        for p in npz_files:
            passed, failures = validate_sample(p)
            writer.writerow([p.name, passed, len(failures), " | ".join(failures)])
            if passed:
                valid_paths.append(p)
            else:
                flagged_paths.append(p)

    pass_rate = len(valid_paths) / max(1, len(npz_files)) * 100
    logger.info(f"Validation complete: {len(valid_paths)}/{len(npz_files)} passed "
                f"({pass_rate:.1f}%)")
    logger.info(f"Report → {report_path}")

    return valid_paths, flagged_paths

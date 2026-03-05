"""
Scenario Generator — Enumerate all fault simulation scenarios for the dataset.
Produces a flat list of parameter dicts; the simulation loop in run_generation.py
iterates over it and calls SimEngine.run_sample() per scenario.
IT22577924 — Karunanayake K.P.A.W.
"""
import random
import logging
from itertools import product

from config import (
    TARGET_SAMPLES, FAULT_IMPEDANCES, FAULT_PHASES,
    LOAD_MULTIPLIERS, DER_PENETRATIONS, START_TIMES_HOUR,
    HIF_R_MIN, HIF_R_MAX, HIF_BUS_FRACTION,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def generate_scenarios(bus_names: list, n_total: int = None) -> list:
    """
    Build a reproducible list of scenario parameter dicts.

    Parameters
    ----------
    bus_names : list[str]  — all bus names from the compiled OpenDSS model
    n_total   : int        — override total sample count (for dry-run)

    Returns
    -------
    list[dict] with keys: fault_type, phase, bus, resistance,
                           load_mult, der_fraction, start_time_hr
    """
    rng = random.Random(RANDOM_SEED)

    # Identify MV buses (≥33 kV prefix or heuristic: name contains feeder label)
    # HIF is only injected on distribution (MV) buses
    mv_buses = [b for b in bus_names if _is_mv_bus(b)]
    if not mv_buses:
        mv_buses = bus_names   # fallback: all buses
    hif_bus_pool = rng.sample(mv_buses, max(1, int(len(mv_buses) * HIF_BUS_FRACTION)))

    start_times = list(START_TIMES_HOUR.values())
    scenarios = []

    # ---------------------------------------------------------------
    # Normal operation
    # ---------------------------------------------------------------
    n_normal = TARGET_SAMPLES["normal"] if n_total is None else max(1, int(n_total * 4000 / 15000))
    for _ in range(n_normal):
        scenarios.append({
            "fault_type":    "normal",
            "phase":         "none",
            "bus":           rng.choice(bus_names),
            "resistance":    0.0,
            "load_mult":     rng.choice(LOAD_MULTIPLIERS),
            "der_fraction":  rng.choice(DER_PENETRATIONS),
            "start_time_hr": rng.choice(start_times),
        })

    # ---------------------------------------------------------------
    # Fault scenarios (LG, LL, LLG, LLL)
    # ---------------------------------------------------------------
    for fault_type in ("LG", "LL", "LLG", "LLL"):
        target = TARGET_SAMPLES[fault_type] if n_total is None else max(1, int(n_total * TARGET_SAMPLES[fault_type] / 15000))
        phases = FAULT_PHASES[fault_type]
        impedances = FAULT_IMPEDANCES[fault_type]

        # Build base combinations
        base_combos = list(product(bus_names, phases, impedances))
        rng.shuffle(base_combos)

        # Sample with replacement if needed
        for idx in range(target):
            bus, phase, rf = base_combos[idx % len(base_combos)]
            scenarios.append({
                "fault_type":    fault_type,
                "phase":         phase,
                "bus":           bus,
                "resistance":    rf,
                "load_mult":     rng.choice(LOAD_MULTIPLIERS),
                "der_fraction":  rng.choice(DER_PENETRATIONS),
                "start_time_hr": rng.choice(start_times),
            })

    # ---------------------------------------------------------------
    # HIF scenarios (variable arc resistance — assigned at sim time)
    # ---------------------------------------------------------------
    n_hif = TARGET_SAMPLES["HIF"] if n_total is None else max(1, int(n_total * 2500 / 15000))
    hif_phases = FAULT_PHASES["HIF"]
    for _ in range(n_hif):
        scenarios.append({
            "fault_type":    "HIF",
            "phase":         rng.choice(hif_phases),
            "bus":           rng.choice(hif_bus_pool),
            "resistance":    rng.uniform(HIF_R_MIN, HIF_R_MAX),   # initial R; updates per step
            "load_mult":     rng.choice(LOAD_MULTIPLIERS),
            "der_fraction":  rng.choice(DER_PENETRATIONS),
            "start_time_hr": rng.choice(start_times),
        })

    # Shuffle all scenarios so NPZ files are written in mixed order
    rng.shuffle(scenarios)

    # Optional trim for dry-run
    if n_total is not None and n_total < len(scenarios):
        scenarios = scenarios[:n_total]

    logger.info(f"Generated {len(scenarios)} scenarios "
                f"({sum(1 for s in scenarios if s['fault_type']=='normal')} normal, "
                f"{sum(1 for s in scenarios if s['fault_type']!='normal')} fault)")
    return scenarios


def _is_mv_bus(bus_name: str) -> bool:
    """Heuristic: Chunnakam MV buses contain feeder labels (f06–f12) or 33kV markers."""
    b = bus_name.lower()
    feeder_tags = ["f06", "f07", "f08", "f09", "f10", "f11", "f12", "mv", "33kv"]
    return any(tag in b for tag in feeder_tags)

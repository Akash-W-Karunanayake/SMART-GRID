"""
Reward Function for Service Restoration (v3 -- DELTA-BASED)
============================================================
Multi-objective reward guiding the MARL agents (5 tie switches).

CRITICAL DESIGN PRINCIPLE:
  The reward measures the IMPACT of the agent's action, not pre-existing
  grid conditions. Uses delta-based metrics relative to the post-isolation
  baseline recorded at episode start.

Incentivises:
  + Restoring load (proportional to kW restored)
  + Restoring critical loads (hospital bonus)
  + Keeping radial topology
  + Power flow convergence

Penalises:
  - NEW voltage violations caused by agent actions (delta from baseline)
  - NEW line thermal overloads caused by agent actions (delta from baseline)
  - Network loops (hard constraint)
  - Non-convergence (hard constraint)
  - Each switching operation (encourages minimal actions)
  - Each timestep (encourages speed)

Enforced constraints:
  - Voltage bounds: 0.94 -- 1.06 pu (CEB standard)
  - Line ampacity: NormalAmps from OpenDSS model
  - Radial topology: no loops allowed
  - Convergence: power flow must converge
"""
from __future__ import annotations

from typing import List

from config import CRITICAL_LOADS, ALL_LOAD_NAMES


# ── Reward weights (v3 -- delta-based) ──────────────────────────────────
# Key change from v2: restoration is proportional to kW (not count),
# violations are DELTA-based (only NEW ones caused by agent actions).
W_RESTORATION      = 100.0     # full restoration of de-energized kW = +100
W_CRITICAL_BONUS   = 50.0      # hospital restoration bonus
W_VIOLATION_DELTA  = -1.0      # per NEW voltage violation (above baseline)
W_OVERLOAD_DELTA   = -5.0      # per NEW overloaded line (above baseline)
W_LOOP_PENALTY     = -50.0     # hard penalty for creating loops
W_NON_CONVERGENCE  = -100.0    # hard penalty for divergence
W_SWITCHING_COST   = -1.0      # per switch toggled this step
W_STEP_PENALTY     = -0.5      # per timestep


def compute_reward(
    engine,
    de_energized_kw: float,
    initial_de_energized: List[str],
    converged: bool,
    num_switches_toggled: int = 0,
    baseline_violations: int = 0,
    baseline_overloads: int = 0,
) -> float:
    """
    Compute scalar team reward after one action step.

    Parameters
    ----------
    engine                : OpenDSSEngine (already solved)
    de_energized_kw       : total kW lost after isolation (denominator for restoration ratio)
    initial_de_energized  : loads offline right after isolation
    converged             : did the last solve converge
    num_switches_toggled  : how many tie switches changed state this step
    baseline_violations   : voltage violations counted right after isolation (before agent acts)
    baseline_overloads    : overloaded lines counted right after isolation (before agent acts)
    """
    if not converged:
        return W_NON_CONVERGENCE

    reward = W_STEP_PENALTY

    # ── Load restoration (proportional to kW) ────────────────────────
    energized = set(engine.get_energized_loads())
    restored_loads = [ld for ld in initial_de_energized if ld in energized]

    if de_energized_kw > 0:
        # Get kW of restored loads (case-insensitive lookup)
        load_powers = engine.get_load_powers()
        load_powers_lower = {k.lower(): v for k, v in load_powers.items()}
        restored_kw = sum(
            load_powers_lower.get(ld.lower(), 0.0) for ld in restored_loads
        )
        restoration_ratio = min(restored_kw / de_energized_kw, 1.0)
        reward += W_RESTORATION * restoration_ratio
    elif len(restored_loads) > 0:
        # Edge case: de_energized_kw was 0 but loads restored
        reward += W_RESTORATION

    # ── Critical load bonus (hospital) ───────────────────────────────
    for cl in CRITICAL_LOADS:
        if cl in energized and cl in initial_de_energized:
            reward += W_CRITICAL_BONUS

    # ── Voltage violations (DELTA from baseline) ─────────────────────
    current_violations = len(engine.get_voltage_violations())
    new_violations = max(0, current_violations - baseline_violations)
    reward += new_violations * W_VIOLATION_DELTA

    # ── Thermal overloads (DELTA from baseline) ──────────────────────
    current_overloads = len(engine.get_overloaded_lines())
    new_overloads = max(0, current_overloads - baseline_overloads)
    reward += new_overloads * W_OVERLOAD_DELTA

    # ── Radiality check ──────────────────────────────────────────────
    from app.services.grid_graph import build_grid_graph, is_radial
    G = build_grid_graph(include_disabled=False)
    if not is_radial(G):
        reward += W_LOOP_PENALTY

    # ── Switching cost ───────────────────────────────────────────────
    reward += num_switches_toggled * W_SWITCHING_COST

    return reward

"""
Restoration Service
====================
After isolation, re-energize de-energized loads by closing TIE SWITCHES.
Only tie switches are used — CBs/SECs are the isolation service's domain.

Two strategies:
  1. MARL agent (primary) — trained model decides which ties to close
  2. Heuristic fallback  — greedy tie closure with constraint validation
"""
from __future__ import annotations

from typing import List

from loguru import logger

from config import (
    TIE_SWITCHES, TIE_SWITCH_NAMES, CRITICAL_LOADS, ALL_LOAD_NAMES,
)
from app.api.schemas.isolation_output import IsolationResult
from app.api.schemas.restoration_output import RestorationResult, RestorationStep
from app.services.opendss_engine import engine
from app.services.grid_graph import build_grid_graph, is_radial


def _heuristic_restoration(isolation: IsolationResult) -> RestorationResult:
    """
    Greedy: try closing each tie switch one-by-one.
    Keep the closure only if power flow converges AND constraints hold.
    Priority: ties adjacent to F11 (hospital) first.
    """
    steps: List[RestorationStep] = []
    step_counter = 0

    # Prioritise ties that could reach F11 hospital
    def _priority(sw):
        if "f11" in sw.bus_from or "f11" in sw.bus_to:
            return 0
        return 1

    for tie in sorted(TIE_SWITCHES, key=_priority):
        logger.info(f"Trying tie: {tie.name}")

        engine.close_switch(tie.name)
        converged = engine.solve()

        if not converged:
            logger.info(f"  {tie.name}: no convergence → revert")
            engine.open_switch(tie.name)
            continue

        G = build_grid_graph(include_disabled=False)
        if not is_radial(G):
            logger.info(f"  {tie.name}: creates loop → revert")
            engine.open_switch(tie.name)
            continue

        violations = engine.get_voltage_violations()
        if violations:
            logger.info(f"  {tie.name}: voltage violations → revert")
            engine.open_switch(tie.name)
            continue

        overloads = engine.get_overloaded_lines()
        if overloads:
            logger.info(f"  {tie.name}: line overloads → revert")
            engine.open_switch(tie.name)
            continue

        step_counter += 1
        steps.append(RestorationStep(
            step=step_counter, switch_name=tie.name, action="close",
            reason=f"Restores path via {tie.bus_from} ↔ {tie.bus_to}",
        ))
        logger.info(f"  {tie.name}: accepted ✓")

    return _assess_result(isolation, steps, "heuristic")


def _marl_restoration(isolation: IsolationResult) -> RestorationResult:
    """Use trained MARL agent.  Falls back to heuristic if unavailable."""
    try:
        from app.ml.agents.marl_agent import MARLAgent

        agent = MARLAgent.load_latest()
        if agent is None:
            raise FileNotFoundError("No trained model")

        state = engine.get_grid_state()
        tie_actions = agent.predict(state)

        steps: List[RestorationStep] = []
        for i, (sw_name, action) in enumerate(tie_actions):
            engine.set_switch(sw_name, closed=(action == "close"))
            steps.append(RestorationStep(
                step=i + 1, switch_name=sw_name, action=action,
                reason="MARL agent decision",
            ))

        engine.solve()
        return _assess_result(isolation, steps, "marl")

    except Exception as e:
        logger.warning(f"MARL unavailable ({e}), falling back to heuristic")
        return _heuristic_restoration(isolation)


def _assess_result(
    isolation: IsolationResult,
    steps: List[RestorationStep],
    strategy: str,
) -> RestorationResult:
    """Measure the outcome after restoration actions have been applied."""
    engine.solve()

    energized = set(engine.get_energized_loads())
    de_energized = [ld for ld in ALL_LOAD_NAMES if ld not in energized]
    restored = [ld for ld in isolation.de_energized_loads if ld in energized]
    critical_ok = [ld for ld in CRITICAL_LOADS if ld in energized]

    total_kw = engine.get_total_load_kw()
    served_kw = sum(engine.get_load_powers().get(ld, 0) for ld in energized)
    pct = (served_kw / total_kw * 100) if total_kw > 0 else 0

    G = build_grid_graph(include_disabled=False)

    return RestorationResult(
        success=len(steps) > 0,
        strategy=strategy,
        restoration_steps=steps,
        loads_restored=restored,
        loads_still_offline=de_energized,
        critical_loads_restored=critical_ok,
        pct_load_restored=round(pct, 2),
        voltage_violations=engine.get_voltage_violations(),
        overloaded_lines=engine.get_overloaded_lines(),
        is_radial=is_radial(G),
        num_switching_ops=len(steps),
        message=f"{strategy} restoration: {len(restored)} loads restored, "
                f"{len(de_energized)} still offline.",
    )


def restore_service(
    isolation: IsolationResult,
    strategy: str = "auto",
) -> RestorationResult:
    if strategy == "heuristic":
        return _heuristic_restoration(isolation)
    elif strategy == "marl":
        return _marl_restoration(isolation)
    else:
        return _marl_restoration(isolation)

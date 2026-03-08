"""
Service Restoration -- Heuristic + MARL strategies
====================================================
After fault isolation, decides which tie switches to close
to restore maximum load while maintaining grid constraints.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import logging

from config.grid_config import TIE_SWITCH_NAMES, CRITICAL_LOADS, ALL_LOAD_NAMES
from services.opendss_engine import engine
from services.grid_graph_service import build_grid_graph, is_radial

logger = logging.getLogger(__name__)


def restore_heuristic() -> Dict:
    """Greedy: try closing each tie switch, keep if it restores load without violations."""
    closed_switches = []
    for sw in TIE_SWITCH_NAMES:
        engine.close_switch(sw)
        converged = engine.solve()

        G = build_grid_graph(include_disabled=False)
        radial = is_radial(G)

        if converged and radial:
            closed_switches.append(sw)
        else:
            engine.open_switch(sw)
            engine.solve()

    energized = set(engine.get_energized_loads())
    de_energized = [ld for ld in ALL_LOAD_NAMES if ld not in energized]

    return {
        "strategy": "heuristic",
        "closed_switches": closed_switches,
        "energized_loads": list(energized),
        "de_energized_loads": de_energized,
        "restoration_pct": (len(energized) / len(ALL_LOAD_NAMES) * 100) if ALL_LOAD_NAMES else 0,
        "converged": bool(engine.solve()),
        "radial": is_radial(build_grid_graph(include_disabled=False)),
    }


def restore_marl(strategy: str = "gnn") -> Dict:
    """Use trained MARL agent for restoration. Falls back to heuristic on failure."""
    try:
        from ml.inference import run_marl_inference
        result = run_marl_inference(engine, strategy=strategy)
        return result
    except Exception as e:
        logger.warning(f"MARL restoration failed ({e}), falling back to heuristic")
        return restore_heuristic()


def restore(strategy: str = "auto") -> Dict:
    """
    Main entry point for service restoration.
    strategy: 'auto' (MARL first, heuristic fallback), 'marl', 'heuristic'
    """
    if strategy == "heuristic":
        return restore_heuristic()
    elif strategy == "marl":
        return restore_marl()
    else:  # auto
        return restore_marl()

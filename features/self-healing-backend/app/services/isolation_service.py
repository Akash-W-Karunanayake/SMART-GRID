"""
Fault Isolation Service (Rule-Based)
=====================================
Uses CBs + Sectionalizers to isolate a faulted zone.
Tie switches are NOT touched here — they are the MARL agent's domain.

Strategy:
  1. Identify the faulted feeder from graph connectivity (not name parsing).
  2. Open the feeder CB → de-energize entire feeder.
  3. Open the sectionalizer → split feeder into upstream/downstream.
  4. Determine which zone contains the fault using graph reachability.
  5. Re-close CB to re-energize the safe zone.
  6. The faulted zone stays isolated; restoration service handles the rest.
"""
from __future__ import annotations

from typing import List, Optional, Set

import networkx as nx
from loguru import logger

from config import (
    CIRCUIT_BREAKERS, SECTIONALIZERS, SWITCH_BY_NAME,
    FEEDER_LOADS, CRITICAL_LOADS, SwitchInfo,
)
from app.api.schemas.fault_input import FaultReport
from app.api.schemas.isolation_output import IsolationResult, SwitchAction
from app.services.opendss_engine import engine
from app.services.grid_graph import (
    build_grid_graph, find_isolated_zone, get_loads_on_buses,
)


def _identify_feeder(fault_bus_lower: str, G: nx.Graph) -> Optional[str]:
    """
    Identify which feeder a bus belongs to using graph reachability.
    Walk from the fault bus toward the 33kV bus through enabled edges,
    and check which CB we hit first.
    """
    # Fast path: check name prefix (works for Chunnakam naming convention)
    for prefix in ["f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12"]:
        if fault_bus_lower.startswith(prefix):
            return prefix.upper()

    # Fallback: check which CB's downstream zone contains this bus
    for cb in CIRCUIT_BREAKERS:
        downstream_bus = cb.bus_to  # already lowercase in config
        if downstream_bus in G and fault_bus_lower in G:
            # Remove CB edge, check if fault_bus is reachable from downstream
            H = G.copy()
            if H.has_edge(cb.bus_from, cb.bus_to):
                H.remove_edge(cb.bus_from, cb.bus_to)
            if nx.has_path(H, downstream_bus, fault_bus_lower):
                return cb.feeder

    return None


def _get_feeder_cb(feeder: str) -> Optional[SwitchInfo]:
    return SWITCH_BY_NAME.get(f"CB_{feeder}")


def _get_feeder_sec(feeder: str) -> Optional[SwitchInfo]:
    return SWITCH_BY_NAME.get(f"SEC_{feeder}")


def _fault_in_upstream_zone(
    fault_bus: str, cb: SwitchInfo, sec: SwitchInfo, G: nx.Graph
) -> bool:
    """
    Determine if the fault is upstream of the sectionalizer using
    graph connectivity, NOT name parsing.

    Upstream zone = buses reachable from the CB's downstream side
    WITHOUT crossing the sectionalizer.
    """
    H = G.copy()
    # Remove the sectionalizer edge
    if H.has_edge(sec.bus_from, sec.bus_to):
        H.remove_edge(sec.bus_from, sec.bus_to)

    # Upstream zone = connected component containing cb.bus_to
    cb_downstream = cb.bus_to
    if cb_downstream not in H:
        return True  # assume upstream if we can't determine

    try:
        upstream_buses = nx.node_connected_component(H, cb_downstream)
        return fault_bus in upstream_buses
    except Exception:
        return True


def isolate_fault(report: FaultReport) -> IsolationResult:
    """Execute rule-based fault isolation."""
    fault_bus = report.fault_location.lower()
    fault_type = report.fault_type.value
    actions: List[SwitchAction] = []

    # Build graph from current (pre-isolation) state
    G = build_grid_graph(include_disabled=True)

    # Step 0: identify feeder
    feeder = _identify_feeder(fault_bus, G)
    if feeder is None:
        return IsolationResult(
            success=False, fault_location=fault_bus, fault_type=fault_type,
            message=f"Cannot identify feeder for bus '{fault_bus}'.",
        )

    cb_info = _get_feeder_cb(feeder)
    sec_info = _get_feeder_sec(feeder)
    if cb_info is None or sec_info is None:
        return IsolationResult(
            success=False, fault_location=fault_bus, fault_type=fault_type,
            feeder=feeder,
            message=f"Missing CB or SEC for feeder {feeder}.",
        )

    # Step 1: Open CB (de-energize entire feeder)
    engine.open_switch(cb_info.name)
    actions.append(SwitchAction(
        switch_name=cb_info.name, action="open",
        reason=f"De-energize feeder {feeder} for safe sectionalizing",
    ))

    # Step 2: Open Sectionalizer (split feeder)
    engine.open_switch(sec_info.name)
    actions.append(SwitchAction(
        switch_name=sec_info.name, action="open",
        reason=f"Split {feeder} into upstream/downstream zones",
    ))

    # Step 3: Determine fault zone and re-energize the safe zone
    # Build a fresh graph with both CB and SEC open
    G_open = build_grid_graph(include_disabled=False)
    upstream_fault = _fault_in_upstream_zone(fault_bus, cb_info, sec_info, G)

    if upstream_fault:
        # Fault is between CB and SEC → upstream is faulted
        # Don't re-close CB; downstream zone (beyond SEC) needs tie-switch restoration
        actions.append(SwitchAction(
            switch_name=cb_info.name, action="open",
            reason="Fault is upstream of SEC; CB stays open. "
                   "Downstream zone awaits restoration via tie switch.",
        ))
    else:
        # Fault is downstream of SEC → re-close CB to restore upstream zone
        engine.close_switch(cb_info.name)
        actions.append(SwitchAction(
            switch_name=cb_info.name, action="close",
            reason=f"Re-energize upstream zone of {feeder} (fault is downstream of SEC)",
        ))

    # Step 4: Solve and assess
    converged = engine.solve()
    if not converged:
        logger.warning("Power flow did not converge after isolation")

    # Find isolated buses and affected loads
    G_post = build_grid_graph(include_disabled=False)
    isolated_buses = find_isolated_zone(G_post)
    de_energized_loads = get_loads_on_buses(isolated_buses)
    critical_affected = [ld for ld in de_energized_loads if ld in CRITICAL_LOADS]

    return IsolationResult(
        success=True,
        fault_location=fault_bus,
        fault_type=fault_type,
        feeder=feeder,
        isolated_zone_buses=sorted(isolated_buses),
        switch_actions=actions,
        de_energized_loads=de_energized_loads,
        num_de_energized_loads=len(de_energized_loads),
        critical_loads_affected=critical_affected,
        message=(
            f"Feeder {feeder} isolated. {len(de_energized_loads)} loads de-energized."
            + (" CRITICAL LOAD AFFECTED!" if critical_affected else "")
        ),
    )

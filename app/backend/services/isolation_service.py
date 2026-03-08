"""
Fault Isolation Service (Rule-Based)
=====================================
Uses CBs + Sectionalizers to isolate a faulted zone.
Tie switches are NOT touched here -- they are the MARL agent's domain.
"""
from __future__ import annotations

from typing import List, Optional, Set

import networkx as nx
import logging

from config.grid_config import (
    CIRCUIT_BREAKERS, SECTIONALIZERS, SWITCH_BY_NAME,
    CRITICAL_LOADS, SwitchInfo,
)
from models.schemas import FaultReport, SwitchAction, IsolationResult
from services.opendss_engine import engine
from services.grid_graph_service import (
    build_grid_graph, find_isolated_zone, get_loads_on_buses,
)

logger = logging.getLogger(__name__)


def _identify_feeder(fault_bus_lower: str, G: nx.Graph) -> Optional[str]:
    for prefix in ["f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12"]:
        if fault_bus_lower.startswith(prefix):
            return prefix.upper()

    for cb in CIRCUIT_BREAKERS:
        downstream_bus = cb.bus_to
        if downstream_bus in G and fault_bus_lower in G:
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
    H = G.copy()
    if H.has_edge(sec.bus_from, sec.bus_to):
        H.remove_edge(sec.bus_from, sec.bus_to)

    cb_downstream = cb.bus_to
    if cb_downstream not in H:
        return True

    try:
        upstream_buses = nx.node_connected_component(H, cb_downstream)
        return fault_bus in upstream_buses
    except Exception:
        return True


def isolate_fault(report: FaultReport) -> IsolationResult:
    fault_bus = report.fault_location.lower()
    fault_type = report.fault_type
    actions: List[SwitchAction] = []

    G = build_grid_graph(include_disabled=True)

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

    # Step 1: Open CB
    engine.open_switch(cb_info.name)
    actions.append(SwitchAction(
        switch_name=cb_info.name, action="open",
        reason=f"De-energize feeder {feeder}",
    ))

    # Step 2: Open SEC
    engine.open_switch(sec_info.name)
    actions.append(SwitchAction(
        switch_name=sec_info.name, action="open",
        reason=f"Split {feeder} into upstream/downstream zones",
    ))

    # Step 3: Determine fault zone
    upstream_fault = _fault_in_upstream_zone(fault_bus, cb_info, sec_info, G)

    if upstream_fault:
        actions.append(SwitchAction(
            switch_name=cb_info.name, action="open",
            reason="Fault is upstream of SEC; CB stays open.",
        ))
    else:
        engine.close_switch(cb_info.name)
        actions.append(SwitchAction(
            switch_name=cb_info.name, action="close",
            reason=f"Re-energize upstream zone of {feeder}",
        ))

    converged = engine.solve()
    if not converged:
        logger.warning("Power flow did not converge after isolation")

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

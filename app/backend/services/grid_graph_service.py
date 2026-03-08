"""
Grid Graph Builder for Self-Healing
====================================
Builds a NetworkX graph from the live OpenDSS model.
All bus names in the graph are LOWERCASE.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import networkx as nx
import opendssdirect as dss
import logging

from config.grid_config import ALL_SWITCH_NAMES

logger = logging.getLogger(__name__)


def _norm(bus_raw: str) -> str:
    return bus_raw.split(".")[0].lower()


def build_grid_graph(include_disabled: bool = False) -> nx.Graph:
    G = nx.Graph()

    flag = dss.Lines.First()
    while flag > 0:
        name = dss.Lines.Name()
        dss.Circuit.SetActiveElement(f"Line.{name}")
        enabled = bool(dss.CktElement.Enabled())

        if not enabled and not include_disabled:
            flag = dss.Lines.Next()
            continue

        buses = dss.CktElement.BusNames()
        if len(buses) >= 2:
            b1, b2 = _norm(buses[0]), _norm(buses[1])
            is_sw = name in ALL_SWITCH_NAMES
            G.add_edge(b1, b2, name=name, etype="switch" if is_sw else "line",
                       enabled=enabled, is_switch=is_sw)
        flag = dss.Lines.Next()

    flag = dss.Transformers.First()
    while flag > 0:
        name = dss.Transformers.Name()
        dss.Circuit.SetActiveElement(f"Transformer.{name}")
        buses = dss.CktElement.BusNames()
        if len(buses) >= 2:
            b1, b2 = _norm(buses[0]), _norm(buses[1])
            G.add_edge(b1, b2, name=name, etype="transformer",
                       enabled=bool(dss.CktElement.Enabled()), is_switch=False)
        flag = dss.Transformers.Next()

    return G


def find_isolated_zone(G: nx.Graph, source_bus: str = "source_132kv") -> Set[str]:
    if source_bus not in G:
        return set()
    connected = nx.node_connected_component(G, source_bus)
    return set(G.nodes()) - connected


def is_radial(G: nx.Graph) -> bool:
    enabled_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("enabled", True)
    ]
    H = nx.Graph()
    H.add_edges_from(enabled_edges)
    try:
        nx.find_cycle(H)
        return False
    except nx.NetworkXNoCycle:
        return True


def get_loads_on_buses(bus_set: Set[str]) -> List[str]:
    result = []
    flag = dss.Loads.First()
    while flag > 0:
        name = dss.Loads.Name()
        dss.Circuit.SetActiveElement(f"Load.{name}")
        buses = dss.CktElement.BusNames()
        if buses and _norm(buses[0]) in bus_set:
            result.append(name)
        flag = dss.Loads.Next()
    return result


def get_adjacency_list() -> List[Tuple[str, str]]:
    G = build_grid_graph(include_disabled=False)
    return list(G.edges())

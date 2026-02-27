"""
Graph Builder — Extract Chunnakam GSS topology from OpenDSS into NetworkX/edge-index format.
Called once after model compile; result cached to datasets/graph/graph_structure.pkl.
IT22577924 — Karunanayake K.P.A.W.
"""
import pickle
import logging
import numpy as np
import networkx as nx
import opendssdirect as dss

from config import GRAPH_DIR

logger = logging.getLogger(__name__)


def build_graph() -> dict:
    """
    Extract bus-level graph from the currently compiled OpenDSS circuit.

    Returns
    -------
    dict with keys:
      bus_names   : list[str]                  — ordered bus index → name
      bus_index   : dict[str, int]             — name → index
      edge_index  : np.ndarray [2, E]          — COO format
      edge_attr   : np.ndarray [E, 3]          — [R_pu, X_pu, rated_kV]
      edge_names  : list[str]                  — element name for each edge
      nx_graph    : networkx.Graph
    """
    bus_names = list(dss.Circuit.AllBusNames())
    bus_index = {name: i for i, name in enumerate(bus_names)}
    N = len(bus_names)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for name, idx in bus_index.items():
        dss.Circuit.SetActiveBus(name)
        G.nodes[idx]["name"] = name
        G.nodes[idx]["kv_base"] = dss.Bus.kVBase()

    src_list, dst_list, attr_list, edge_names = [], [], [], []

    # --- Lines ---
    dss.Lines.First()
    while True:
        lname = dss.Lines.Name()
        if not lname:
            break

        bus1 = dss.Lines.Bus1().split(".")[0].lower()
        bus2 = dss.Lines.Bus2().split(".")[0].lower()

        if bus1 in bus_index and bus2 in bus_index:
            i, j = bus_index[bus1], bus_index[bus2]

            # Get R1, X1 per-unit values from OpenDSS
            r1 = dss.Lines.R1()     # Ω/km
            x1 = dss.Lines.X1()     # Ω/km
            length = dss.Lines.Length()  # km

            # Rated kV (take sending bus base)
            dss.Circuit.SetActiveBus(bus1)
            rated_kv = dss.Bus.kVBase()

            # Convert to pu on 33kV, 100MVA base
            z_base = (rated_kv ** 2) / 100.0 if rated_kv > 0 else 1.0
            r_pu = (r1 * length) / z_base
            x_pu = (x1 * length) / z_base

            attr = [r_pu, x_pu, rated_kv]
            src_list += [i, j]
            dst_list += [j, i]
            attr_list += [attr, attr]
            edge_names += [f"Line.{lname}", f"Line.{lname}"]
            G.add_edge(i, j, r_pu=r_pu, x_pu=x_pu, rated_kv=rated_kv, name=f"Line.{lname}")

        if not dss.Lines.Next():
            break

    # --- Transformers ---
    dss.Transformers.First()
    while True:
        tname = dss.Transformers.Name()
        if not tname:
            break

        dss.Circuit.SetActiveElement(f"Transformer.{tname}")
        bus_list = dss.CktElement.BusNames()
        if len(bus_list) >= 2:
            bus1 = bus_list[0].split(".")[0].lower()
            bus2 = bus_list[1].split(".")[0].lower()

            if bus1 in bus_index and bus2 in bus_index:
                i, j = bus_index[bus1], bus_index[bus2]

                # Use leakage reactance as X_pu; R≈0 for ideal transformer model
                dss.Transformers.Wdg(1)
                kv1 = dss.Transformers.kV()
                kva = dss.Transformers.kVA()

                r_pu = 0.005   # typical leakage resistance (0.5%)
                x_pu = 0.05    # typical leakage reactance (5%)
                rated_kv = kv1

                attr = [r_pu, x_pu, rated_kv]
                src_list += [i, j]
                dst_list += [j, i]
                attr_list += [attr, attr]
                edge_names += [f"Transformer.{tname}", f"Transformer.{tname}"]
                G.add_edge(i, j, r_pu=r_pu, x_pu=x_pu, rated_kv=rated_kv,
                           name=f"Transformer.{tname}")

        if not dss.Transformers.Next():
            break

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr  = np.array(attr_list, dtype=np.float32)

    logger.info(f"Graph: {N} buses, {len(src_list)//2} edges ({len(src_list)} directed)")

    graph = dict(
        bus_names=bus_names,
        bus_index=bus_index,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_names=edge_names,
        nx_graph=G,
        N_buses=N,
        N_edges=len(src_list),
    )
    return graph


def save_graph(graph: dict, path=None) -> None:
    if path is None:
        path = GRAPH_DIR / "graph_structure.pkl"
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"Graph saved → {path}")


def load_graph(path=None) -> dict:
    if path is None:
        path = GRAPH_DIR / "graph_structure.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

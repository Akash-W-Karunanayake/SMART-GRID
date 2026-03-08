"""
MARL Inference for Service Restoration
=======================================
Loads trained MARL agent checkpoint and runs greedy inference
on the current grid state for tie switch restoration decisions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import numpy as np

from config.grid_config import (
    TIE_SWITCH_NAMES, TIE_SWITCHES, NUM_TIE_SWITCHES,
    ALL_SWITCH_NAMES, ALL_LOAD_NAMES, CRITICAL_LOADS,
)
from services.grid_graph_service import build_grid_graph, is_radial, get_adjacency_list

logger = logging.getLogger(__name__)

# Default checkpoint paths (relative to backend dir)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_GNN_PATH = _BACKEND_DIR / "ml" / "checkpoints" / "gnn_marl_checkpoint.pt"
_DEFAULT_FLAT_PATH = _BACKEND_DIR / "ml" / "checkpoints" / "flat_marl_checkpoint.pt"

# Tie switch bus pairs for GNN local embeddings
TIE_BUS_PAIRS: Dict[str, Tuple[str, str]] = {
    sw.name: (sw.bus_from, sw.bus_to) for sw in TIE_SWITCHES
}

# Cached agents
_gnn_agent = None
_flat_agent = None


def _build_flat_obs(engine) -> Dict[str, np.ndarray]:
    """Build per-agent flat observations from current engine state."""
    all_sw = engine.get_all_switch_states()
    tie_sw = engine.get_tie_switch_states()
    energized = set(engine.get_energized_loads())

    all_sw_vec = np.array(
        [float(all_sw.get(n, False)) for n in ALL_SWITCH_NAMES], dtype=np.float32
    )
    tie_sw_vec = np.array(
        [float(tie_sw.get(n, False)) for n in TIE_SWITCH_NAMES], dtype=np.float32
    )
    load_vec = np.array(
        [1.0 if ld in energized else 0.0 for ld in ALL_LOAD_NAMES], dtype=np.float32
    )

    obs = {}
    for i, sw_name in enumerate(TIE_SWITCH_NAMES):
        agent_id = np.zeros(NUM_TIE_SWITCHES, dtype=np.float32)
        agent_id[i] = 1.0
        obs[sw_name] = np.concatenate([tie_sw_vec, all_sw_vec, load_vec, agent_id])
    return obs


def _load_agent(checkpoint_path: Path):
    """Load a MARL agent from checkpoint (lazy import to avoid torch dependency at startup)."""
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    use_gnn = ckpt.get("use_gnn", False)

    # Lazy import to avoid circular deps and optional torch-geometric
    from ml.agent_networks import QNetwork, HybridQNetwork

    if use_gnn and "gnn_encoder" in ckpt:
        from ml.agent_networks import GNNEncoder
        flat_w = ckpt["q_net"]["flat_branch.0.weight"]
        gnn_w = ckpt["q_net"]["gnn_branch.0.weight"]
        flat_dim = flat_w.shape[1]
        gnn_dim = gnn_w.shape[1]
        hidden_dim = flat_w.shape[0] * 2
        gnn_embed_dim = ckpt.get("gnn_embed_dim", 32)

        q_net = HybridQNetwork(flat_dim, gnn_dim, hidden_dim)
        q_net.load_state_dict(ckpt["q_net"])
        q_net.eval()

        gnn_encoder = GNNEncoder(in_features=1, hidden_dim=64, embed_dim=gnn_embed_dim)
        gnn_encoder.load_state_dict(ckpt["gnn_encoder"])
        gnn_encoder.eval()

        return {"q_net": q_net, "gnn_encoder": gnn_encoder, "use_gnn": True,
                "gnn_embed_dim": gnn_embed_dim}
    else:
        w = ckpt["q_net"]["net.0.weight"]
        obs_dim = w.shape[1]
        hidden_dim = w.shape[0]

        q_net = QNetwork(obs_dim, hidden_dim)
        q_net.load_state_dict(ckpt["q_net"])
        q_net.eval()

        return {"q_net": q_net, "use_gnn": False}


def _select_actions_greedy(agent_data: dict, obs: Dict[str, np.ndarray],
                           engine=None) -> Dict[str, int]:
    """Run greedy action selection using the loaded Q-network."""
    q_net = agent_data["q_net"]
    actions = {}

    gnn_embeds = None
    if agent_data["use_gnn"] and engine is not None:
        gnn_encoder = agent_data["gnn_encoder"]
        gnn_embed_dim = agent_data["gnn_embed_dim"]

        try:
            from ml.agent_networks import grid_state_to_pyg_data
            bus_voltages = engine.get_bus_voltages_pu()
            adj_edges = get_adjacency_list()
            pyg_data = grid_state_to_pyg_data(bus_voltages, adj_edges)

            with torch.no_grad():
                node_embeds = gnn_encoder(pyg_data.x, pyg_data.edge_index)

            bus_names = list(bus_voltages.keys())
            bus_to_idx = {b: i for i, b in enumerate(bus_names)}

            global_embed = node_embeds.mean(dim=0)
            gnn_embeds = {}
            for sw_name in TIE_SWITCH_NAMES:
                b1, b2 = TIE_BUS_PAIRS[sw_name]
                idx1 = bus_to_idx.get(b1, 0)
                idx2 = bus_to_idx.get(b2, 0)
                local_embed = torch.cat([node_embeds[idx1], node_embeds[idx2]])
                gnn_embeds[sw_name] = torch.cat([global_embed, local_embed])
        except Exception as e:
            logger.warning(f"GNN embedding failed: {e}, using flat only")
            gnn_embeds = None

    with torch.no_grad():
        for sw_name in TIE_SWITCH_NAMES:
            flat_t = torch.from_numpy(obs[sw_name]).unsqueeze(0)

            if gnn_embeds is not None and sw_name in gnn_embeds:
                gnn_t = gnn_embeds[sw_name].unsqueeze(0)
                q_values = q_net(flat_t, gnn_t)
            elif agent_data["use_gnn"]:
                # GNN failed, skip this agent (keep open)
                actions[sw_name] = 0
                continue
            else:
                q_values = q_net(flat_t)

            actions[sw_name] = int(q_values.argmax(dim=1).item())

    return actions


def run_marl_inference(engine, strategy: str = "gnn") -> Dict:
    """
    Run MARL inference on current grid state.
    strategy: 'gnn' or 'flat'
    """
    global _gnn_agent, _flat_agent

    if strategy == "gnn":
        if _gnn_agent is None:
            _gnn_agent = _load_agent(_DEFAULT_GNN_PATH)
        agent_data = _gnn_agent
    else:
        if _flat_agent is None:
            _flat_agent = _load_agent(_DEFAULT_FLAT_PATH)
        agent_data = _flat_agent

    if agent_data is None:
        raise RuntimeError(f"No trained {strategy} model available")

    obs = _build_flat_obs(engine)
    actions = _select_actions_greedy(agent_data, obs, engine=engine)

    # Apply actions
    closed_switches = []
    for sw_name, act in actions.items():
        engine.set_switch(sw_name, closed=bool(act))
        if act == 1:
            closed_switches.append(sw_name)

    converged = engine.solve()
    energized = set(engine.get_energized_loads())
    de_energized = [ld for ld in ALL_LOAD_NAMES if ld not in energized]

    return {
        "strategy": f"marl_{strategy}",
        "closed_switches": closed_switches,
        "actions": actions,
        "energized_loads": list(energized),
        "de_energized_loads": de_energized,
        "restoration_pct": (len(energized) / len(ALL_LOAD_NAMES) * 100) if ALL_LOAD_NAMES else 0,
        "converged": converged,
        "radial": is_radial(build_grid_graph(include_disabled=False)),
        "voltage_violations": engine.get_voltage_violations(),
        "overloaded_lines": engine.get_overloaded_lines(),
    }

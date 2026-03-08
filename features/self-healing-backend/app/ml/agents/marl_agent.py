"""
MARL Agent -- Multi-Agent DQN with GNN for 5 Tie Switches
==========================================================
Each tie switch is controlled by an independent DQN agent.
All agents share the same Q-network weights (parameter sharing)
for "Centralized Training, Decentralized Execution" (CTDE).

The GNN encoder processes the grid graph to produce topology-aware
embeddings that are combined with flat features before the Q-network.

Architecture:
  grid_graph --> GNNEncoder --> global_embed + agent_local_embed
  flat_features (switches, loads, agent_id) ---|
                                               +--> HybridQNetwork --> Q(s,a)
  gnn_embed (global + local) -----------------|

Action space: 5 agents x Discrete(2)
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from config import TIE_SWITCH_NAMES, NUM_TIE_SWITCHES, ALL_LOAD_NAMES, settings
from config import TIE_SWITCHES, ALL_SWITCH_NAMES

try:
    from app.ml.models.gnn_encoder import GNNEncoder, grid_state_to_pyg_data, HAS_PYG
except ImportError:
    HAS_PYG = False


# ── Tie switch bus mapping (for agent-local GNN embeddings) ──────────────
TIE_BUS_PAIRS: Dict[str, Tuple[str, str]] = {
    sw.name: (sw.bus_from, sw.bus_to) for sw in TIE_SWITCHES
}


# ── Network architectures ───────────────────────────────────────────────

class QNetwork(nn.Module):
    """Original flat Q-network: obs_vector -> Q(s, open), Q(s, close).
    Kept for backward compatibility when GNN is not available."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridQNetwork(nn.Module):
    """
    GNN-enhanced Q-network combining flat features with topology-aware
    graph embeddings.

    Input:
      flat_obs: (batch, flat_dim) -- switch states, load flags, agent ID
      gnn_embed: (batch, gnn_dim) -- global + agent-local GNN embeddings

    Output:
      Q-values: (batch, 2) -- Q(s, open), Q(s, close)
    """

    def __init__(self, flat_dim: int, gnn_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.flat_dim = flat_dim
        self.gnn_dim = gnn_dim

        # Process flat features
        self.flat_branch = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Process GNN embeddings
        self.gnn_branch = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Fusion and Q-value head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, flat_obs: torch.Tensor, gnn_embed: torch.Tensor,
    ) -> torch.Tensor:
        flat_h = self.flat_branch(flat_obs)
        gnn_h = self.gnn_branch(gnn_embed)
        combined = torch.cat([flat_h, gnn_h], dim=-1)
        return self.fusion(combined)


# ── Replay buffer ────────────────────────────────────────────────────────

@dataclass
class Transition:
    flat_obs: Dict[str, np.ndarray]
    gnn_embed: Optional[Dict[str, np.ndarray]]  # per-agent GNN embeddings
    actions: Dict[str, int]
    reward: float
    next_flat_obs: Dict[str, np.ndarray]
    next_gnn_embed: Optional[Dict[str, np.ndarray]]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, n: int) -> List[Transition]:
        return random.sample(self.buf, min(n, len(self.buf)))

    def __len__(self):
        return len(self.buf)


# ── Main agent class ────────────────────────────────────────────────────

class MARLAgent:
    """
    Multi-Agent DQN for 5 tie switches with shared Q-network.
    Supports two modes:
      1. Hybrid (GNN + flat): when use_gnn=True and torch_geometric is available
      2. Flat-only: fallback when GNN is unavailable
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu",
        use_gnn: bool = True,
        gnn_in_features: int = 1,
        gnn_hidden_dim: int = 64,
        gnn_embed_dim: int = 32,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self._steps = 0

        # Determine mode
        self.use_gnn = use_gnn and HAS_PYG
        if use_gnn and not HAS_PYG:
            logger.warning("torch_geometric not available, falling back to flat-only mode")

        if self.use_gnn:
            self.gnn_embed_dim = gnn_embed_dim
            # GNN dim per agent = global_embed (embed_dim) + local_embed (2 * embed_dim)
            gnn_dim = gnn_embed_dim * 3
            # Flat dim = obs_dim without bus voltages (will be computed dynamically)
            # For now, flat_dim is passed explicitly
            self.flat_dim = obs_dim

            self.gnn_encoder = GNNEncoder(
                in_features=gnn_in_features,
                hidden_dim=gnn_hidden_dim,
                embed_dim=gnn_embed_dim,
            ).to(self.device)

            self.q_net = HybridQNetwork(
                flat_dim=obs_dim, gnn_dim=gnn_dim, hidden_dim=hidden_dim,
            ).to(self.device)
            self.target_net = HybridQNetwork(
                flat_dim=obs_dim, gnn_dim=gnn_dim, hidden_dim=hidden_dim,
            ).to(self.device)
            self.target_gnn = GNNEncoder(
                in_features=gnn_in_features,
                hidden_dim=gnn_hidden_dim,
                embed_dim=gnn_embed_dim,
            ).to(self.device)
            self.target_gnn.load_state_dict(self.gnn_encoder.state_dict())

            # Single optimizer for all parameters (end-to-end training)
            all_params = list(self.q_net.parameters()) + list(self.gnn_encoder.parameters())
            self.optimizer = optim.Adam(all_params, lr=lr)
        else:
            self.gnn_encoder = None
            self.gnn_embed_dim = 0

            self.q_net = QNetwork(obs_dim, hidden_dim).to(self.device)
            self.target_net = QNetwork(obs_dim, hidden_dim).to(self.device)
            self.target_gnn = None
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.replay = ReplayBuffer()

    # ── GNN embedding computation ────────────────────────────────────────

    def compute_gnn_embeddings(
        self, graph_data: dict, use_target: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run GNN on graph data and return per-agent embedding vectors.

        graph_data should contain:
          - bus_voltages: Dict[str, float]
          - adjacency_edges: List[Tuple[str, str]]

        Returns: {agent_name: np.ndarray of shape (3 * embed_dim,)}
        """
        if not self.use_gnn:
            return {}

        from app.ml.models.gnn_encoder import grid_state_to_pyg_data

        bus_voltages = graph_data["bus_voltages"]
        adjacency_edges = graph_data["adjacency_edges"]

        pyg_data, bus_to_idx = grid_state_to_pyg_data(bus_voltages, adjacency_edges)
        if pyg_data is None:
            return {}

        pyg_data = pyg_data.to(self.device)
        encoder = self.target_gnn if use_target else self.gnn_encoder

        with torch.no_grad() if use_target else torch.enable_grad():
            node_embeddings = encoder(pyg_data)

        global_embed = encoder.get_global_embedding(node_embeddings)

        agent_embeds = {}
        for sw_name in TIE_SWITCH_NAMES:
            bus_from, bus_to = TIE_BUS_PAIRS[sw_name]
            local_embed = encoder.get_agent_local_embedding(
                node_embeddings, bus_to_idx, bus_from, bus_to,
            )
            # Concat global + local: (embed_dim + 2*embed_dim) = (3*embed_dim)
            full_embed = torch.cat([global_embed, local_embed])
            agent_embeds[sw_name] = full_embed.detach().cpu().numpy()

        return agent_embeds

    def _compute_gnn_embeddings_for_training(
        self, graph_data: dict, use_target: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, int]]]:
        """
        Same as compute_gnn_embeddings but returns tensors (with grad)
        for use in the training loop.
        """
        from app.ml.models.gnn_encoder import grid_state_to_pyg_data

        bus_voltages = graph_data["bus_voltages"]
        adjacency_edges = graph_data["adjacency_edges"]

        pyg_data, bus_to_idx = grid_state_to_pyg_data(bus_voltages, adjacency_edges)
        if pyg_data is None:
            return {}, None

        pyg_data = pyg_data.to(self.device)
        encoder = self.target_gnn if use_target else self.gnn_encoder

        node_embeddings = encoder(pyg_data)
        global_embed = encoder.get_global_embedding(node_embeddings)

        agent_embeds = {}
        for sw_name in TIE_SWITCH_NAMES:
            bus_from, bus_to = TIE_BUS_PAIRS[sw_name]
            local_embed = encoder.get_agent_local_embedding(
                node_embeddings, bus_to_idx, bus_from, bus_to,
            )
            agent_embeds[sw_name] = torch.cat([global_embed, local_embed])

        return agent_embeds, bus_to_idx

    # ── action selection ────────────────────────────────────────────────

    def select_actions(
        self,
        obs: Dict[str, np.ndarray],
        graph_data: Optional[dict] = None,
        greedy: bool = False,
    ) -> Dict[str, int]:
        """
        Select actions for all 5 tie switches.

        Args:
            obs: flat observations per agent {switch_name: np.ndarray}
            graph_data: raw graph data for GNN (bus_voltages, adjacency_edges)
            greedy: if True, skip epsilon-greedy exploration
        """
        # Compute GNN embeddings if available
        gnn_embeds = {}
        if self.use_gnn and graph_data is not None:
            gnn_embeds = self.compute_gnn_embeddings(graph_data)

        actions = {}
        for sw in TIE_SWITCH_NAMES:
            if not greedy and random.random() < self.epsilon:
                actions[sw] = random.randint(0, 1)
            else:
                with torch.no_grad():
                    flat = torch.FloatTensor(obs[sw]).unsqueeze(0).to(self.device)

                    if self.use_gnn and sw in gnn_embeds:
                        gnn_e = torch.FloatTensor(gnn_embeds[sw]).unsqueeze(0).to(self.device)
                        q_vals = self.q_net(flat, gnn_e)
                    else:
                        q_vals = self.q_net(flat)

                    actions[sw] = int(q_vals.argmax(1).item())
        return actions

    def predict(self, grid_state: dict) -> List[Tuple[str, str]]:
        """Inference for REST API. Returns list of (switch, action) to CHANGE."""
        tie_states = grid_state.get("tie_switch_states", {})
        all_sw = grid_state.get("all_switch_states", {})
        bus_v = grid_state.get("bus_voltages_pu", {})
        energized = set(grid_state.get("energized_loads", []))

        all_sw_vec = np.array(
            [float(all_sw.get(n, False)) for n in ALL_SWITCH_NAMES], dtype=np.float32
        )
        tie_sw_vec = np.array(
            [float(tie_states.get(n, False)) for n in TIE_SWITCH_NAMES], dtype=np.float32
        )
        load_vec = np.array(
            [1.0 if ld in energized else 0.0 for ld in ALL_LOAD_NAMES], dtype=np.float32
        )

        # Compute GNN embeddings from grid state
        gnn_embeds = {}
        if self.use_gnn:
            from app.services.grid_graph import get_adjacency_list
            graph_data = {
                "bus_voltages": bus_v,
                "adjacency_edges": get_adjacency_list(),
            }
            gnn_embeds = self.compute_gnn_embeddings(graph_data)

        changes = []
        for i, sw in enumerate(TIE_SWITCH_NAMES):
            agent_id = np.zeros(NUM_TIE_SWITCHES, dtype=np.float32)
            agent_id[i] = 1.0
            flat_vec = np.concatenate([tie_sw_vec, all_sw_vec, load_vec, agent_id])

            with torch.no_grad():
                flat = torch.FloatTensor(flat_vec).unsqueeze(0).to(self.device)

                if self.use_gnn and sw in gnn_embeds:
                    gnn_e = torch.FloatTensor(gnn_embeds[sw]).unsqueeze(0).to(self.device)
                    q_vals = self.q_net(flat, gnn_e)
                else:
                    q_vals = self.q_net(flat)

                action = int(q_vals.argmax(1).item())

            new_closed = bool(action)
            old_closed = tie_states.get(sw, False)
            if new_closed != old_closed:
                changes.append((sw, "close" if new_closed else "open"))

        return changes

    # ── training ────────────────────────────────────────────────────────

    def store(self, obs, actions, reward, next_obs, done,
              gnn_embed=None, next_gnn_embed=None):
        self.replay.push(Transition(
            flat_obs=obs,
            gnn_embed=gnn_embed,
            actions=actions,
            reward=reward,
            next_flat_obs=next_obs,
            next_gnn_embed=next_gnn_embed,
            done=done,
        ))

    def update(self):
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        total_loss = torch.tensor(0.0, device=self.device)

        has_gnn = self.use_gnn and batch[0].gnn_embed is not None

        for sw in TIE_SWITCH_NAMES:
            flat_b = torch.FloatTensor(
                np.array([t.flat_obs[sw] for t in batch])
            ).to(self.device)
            act_b = torch.LongTensor(
                [t.actions[sw] for t in batch]
            ).to(self.device)
            rew_b = torch.FloatTensor(
                [t.reward for t in batch]
            ).to(self.device)
            nxt_flat_b = torch.FloatTensor(
                np.array([t.next_flat_obs[sw] for t in batch])
            ).to(self.device)
            don_b = torch.FloatTensor(
                [float(t.done) for t in batch]
            ).to(self.device)

            if has_gnn:
                gnn_b = torch.FloatTensor(
                    np.array([t.gnn_embed[sw] for t in batch])
                ).to(self.device)
                nxt_gnn_b = torch.FloatTensor(
                    np.array([t.next_gnn_embed[sw] for t in batch])
                ).to(self.device)

                q = self.q_net(flat_b, gnn_b).gather(1, act_b.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    nxt_q = self.target_net(nxt_flat_b, nxt_gnn_b).max(1).values
                    target = rew_b + self.gamma * nxt_q * (1 - don_b)
            else:
                q = self.q_net(flat_b).gather(1, act_b.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    nxt_q = self.target_net(nxt_flat_b).max(1).values
                    target = rew_b + self.gamma * nxt_q * (1 - don_b)

            total_loss = total_loss + nn.functional.mse_loss(q, target)

        total_loss = total_loss / NUM_TIE_SWITCHES
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._steps += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay_steps,
        )
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            if self.use_gnn and self.target_gnn is not None:
                self.target_gnn.load_state_dict(self.gnn_encoder.state_dict())

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str):
        state = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self._steps,
            "use_gnn": self.use_gnn,
        }
        if self.use_gnn and self.gnn_encoder is not None:
            state["gnn_encoder"] = self.gnn_encoder.state_dict()
            state["target_gnn"] = self.target_gnn.state_dict()
            state["gnn_embed_dim"] = self.gnn_embed_dim
        torch.save(state, path)
        logger.info(f"Saved -> {path}")

    @classmethod
    def load_latest(cls) -> Optional["MARLAgent"]:
        mp = settings.TRAINED_MODEL_PATH
        if mp and Path(mp).exists():
            ckpt = torch.load(mp, map_location="cpu", weights_only=False)
            use_gnn = ckpt.get("use_gnn", False)

            if use_gnn and "gnn_encoder" in ckpt:
                gnn_embed_dim = ckpt.get("gnn_embed_dim", 32)
                # Infer flat_dim from q_net weights
                flat_w = ckpt["q_net"]["flat_branch.0.weight"]
                flat_dim = flat_w.shape[1]
                gnn_w = ckpt["q_net"]["gnn_branch.0.weight"]
                gnn_dim = gnn_w.shape[1]
                hidden_dim = flat_w.shape[0] * 2

                agent = cls(
                    obs_dim=flat_dim,
                    hidden_dim=hidden_dim,
                    use_gnn=True,
                    gnn_embed_dim=gnn_embed_dim,
                )
                agent.gnn_encoder.load_state_dict(ckpt["gnn_encoder"])
                agent.target_gnn.load_state_dict(ckpt.get("target_gnn", ckpt["gnn_encoder"]))
            else:
                w = ckpt["q_net"]["net.0.weight"]
                agent = cls(obs_dim=w.shape[1], hidden_dim=w.shape[0], use_gnn=False)

            agent.q_net.load_state_dict(ckpt["q_net"])
            agent.target_net.load_state_dict(ckpt["target_net"])
            agent.epsilon = ckpt.get("epsilon", 0.05)
            logger.info(f"Loaded model from {mp} (use_gnn={use_gnn})")
            return agent
        return None

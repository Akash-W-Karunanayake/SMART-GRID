"""
Tests for GNN-MARL integration.
Verifies that the hybrid architecture correctly wires GNN embeddings
into the MARL agent's Q-network.

Run: pytest tests/test_gnn_marl.py -v
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from config.grid_config import (
    TIE_SWITCH_NAMES, TIE_SWITCHES, NUM_TIE_SWITCHES,
    ALL_SWITCH_NAMES, ALL_LOAD_NAMES,
)

# Check if torch_geometric is available
try:
    from torch_geometric.data import Data
    from app.ml.models.gnn_encoder import GNNEncoder, grid_state_to_pyg_data, HAS_PYG
    HAS_PYG_ACTUAL = True
except ImportError:
    HAS_PYG_ACTUAL = False

from app.ml.agents.marl_agent import (
    QNetwork, HybridQNetwork, MARLAgent, TIE_BUS_PAIRS,
)


# ── GNN Encoder Tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_PYG_ACTUAL, reason="torch_geometric not installed")
class TestGNNEncoder:

    def test_encoder_output_shape(self):
        """GNN produces (N_nodes, embed_dim) embeddings."""
        encoder = GNNEncoder(in_features=1, hidden_dim=16, embed_dim=8)
        N = 10
        x = torch.randn(N, 1)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        out = encoder(data)
        assert out.shape == (N, 8)

    def test_global_embedding_shape(self):
        """Mean-pooled global embedding is (embed_dim,)."""
        encoder = GNNEncoder(in_features=1, hidden_dim=16, embed_dim=8)
        node_embs = torch.randn(10, 8)
        global_emb = encoder.get_global_embedding(node_embs)
        assert global_emb.shape == (8,)

    def test_agent_local_embedding_shape(self):
        """Agent-local embedding = concat of 2 bus embeddings = (2*embed_dim,)."""
        encoder = GNNEncoder(in_features=1, hidden_dim=16, embed_dim=8)
        node_embs = torch.randn(5, 8)
        bus_to_idx = {"bus_a": 0, "bus_b": 1, "bus_c": 2}

        local = encoder.get_agent_local_embedding(node_embs, bus_to_idx, "bus_a", "bus_b")
        assert local.shape == (16,)  # 2 * 8

    def test_agent_local_embedding_missing_bus(self):
        """Falls back to zeros for missing bus names."""
        encoder = GNNEncoder(in_features=1, hidden_dim=16, embed_dim=8)
        node_embs = torch.randn(5, 8)
        bus_to_idx = {"bus_a": 0}

        local = encoder.get_agent_local_embedding(node_embs, bus_to_idx, "bus_a", "missing_bus")
        assert local.shape == (16,)
        # Second half should be zeros
        assert torch.all(local[8:] == 0)

    def test_grid_state_to_pyg_data(self):
        """Conversion from grid state dict to PyG Data object."""
        bus_v = {"bus_a": 1.0, "bus_b": 0.95, "bus_c": 1.02}
        edges = [("bus_a", "bus_b"), ("bus_b", "bus_c")]

        data, bus_to_idx = grid_state_to_pyg_data(bus_v, edges)
        assert data is not None
        assert data.x.shape == (3, 1)
        assert data.edge_index.shape[0] == 2
        # Undirected: 2 edges * 2 directions = 4
        assert data.edge_index.shape[1] == 4
        assert len(bus_to_idx) == 3

    def test_grid_state_empty_graph(self):
        """Empty edge list produces valid Data with no edges."""
        bus_v = {"bus_a": 1.0}
        edges = []

        data, bus_to_idx = grid_state_to_pyg_data(bus_v, edges)
        assert data is not None
        assert data.edge_index.shape == (2, 0)


# ── Network Architecture Tests ────────────────────────────────────────

class TestQNetwork:

    def test_flat_qnetwork_output(self):
        """Flat Q-network produces 2 Q-values per input."""
        net = QNetwork(obs_dim=53, hidden_dim=64)
        x = torch.randn(4, 53)  # batch of 4
        q = net(x)
        assert q.shape == (4, 2)

    def test_hybrid_qnetwork_output(self):
        """Hybrid Q-network produces 2 Q-values from flat + GNN input."""
        net = HybridQNetwork(flat_dim=53, gnn_dim=96, hidden_dim=128)
        flat = torch.randn(4, 53)
        gnn = torch.randn(4, 96)
        q = net(flat, gnn)
        assert q.shape == (4, 2)

    def test_hybrid_qnetwork_gradients_flow(self):
        """Gradients flow through both branches."""
        net = HybridQNetwork(flat_dim=10, gnn_dim=20, hidden_dim=32)
        flat = torch.randn(2, 10, requires_grad=True)
        gnn = torch.randn(2, 20, requires_grad=True)
        q = net(flat, gnn)
        loss = q.sum()
        loss.backward()
        assert flat.grad is not None
        assert gnn.grad is not None


# ── MARL Agent Tests ──────────────────────────────────────────────────

class TestMARLAgent:

    def test_flat_agent_creation(self):
        """Agent works in flat-only mode."""
        agent = MARLAgent(obs_dim=53, hidden_dim=64, use_gnn=False)
        assert not agent.use_gnn
        assert isinstance(agent.q_net, QNetwork)

    @pytest.mark.skipif(not HAS_PYG_ACTUAL, reason="torch_geometric not installed")
    def test_gnn_agent_creation(self):
        """Agent works in GNN mode."""
        agent = MARLAgent(obs_dim=53, hidden_dim=64, use_gnn=True, gnn_embed_dim=16)
        assert agent.use_gnn
        assert isinstance(agent.q_net, HybridQNetwork)
        assert agent.gnn_encoder is not None

    def test_flat_select_actions(self):
        """Flat agent returns valid actions for all 5 tie switches."""
        agent = MARLAgent(obs_dim=53, hidden_dim=64, use_gnn=False)
        obs = {sw: np.random.randn(53).astype(np.float32) for sw in TIE_SWITCH_NAMES}
        actions = agent.select_actions(obs, greedy=True)
        assert len(actions) == 5
        for sw, act in actions.items():
            assert sw in TIE_SWITCH_NAMES
            assert act in (0, 1)

    @pytest.mark.skipif(not HAS_PYG_ACTUAL, reason="torch_geometric not installed")
    def test_gnn_select_actions(self):
        """GNN agent returns valid actions with graph data."""
        agent = MARLAgent(obs_dim=53, hidden_dim=64, use_gnn=True, gnn_embed_dim=8)
        obs = {sw: np.random.randn(53).astype(np.float32) for sw in TIE_SWITCH_NAMES}

        # Minimal synthetic graph data
        graph_data = {
            "bus_voltages": {
                "bus_33kv_a": 1.0, "bus_33kv_b": 1.0,
                "f06_node4": 0.98, "f07_node4": 0.97,
                "f07_node4": 0.97, "f08_node4": 0.96,
                "f09_node5": 0.95, "f10_node4": 0.94,
                "f10_node4": 0.94, "f11_node5": 0.93,
            },
            "adjacency_edges": [
                ("bus_33kv_a", "f06_node4"),
                ("f06_node4", "f07_node4"),
                ("f07_node4", "f08_node4"),
                ("bus_33kv_b", "f09_node5"),
                ("f09_node5", "f10_node4"),
                ("f10_node4", "f11_node5"),
            ],
        }

        actions = agent.select_actions(obs, graph_data=graph_data, greedy=True)
        assert len(actions) == 5
        for sw, act in actions.items():
            assert act in (0, 1)

    @pytest.mark.skipif(not HAS_PYG_ACTUAL, reason="torch_geometric not installed")
    def test_gnn_compute_embeddings(self):
        """GNN embeddings have correct shape per agent."""
        agent = MARLAgent(obs_dim=53, hidden_dim=64, use_gnn=True, gnn_embed_dim=8)

        graph_data = {
            "bus_voltages": {"bus_a": 1.0, "bus_b": 0.95, "bus_c": 0.98},
            "adjacency_edges": [("bus_a", "bus_b"), ("bus_b", "bus_c")],
        }

        embeds = agent.compute_gnn_embeddings(graph_data)
        assert len(embeds) == NUM_TIE_SWITCHES
        for sw, emb in embeds.items():
            # global (8) + local (2*8) = 24
            assert emb.shape == (24,)

    def test_flat_store_and_update(self):
        """Flat agent can store transitions and run update without errors."""
        agent = MARLAgent(obs_dim=10, hidden_dim=32, use_gnn=False, batch_size=4)
        obs = {sw: np.random.randn(10).astype(np.float32) for sw in TIE_SWITCH_NAMES}
        actions = {sw: random.choice([0, 1]) for sw in TIE_SWITCH_NAMES}
        nxt = {sw: np.random.randn(10).astype(np.float32) for sw in TIE_SWITCH_NAMES}

        # Store enough for a batch
        for _ in range(10):
            agent.store(obs, actions, 1.0, nxt, False)

        # Should not raise
        agent.update()

    @pytest.mark.skipif(not HAS_PYG_ACTUAL, reason="torch_geometric not installed")
    def test_gnn_store_and_update(self):
        """GNN agent can store transitions with embeddings and run update."""
        agent = MARLAgent(obs_dim=10, hidden_dim=32, use_gnn=True,
                          gnn_embed_dim=4, batch_size=4)
        obs = {sw: np.random.randn(10).astype(np.float32) for sw in TIE_SWITCH_NAMES}
        gnn_e = {sw: np.random.randn(12).astype(np.float32) for sw in TIE_SWITCH_NAMES}
        actions = {sw: random.choice([0, 1]) for sw in TIE_SWITCH_NAMES}
        nxt = {sw: np.random.randn(10).astype(np.float32) for sw in TIE_SWITCH_NAMES}
        nxt_gnn = {sw: np.random.randn(12).astype(np.float32) for sw in TIE_SWITCH_NAMES}

        for _ in range(10):
            agent.store(obs, actions, 1.0, nxt, False,
                        gnn_embed=gnn_e, next_gnn_embed=nxt_gnn)

        agent.update()

    @pytest.mark.skipif(not HAS_PYG_ACTUAL, reason="torch_geometric not installed")
    def test_save_and_load_gnn_agent(self, tmp_path):
        """GNN agent save/load preserves architecture and weights."""
        agent = MARLAgent(obs_dim=53, hidden_dim=64, use_gnn=True, gnn_embed_dim=8)
        path = str(tmp_path / "test_model.pt")
        agent.save(path)

        backup = settings.TRAINED_MODEL_PATH
        settings.TRAINED_MODEL_PATH = path

        loaded = MARLAgent.load_latest()
        assert loaded is not None
        assert loaded.use_gnn

        settings.TRAINED_MODEL_PATH = backup

    def test_save_and_load_flat_agent(self, tmp_path):
        """Flat agent save/load preserves architecture and weights."""
        agent = MARLAgent(obs_dim=53, hidden_dim=64, use_gnn=False)
        path = str(tmp_path / "test_flat.pt")
        agent.save(path)

        backup = settings.TRAINED_MODEL_PATH
        settings.TRAINED_MODEL_PATH = path

        loaded = MARLAgent.load_latest()
        assert loaded is not None
        assert not loaded.use_gnn

        settings.TRAINED_MODEL_PATH = backup


# ── Tie Switch Bus Pair Tests ─────────────────────────────────────────

class TestTieBusPairs:

    def test_all_ties_have_bus_pairs(self):
        """Every tie switch has a (bus_from, bus_to) mapping."""
        for sw in TIE_SWITCH_NAMES:
            assert sw in TIE_BUS_PAIRS
            pair = TIE_BUS_PAIRS[sw]
            assert len(pair) == 2

    def test_bus_pairs_are_lowercase(self):
        """All bus names in tie pairs must be lowercase."""
        for sw, (b_from, b_to) in TIE_BUS_PAIRS.items():
            assert b_from == b_from.lower(), f"{sw}: bus_from not lowercase"
            assert b_to == b_to.lower(), f"{sw}: bus_to not lowercase"

    def test_bus_pairs_match_config(self):
        """Bus pairs match the SwitchInfo from grid_config."""
        for sw_info in TIE_SWITCHES:
            pair = TIE_BUS_PAIRS[sw_info.name]
            assert pair == (sw_info.bus_from, sw_info.bus_to)


from config import settings

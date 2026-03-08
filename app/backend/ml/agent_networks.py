"""
Neural network architectures for MARL inference.
Mirrors the feature backend's QNetwork, HybridQNetwork, and GNNEncoder
but only includes forward pass (no training logic).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Optional PyG dependency
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class QNetwork(nn.Module):
    """Flat-only Q-network."""
    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridQNetwork(nn.Module):
    """Q-network combining flat features with GNN embeddings."""
    def __init__(self, flat_dim: int, gnn_dim: int, hidden_dim: int = 128, n_actions: int = 2):
        super().__init__()
        branch_dim = hidden_dim // 2
        self.flat_branch = nn.Sequential(
            nn.Linear(flat_dim, branch_dim),
            nn.ReLU(),
        )
        self.gnn_branch = nn.Sequential(
            nn.Linear(gnn_dim, branch_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, flat_obs: torch.Tensor, gnn_embed: torch.Tensor) -> torch.Tensor:
        flat_out = self.flat_branch(flat_obs)
        gnn_out = self.gnn_branch(gnn_embed)
        fused = torch.cat([flat_out, gnn_out], dim=-1)
        return self.fusion(fused)


class GNNEncoder(nn.Module):
    """2-layer GCN producing per-node embeddings."""
    def __init__(self, in_features: int = 1, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric required for GNN encoder")
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h


def grid_state_to_pyg_data(bus_voltages: Dict[str, float],
                           adjacency_edges: List[Tuple[str, str]]) -> "Data":
    """Convert grid state to PyG Data object."""
    if not HAS_PYG:
        raise ImportError("torch_geometric required")

    bus_names = list(bus_voltages.keys())
    bus_to_idx = {b: i for i, b in enumerate(bus_names)}

    x = torch.tensor([[bus_voltages[b]] for b in bus_names], dtype=torch.float32)

    edges_src, edges_dst = [], []
    for b1, b2 in adjacency_edges:
        if b1 in bus_to_idx and b2 in bus_to_idx:
            i, j = bus_to_idx[b1], bus_to_idx[b2]
            edges_src.extend([i, j])
            edges_dst.extend([j, i])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

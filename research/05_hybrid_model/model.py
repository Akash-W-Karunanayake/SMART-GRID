"""
Hybrid CNN-Transformer + R-GNN Model.
Integrates temporal and spatial branches via Graph-Informed Attention Fusion.
IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import sub-models by full path to avoid module name collision
# (both files are named 'model.py' in different stage directories)
def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_STAGE03 = Path(__file__).parent.parent / "03_cnn_transformer"
_STAGE04 = Path(__file__).parent.parent / "04_r_gnn"

sys.path.insert(0, str(_STAGE03))
sys.path.insert(0, str(_STAGE04))

_cnn_t_mod = _import_module("cnn_transformer_model", _STAGE03 / "model.py")
_rgnn_mod  = _import_module("rgnn_model",            _STAGE04 / "model.py")

CNNTransformerModel = _cnn_t_mod.CNNTransformerModel
RGNNModel           = _rgnn_mod.RGNNModel

from fusion import GraphInformedFusion

from config import (
    CNN_T_DIM, RGNN_DIM, FUSED_DIM,
    N_CLASSES_DETECTION, N_CLASSES_TYPE, N_CLASSES_PHASE,
)


class HybridModel(nn.Module):
    """
    Full hybrid model for multi-task fault diagnostics.

    Parameters
    ----------
    in_features_cnn : int  — N_branches × 3 (temporal input)
    n_buses         : int  — number of grid buses
    """

    def __init__(self, in_features_cnn: int, n_buses: int):
        super().__init__()
        self.n_buses = n_buses

        # Sub-models (branch-level; heads will be replaced by hybrid heads)
        self.cnn_transformer = CNNTransformerModel(
            in_features=in_features_cnn, n_buses=n_buses
        )
        self.r_gnn = RGNNModel(n_buses=n_buses)

        # Fusion
        self.fusion = GraphInformedFusion()

        # Hybrid output heads (richer: uses FUSED_DIM)
        self.head_detection = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, N_CLASSES_DETECTION)
        )
        self.head_type = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, N_CLASSES_TYPE)
        )
        self.head_phase = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, N_CLASSES_PHASE)
        )
        # Location head: fuse graph-level + per-node spatial context
        self.head_location = nn.Sequential(
            nn.Linear(FUSED_DIM + RGNN_DIM, 128), nn.ReLU(),
            nn.Linear(128, n_buses + 1)
        )

    def forward(self, x_current, x_voltage, edge_index, edge_attr=None):
        """
        Parameters
        ----------
        x_current  : [B, T, N_branches*3]  — current sequences (CNN-T input)
        x_voltage  : [B, T, N_buses, 6]    — voltage phasors (R-GNN input)
        edge_index : [2, E]
        edge_attr  : [E, 3]

        Returns
        -------
        dict with logits for detection, type, phase, location
        """
        # --- Temporal branch ---
        cnn_out = self.cnn_transformer(x_current)
        temporal_feat = cnn_out["feat"]           # [B, CNN_T_DIM]

        # --- Spatial branch ---
        gnn_out = self.r_gnn(x_voltage, edge_index, edge_attr)
        spatial_feat = gnn_out["graph_feat"]      # [B, RGNN_DIM]
        node_feats   = gnn_out["node_feats"]      # [B, N_buses, RGNN_DIM]

        # --- Fusion ---
        fused = self.fusion(temporal_feat, spatial_feat)   # [B, FUSED_DIM]

        # --- Output heads ---
        loc_input = torch.cat([fused, node_feats.mean(dim=1)], dim=-1)

        return {
            "detection": self.head_detection(fused),
            "type":      self.head_type(fused),
            "phase":     self.head_phase(fused),
            "location":  self.head_location(loc_input),
            "fused_feat": fused,
        }

    def freeze_branches(self, freeze_cnn: bool = True, freeze_gnn: bool = True):
        """Optionally freeze sub-model weights for fine-tuning fusion only."""
        for p in self.cnn_transformer.parameters():
            p.requires_grad = not freeze_cnn
        for p in self.r_gnn.parameters():
            p.requires_grad = not freeze_gnn

"""
Hybrid CNN-Transformer + R-GNN Model — v2 with BHAF fusion.
Bidirectional Hierarchical Attention Fusion for Multi-Modal
Power System Fault Diagnostics.

Key fixes:
  - Bug 1: Degenerate cross-attention → BHAF with per-node K/V
  - Bug 5: Location head uses per-node S→T features, not mean-pooled

IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import sub-models by full path to avoid module name collision
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

from fusion import BidirectionalHierarchicalFusion

from config import (
    CNN_T_DIM, RGNN_DIM, FUSED_DIM,
    N_CLASSES_DETECTION, N_CLASSES_TYPE, N_CLASSES_PHASE,
    F_EARLY_DIM, F_MID_DIM,
)


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weighting (Kendall et al.).
    Loss = Σᵢ (1/(2σᵢ²)) · Lᵢ + log(σᵢ)
    """

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        # log(σ²) parameters — initialized to 0 (σ=1)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.log_vars.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class HybridModel(nn.Module):
    """
    BHAF Hybrid model for multi-task fault diagnostics.

    Parameters
    ----------
    in_features_cnn : int  — N_branches × 6 (temporal input)
    n_buses         : int  — number of grid buses
    """

    def __init__(self, in_features_cnn: int, n_buses: int):
        super().__init__()
        self.n_buses = n_buses

        # Sub-models
        self.cnn_transformer = CNNTransformerModel(
            in_features=in_features_cnn, n_buses=n_buses
        )
        self.r_gnn = RGNNModel(n_buses=n_buses)

        # BHAF fusion
        self.fusion = BidirectionalHierarchicalFusion(
            f_early_dim=F_EARLY_DIM, f_mid_dim=F_MID_DIM)

        # Task heads — detection, type, phase use h_fused
        self.head_detection = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, N_CLASSES_DETECTION)
        )
        self.head_type = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, N_CLASSES_TYPE)
        )
        self.head_phase = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, N_CLASSES_PHASE)
        )

        # Location head: uses per-node S→T features + node_feats (Bug 5 fix)
        # Input: h_st [B, N, FUSED_DIM] concat node_feats [B, N, RGNN_DIM]
        loc_in = FUSED_DIM + RGNN_DIM
        self.loc_node_proj = nn.Sequential(
            nn.Linear(loc_in, 64), nn.GELU(),
            nn.Linear(64, 1),  # per-node score
        )
        # "No fault" logit from h_fused
        self.loc_nofault = nn.Linear(FUSED_DIM, 1)

        # Uncertainty-weighted multi-task loss
        self.uncertainty_loss = UncertaintyWeightedLoss(n_tasks=4)

    def forward(self, x_current, x_voltage, edge_index, edge_attr=None):
        """
        Parameters
        ----------
        x_current  : [B, T, N_branches*6]  — current phasor sequences
        x_voltage  : [B, T, N_buses, 6]    — voltage phasors (R-GNN input)
        edge_index : [2, E]
        edge_attr  : [E, 3]

        Returns
        -------
        dict with logits for detection, type, phase, location, plus fused features
        """
        # --- Temporal branch ---
        cnn_out = self.cnn_transformer(x_current)
        temporal_feat = cnn_out["feat"]       # [B, CNN_T_DIM=256]
        f_early = cnn_out["f_early"]          # [B, 208]
        f_mid = cnn_out["f_mid"]              # [B, 392]

        # --- Spatial branch ---
        gnn_out = self.r_gnn(x_voltage, edge_index, edge_attr)
        node_feats = gnn_out["node_feats"]    # [B, N_buses, RGNN_DIM=128]

        # --- BHAF Fusion ---
        h_fused, h_st = self.fusion(
            temporal_feat, node_feats, f_early, f_mid)
        # h_fused: [B, FUSED_DIM=192], h_st: [B, N, FUSED_DIM=192]

        # --- Task heads ---
        # Location: per-node scoring from enriched features
        loc_input = torch.cat([h_st, node_feats], dim=-1)  # [B, N, FUSED_DIM+RGNN_DIM]
        node_scores = self.loc_node_proj(loc_input).squeeze(-1)  # [B, N]
        nofault_score = self.loc_nofault(h_fused)                # [B, 1]
        loc_logits = torch.cat([node_scores, nofault_score], dim=-1)  # [B, N+1]

        return {
            "detection":  self.head_detection(h_fused),
            "type":       self.head_type(h_fused),
            "phase":      self.head_phase(h_fused),
            "location":   loc_logits,
            "fused_feat": h_fused,
            "h_st":       h_st,           # for analysis
        }

    def compute_multitask_loss(self, outputs, labels, loss_fns):
        """
        Compute multi-task loss with uncertainty weighting.

        Parameters
        ----------
        outputs  : dict from forward()
        labels   : (y_det, y_type, y_ph, y_loc) tensors
        loss_fns : (ce_det, fl_type, fl_phase, ce_loc) loss functions
        """
        ce_det, fl_type, fl_phase, ce_loc = loss_fns
        y_det, y_type, y_ph, y_loc = labels

        losses = [
            ce_det(outputs["detection"], y_det),
            fl_type(outputs["type"], y_type),
            fl_phase(outputs["phase"], y_ph),
            ce_loc(outputs["location"], y_loc),
        ]
        return self.uncertainty_loss(losses)

    def freeze_branches(self, freeze_cnn: bool = True, freeze_gnn: bool = True):
        """Freeze sub-model weights for fine-tuning fusion only."""
        for p in self.cnn_transformer.parameters():
            p.requires_grad = not freeze_cnn
        for p in self.r_gnn.parameters():
            p.requires_grad = not freeze_gnn

    def load_pretrained_branches(self, cnn_ckpt_path: Path = None,
                                  gnn_ckpt_path: Path = None):
        """Load pre-trained branch weights from Phase 1/2 checkpoints."""
        if cnn_ckpt_path and cnn_ckpt_path.exists():
            ckpt = torch.load(cnn_ckpt_path, map_location="cpu")
            self.cnn_transformer.load_state_dict(ckpt["model_state"], strict=False)
        if gnn_ckpt_path and gnn_ckpt_path.exists():
            ckpt = torch.load(gnn_ckpt_path, map_location="cpu")
            self.r_gnn.load_state_dict(ckpt["model_state"], strict=False)

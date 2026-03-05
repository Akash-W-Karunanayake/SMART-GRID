"""
Fault Detection Service — loads the trained CNN-Transformer + R-GNN hybrid model
and runs inference on voltage/current phasor sequences.

Design decisions (from Phase 1):
  - Q8: Always show predictions regardless of confidence
  - Q9: Frontend receives full per-class probability vectors
  - Q17: Option B — sub-cycle snapshots for accurate inference
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)

# ── Research module paths ──────────────────────────────────────────
_research_root = settings.BASE_DIR / "research"

# ── Label maps (mirror research/01_dataset_generation/config.py) ────
DETECTION_LABELS = {0: "Normal", 1: "Fault"}
TYPE_LABELS = {0: "Normal", 1: "LG", 2: "LL", 3: "LLG", 4: "LLL", 5: "HIF"}
PHASE_LABELS = {0: "none", 1: "A", 2: "B", 3: "C", 4: "AB", 5: "BC", 6: "CA", 7: "ABC"}


@dataclass
class FaultPrediction:
    """Structured model output for one inference call."""
    is_fault: bool
    detection_confidence: float                     # P(Fault)

    fault_type: str                                 # e.g. "LG"
    type_probabilities: Dict[str, float]            # {"Normal": 0.02, "LG": 0.91, ...}

    fault_phase: str                                # e.g. "A"
    phase_probabilities: Dict[str, float]           # {"none": 0.01, "A": 0.88, ...}

    fault_location_bus: str                         # most-likely bus name
    location_probabilities: Dict[str, float]        # {bus_name: prob, ...}

    step_injected: Optional[int] = None             # simulation step when fault was injected
    step_detected: Optional[int] = None             # simulation step when detection ran


class FaultDetectionService:
    """Loads the hybrid model and runs inference on phasor sequences."""

    def __init__(self):
        self._model = None
        self._normalizer = None
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_attr: Optional[torch.Tensor] = None
        self._bus_names: list = []
        self._n_buses: int = 0
        self._in_features_cnn: int = 0
        self._device = torch.device("cpu")
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Load model checkpoint, normalizer, and graph structure."""
        try:
            model_path = _research_root / "models" / "hybrid" / "best_model.pt"
            norm_path = _research_root / "datasets" / "processed" / "normalizer.pkl"
            graph_path = _research_root / "datasets" / "graph" / "graph_structure.pkl"

            # ── Import + instantiate research model ──
            # The research code does bare `from config import ...` at both
            # import time AND inside __init__. We must keep the research
            # config.py active for the entire import→instantiate→load_state
            # sequence, then restore the backend config afterwards.
            hybrid_dir = str(_research_root / "05_hybrid_model")
            preproc_dir = str(_research_root / "02_data_preprocessing")

            # Save and evict the backend's config module
            saved_config = sys.modules.pop("config", None)
            # Evict stale research modules so they reimport cleanly
            for mod_name in list(sys.modules):
                if mod_name in ("model", "fusion", "normalizer"):
                    sys.modules.pop(mod_name, None)

            sys.path.insert(0, hybrid_dir)
            sys.path.insert(0, preproc_dir)

            try:
                from model import HybridModel
                from normalizer import SequenceNormalizer

                # ── Checkpoint ──
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                self._n_buses = ckpt["n_buses"]
                self._in_features_cnn = ckpt["in_features_cnn"]

                self._model = HybridModel(
                    in_features_cnn=self._in_features_cnn,
                    n_buses=self._n_buses,
                )
                self._model.load_state_dict(ckpt["model_state"])
                self._model.to(self._device)
                self._model.eval()
                logger.info(f"Hybrid model loaded (epoch={ckpt['epoch']}, "
                            f"n_buses={self._n_buses}, cnn_feat={self._in_features_cnn})")
            finally:
                # Clean up sys.path
                if hybrid_dir in sys.path:
                    sys.path.remove(hybrid_dir)
                if preproc_dir in sys.path:
                    sys.path.remove(preproc_dir)
                # Remove the research config from cache and restore backend's
                sys.modules.pop("config", None)
                if saved_config is not None:
                    sys.modules["config"] = saved_config

            # ── Normalizer ──
            self._normalizer = SequenceNormalizer.load(norm_path)
            logger.info("Normalizer loaded")

            # ── Graph ──
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)
            self._bus_names = graph["bus_names"]
            self._edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
            self._edge_attr = torch.tensor(graph["edge_attr"], dtype=torch.float32)
            logger.info(f"Graph loaded: {len(self._bus_names)} buses, "
                        f"{self._edge_index.shape[1]} directed edges")

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load fault detection model: {e}", exc_info=True)
            self._loaded = False
            return False

    def run_inference(self,
                      voltage_seq: np.ndarray,
                      current_seq: np.ndarray) -> FaultPrediction:
        """
        Run the hybrid model on a single observation window.

        Args:
            voltage_seq: [T, N_buses, 6] raw voltage phasors
            current_seq: [T, N_branches, 6] raw current phasors

        Returns:
            FaultPrediction dataclass with all outputs.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # ── Normalize ──
        v_norm = self._normalizer.transform_v(voltage_seq)   # [T, 80, 6]
        i_norm = self._normalizer.transform_i(current_seq)   # [T, 87, 6]

        # ── Reshape for model ──
        T = v_norm.shape[0]
        x_voltage = torch.tensor(v_norm, dtype=torch.float32).unsqueeze(0)   # [1, T, 80, 6]
        x_current_3d = torch.tensor(i_norm, dtype=torch.float32)              # [T, 87, 6]
        x_current = x_current_3d.reshape(T, -1).unsqueeze(0)                  # [1, T, 522]

        # ── Inference ──
        with torch.no_grad():
            out = self._model(x_current, x_voltage, self._edge_index, self._edge_attr)

        # ── Decode outputs ──
        det_probs = F.softmax(out["detection"][0], dim=-1).numpy()     # [2]
        typ_probs = F.softmax(out["type"][0], dim=-1).numpy()          # [6]
        pha_probs = F.softmax(out["phase"][0], dim=-1).numpy()         # [8]
        loc_probs = F.softmax(out["location"][0], dim=-1).numpy()      # [81]

        is_fault = bool(det_probs.argmax() == 1)
        detection_confidence = float(det_probs[1])

        type_idx = int(typ_probs.argmax())
        phase_idx = int(pha_probs.argmax())
        loc_idx = int(loc_probs.argmax())

        # Build named probability dicts
        type_prob_dict = {TYPE_LABELS[i]: round(float(typ_probs[i]), 4) for i in range(6)}
        phase_prob_dict = {PHASE_LABELS[i]: round(float(pha_probs[i]), 4) for i in range(8)}

        # Location: first N_buses entries = bus probs, last = no-fault
        loc_prob_dict = {}
        for i, bname in enumerate(self._bus_names):
            loc_prob_dict[bname] = round(float(loc_probs[i]), 4)
        loc_prob_dict["no_fault"] = round(float(loc_probs[self._n_buses]), 4)

        # Most likely bus (ignoring no_fault index)
        if loc_idx < self._n_buses:
            fault_bus = self._bus_names[loc_idx]
        else:
            fault_bus = "none"

        return FaultPrediction(
            is_fault=is_fault,
            detection_confidence=round(detection_confidence, 4),
            fault_type=TYPE_LABELS[type_idx],
            type_probabilities=type_prob_dict,
            fault_phase=PHASE_LABELS[phase_idx],
            phase_probabilities=phase_prob_dict,
            fault_location_bus=fault_bus,
            location_probabilities=loc_prob_dict,
        )


# Singleton
fault_detection_service = FaultDetectionService()

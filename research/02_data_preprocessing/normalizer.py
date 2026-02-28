"""
Normalizer — Z-score normalization for voltage and current sequences.
Fit statistics on training set only; applied to val/test consistently.

Updated for 6-feature currents [Ia_mag, Ia_ang, Ib_mag, Ib_ang, Ic_mag, Ic_ang].

IT22577924 — Karunanayake K.P.A.W.
"""
import pickle
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

EPS = 1e-8   # prevent division by zero


class SequenceNormalizer:
    """
    Per-feature Z-score normalizer for [T, N, F] sequence arrays.

    Voltage arrays  : [T, N_buses, 6]     — [mag, ang] × 3 phases
    Current arrays  : [T, N_branches, 6]  — [mag, ang] × 3 phases

    Angle features (columns 1, 3, 5) are left as-is (pass-through)
    because angles are already bounded — only magnitudes are Z-scored.
    """

    def __init__(self):
        self.v_mean: np.ndarray = None   # [N_buses, 6]
        self.v_std:  np.ndarray = None
        self.i_mean: np.ndarray = None   # [N_branches, 6]
        self.i_std:  np.ndarray = None
        self._fitted = False

    def fit(self, v_list: list, i_list: list) -> "SequenceNormalizer":
        """
        Compute mean/std from lists of voltage and current arrays.

        Parameters
        ----------
        v_list : list of np.ndarray [T, N_buses, 6]
        i_list : list of np.ndarray [T, N_branches, 6]
        """
        # Stack along sample and time axes → [S*T, N, F]
        V = np.concatenate(v_list, axis=0)   # [S*T, N_buses, 6]
        I = np.concatenate(i_list, axis=0)   # [S*T, N_branches, 6]

        # Voltage normalization (skip angle columns)
        self.v_mean = V.mean(axis=0)         # [N_buses, 6]
        self.v_std  = V.std(axis=0) + EPS
        angle_cols = [1, 3, 5]
        self.v_mean[:, angle_cols] = 0.0
        self.v_std[:, angle_cols]  = 1.0

        # Current normalization (skip angle columns)
        self.i_mean = I.mean(axis=0)         # [N_branches, 6]
        self.i_std  = I.std(axis=0) + EPS
        self.i_mean[:, angle_cols] = 0.0
        self.i_std[:, angle_cols]  = 1.0

        self._fitted = True
        logger.info(f"Normalizer fitted: V mean range [{self.v_mean.min():.4f}, "
                    f"{self.v_mean.max():.4f}], "
                    f"I mean range [{self.i_mean.min():.4f}, {self.i_mean.max():.4f}]")
        return self

    def transform_v(self, V: np.ndarray) -> np.ndarray:
        """Normalize voltage array [T, N_buses, 6]."""
        self._check_fitted()
        return (V - self.v_mean) / self.v_std

    def transform_i(self, I: np.ndarray) -> np.ndarray:
        """Normalize current array [T, N_branches, 6]."""
        self._check_fitted()
        return (I - self.i_mean) / self.i_std

    def inverse_v(self, V_norm: np.ndarray) -> np.ndarray:
        return V_norm * self.v_std + self.v_mean

    def inverse_i(self, I_norm: np.ndarray) -> np.ndarray:
        return I_norm * self.i_std + self.i_mean

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"v_mean": self.v_mean, "v_std": self.v_std,
                         "i_mean": self.i_mean, "i_std": self.i_std}, f)
        logger.info(f"Normalizer saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "SequenceNormalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        n = cls()
        n.v_mean, n.v_std = d["v_mean"], d["v_std"]
        n.i_mean, n.i_std = d["i_mean"], d["i_std"]
        n._fitted = True
        return n

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
